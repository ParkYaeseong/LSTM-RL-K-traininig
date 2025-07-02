# -*- coding: utf-8 -*-
# 파일명: websocket_client.py (Kiwoom 최종본 - 기본 주문 로직 구현)
# 실행 환경: 64비트 Python (myenv)
# 필요 라이브러리: pip install websockets==11.0.3 tensorflow pandas numpy requests pandas_ta joblib httpx

import logging
import logging.handlers
import asyncio
import websockets # 버전 11.0.3 권장
import json
import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque
import requests # 동기 REST API 호출용
# import httpx     # (현재 주문 로직에서는 미사용)
import traceback
import pprint
import functools
from datetime import datetime, timedelta
import joblib
import pandas_ta as ta

# --- 설정 (키움증권 API 기준) ---
KIWOOM_TOKEN = "oaAV7Ho-_HNCVprNH1490afqY5ES5MRGGmp2brdUoAxab2kfHu38aPu8RKqRtoe8fi_fob9l_NAOLF7niIJ4dQ" # 웹소켓 로그인 및 주문용 토큰 (이전 로그 토큰 사용)
KIWOOM_ACCOUNT_NO_STR = "8102624701" # 실제 계좌번호 사용
KIWOOM_APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss" # (현재 주문 로직에서는 미사용)
KIWOOM_APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw" # (현재 주문 로직에서는 미사용)
KIWOOM_USER_ID = "" # 필요 시 사용자 ID 설정

IS_MOCK_TRADING = True # 모의투자 여부

# --- 주문 로직 관련 설정 (★★★ 튜닝 및 검증 필수 ★★★) ---
ORDER_QUANTITY = 1  # 기본 주문 수량
BUY_THRESHOLD = 0.02 # 매수 신호 임계값 (예측값이 이 값보다 클 때 매수 시도)
SELL_THRESHOLD = -0.01 # 매도 신호 임계값 (예측값이 이 값보다 작을 때 매도 시도)
STOP_LOSS_PCT = -0.05 # 손절매 비율 (매입가 대비 -5% 하락 시 매도)

MODEL_SAVE_PATH = r'G:\내 드라이브\lstm_models_per_stock_v2'
SCALER_SAVE_PATH = r'G:\내 드라이브\processed_stock_data_full_v2\scalers'

TIME_STEPS = 20
FEATURE_COLUMNS_NORM = [
    'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 'amount_norm',
    'SMA_5_norm', 'SMA_20_norm', 'SMA_60_norm', 'SMA_120_norm', 'RSI_14_norm',
    'MACD_12_26_9_norm', 'MACDs_12_26_9_norm', 'MACDh_12_26_9_norm',
    'BBL_20_2.0_norm', 'BBM_20_2.0_norm', 'BBU_20_2.0_norm', 'BBB_20_2.0_norm', 'BBP_20_2.0_norm',
    'ATRr_14_norm', 'OBV_norm', 'STOCHk_14_3_3_norm', 'STOCHd_14_3_3_norm',
    'PBR_norm', 'PER_norm', 'USD_KRW_norm', 'is_month_end'
]
NUM_FEATURES = len(FEATURE_COLUMNS_NORM)

if IS_MOCK_TRADING:
    KIWOOM_WEBSOCKET_URL = "wss://mockapi.kiwoom.com:10000/api/dostk/websocket"
    KIWOOM_REST_BASE_URL = "https://mockapi.kiwoom.com"
    print(">> 키움 모의투자 환경으로 설정되었습니다.")
else:
    KIWOOM_WEBSOCKET_URL = "wss://openapi.kiwoom.com:10000/api/dostk/websocket"
    KIWOOM_REST_BASE_URL = "https://openapi.kiwoom.com"
    print(">> 키움 실전투자 환경으로 설정되었습니다.")

KIWOOM_API_URL_ORDER = f"{KIWOOM_REST_BASE_URL}/api/dostk/ordr"
KIWOOM_ORDER_TR_ID_BUY = "kt10000"
KIWOOM_ORDER_TR_ID_SELL = "kt10001"

REGISTER_LIST = [{
    "item": ["001450"],
    "type": ["0B"],
    "fids": ["20", "10", "15"] # 클라이언트 처리용 FID (REG 요청에는 미포함)
}]

class WebSocketClient:
    def __init__(self, socket_uri, token, account_no, app_key, app_secret, user_id,
                 is_dev_mode=True, monitored_tickers_list=None,
                 custom_ta_params=None, model_dir=MODEL_SAVE_PATH, scaler_dir=SCALER_SAVE_PATH,
                 log_file_path="trading_log_kiwoom_final.log"):

        # --- 로거 설정 ---
        self.logger = logging.getLogger("WebSocketClient")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler(); ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'); ch.setFormatter(ch_formatter); self.logger.addHandler(ch)
            try:
                log_dir = os.path.dirname(log_file_path);
                if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
                fh = logging.handlers.TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=30, encoding='utf-8'); fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'); fh.setFormatter(fh_formatter); self.logger.addHandler(fh)
            except Exception as e: print(f"파일 로거 설정 실패: {e}")
        self.logger.info(f"WebSocketClient 초기화 시작. 모의투자: {IS_MOCK_TRADING}")

        # --- 속성 초기화 ---
        self.uri = socket_uri
        self.token = token
        self.account_no = account_no
        self.app_key = app_key
        self.app_secret = app_secret
        self.user_id = user_id

        self.websocket = None; self.connected = False; self.keep_running = True
        self.login_success = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        self.bar_interval_minutes = 1; self.current_bars = {}; self.completed_bars = {}
        self.feature_sequences = {}; self.scalers = {}; self.models = {}; self.trade_triggered = {} # self.trade_triggered는 현재 로직에서 사용 안 함
        self.feature_order = FEATURE_COLUMNS_NORM; self.model_path_root = model_dir; self.scaler_path_root = scaler_dir
        default_ta_params = {'sma_lengths': [5, 20, 60, 120], 'rsi_length': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'bbands_length': 20, 'bbands_std': 2.0, 'atr_length': 14, 'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth_k': 3}
        self.ta_params = custom_ta_params if custom_ta_params else default_ta_params
        self.tickers_to_monitor = monitored_tickers_list if monitored_tickers_list else [item["item"][0] for item in REGISTER_LIST if item.get("item")]

        # === 포지션 관리 딕셔너리 추가 ===
        self.positions = {} # key: ticker, value: {'quantity': int, 'entry_price': float}

        for ticker in self.tickers_to_monitor:
            all_lengths = self.ta_params.get('sma_lengths', []) + [self.ta_params.get('rsi_length', 0)+1, self.ta_params.get('bbands_length', 0), self.ta_params.get('macd_slow', 0)+self.ta_params.get('macd_signal', 0)+1, self.ta_params.get('atr_length',0)+1, self.ta_params.get('stoch_k',0)+self.ta_params.get('stoch_d',0)+self.ta_params.get('stoch_smooth_k', 0)+1]
            max_lookback = max(all_lengths) if all_lengths else 0
            self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + max_lookback + 150) # 충분한 과거 데이터 저장
            self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS);
            if self.model_path_root: self.load_model(ticker)
            if self.scaler_path_root: self.load_scalers(ticker)
        self.logger.info(f"WebSocketClient 초기화 완료. 모니터링 종목: {self.tickers_to_monitor}")
        self.logger.info(f"주문 설정: 수량={ORDER_QUANTITY}, 매수 Thresh={BUY_THRESHOLD}, 매도 Thresh={SELL_THRESHOLD}, 손절={STOP_LOSS_PCT*100:.1f}%")


    async def connect(self):
        self.logger.info(f"Kiwoom WebSocket 서버 연결 시도: {self.uri}")
        try:
            # ping_interval을 설정하여 클라이언트가 주기적으로 서버에 PING 전송 (서버 Heartbeat 요구사항 대응에 간접 도움)
            self.websocket = await websockets.connect(self.uri, ping_interval=25, ping_timeout=20, close_timeout=10)
            self.connected = True; self.logger.info("Kiwoom WebSocket 서버 연결 성공."); self.reconnect_attempts = 0
            if self.token:
                login_packet = {'trnm': 'LOGIN', 'token': self.token}
                if self.user_id: login_packet['user_id'] = self.user_id
                self.logger.info(f"Kiwoom LOGIN 패킷 전송 시도: {login_packet}")
                await self.send_message(login_packet)
            else: self.logger.warning("WebSocket LOGIN 토큰 미설정.")
        except Exception as e: self.logger.error(f'Kiwoom WebSocket 연결 오류: {e}'); self.connected = False; self.login_success = False

    async def send_message(self, message):
        if self.connected and self.websocket:
            if not isinstance(message, str): message_str = json.dumps(message)
            else: message_str = message
            try:
                await self.websocket.send(message_str)
                # self.logger.debug(f"Kiwoom 전송: {message_str}") # 디버깅 시 활성화
            except websockets.exceptions.ConnectionClosed as e: self.logger.error(f"메시지 전송 중 연결 끊김: {e}"); self.connected = False; self.login_success = False
            except Exception as e: self.logger.error(f"메시지 전송 오류: {e}")
        else: self.logger.warning("WebSocket 미연결 상태")

    def load_model(self, ticker):
        if ticker in self.models and self.models[ticker] is not None: return self.models[ticker]
        model_file_path = os.path.join(self.model_path_root, f"{ticker}.keras")
        if os.path.exists(model_file_path):
            try:
                self.models[ticker] = tf.keras.models.load_model(model_file_path, compile=False)
                self.logger.info(f"모델 로딩 성공: {ticker}")
                return self.models[ticker]
            except Exception as e:
                self.logger.error(f"모델 로딩 실패 ({ticker}): {e}")
                self.models[ticker] = None
                return None
        else:
            self.logger.warning(f"모델 파일 없음: {model_file_path}")
            self.models[ticker] = None
            return None

    def load_scalers(self, ticker):
        if ticker in self.scalers and self.scalers[ticker]: return self.scalers[ticker]
        scaler_file_path = os.path.join(self.scaler_path_root, f'{ticker}_scalers.joblib')
        if os.path.exists(scaler_file_path):
            try:
                self.scalers[ticker] = joblib.load(scaler_file_path)
                self.logger.info(f"스케일러 로딩 성공 ({ticker}): {len(self.scalers[ticker])}개")
                return self.scalers[ticker]
            except Exception as e:
                self.logger.error(f"스케일러 로딩 실패 ({ticker}): {e}")
                self.scalers[ticker] = {}
                return {}
        else:
            self.logger.warning(f"스케일러 파일 없음 ({ticker}): {scaler_file_path}")
            self.scalers[ticker] = {}
            return {}

    def calculate_features(self, ticker):
        # ... (이전 코드와 동일, 피처 계산 로직) ...
        all_lengths = self.ta_params.get('sma_lengths', []) + [self.ta_params.get('rsi_length', 0)+1, self.ta_params.get('bbands_length', 0), self.ta_params.get('macd_slow', 0)+self.ta_params.get('macd_signal', 0)+1, self.ta_params.get('atr_length',0)+1, self.ta_params.get('stoch_k',0)+self.ta_params.get('stoch_d',0)+self.ta_params.get('stoch_smooth_k', 0)+1]
        min_data_length = (max(all_lengths) if all_lengths else 0) + 15 # 충분한 데이터 확보
        if ticker not in self.completed_bars or len(self.completed_bars[ticker]) < min_data_length: self.logger.debug(f"({ticker}) 지표 계산 데이터 부족: {len(self.completed_bars.get(ticker, []))} / {min_data_length}"); return None
        valid_bars_list = [bar for bar in self.completed_bars[ticker] if all(pd.notna(bar.get(k)) for k in ['open', 'high', 'low', 'close', 'volume'])]
        if len(valid_bars_list) < min_data_length: self.logger.warning(f"({ticker}) 유효 봉 데이터 부족 (NaN 제외): {len(valid_bars_list)} / {min_data_length}"); return None
        df_bars = pd.DataFrame(valid_bars_list).set_index('time')
        if not all(col in df_bars.columns for col in ['open', 'high', 'low', 'close', 'volume']): self.logger.error(f"({ticker}) OHLCV 컬럼 없음."); return None
        self.logger.debug(f"({ticker}) 지표 계산 시작. 데이터: {len(df_bars)}개")
        try:
            custom_ta_list = [ {"kind": "sma", "length": l} for l in self.ta_params['sma_lengths']] + [{"kind": "rsi", "length": self.ta_params['rsi_length']}, {"kind": "macd", "fast": self.ta_params['macd_fast'], "slow": self.ta_params['macd_slow'], "signal": self.ta_params['macd_signal']}, {"kind": "bbands", "length": self.ta_params['bbands_length'], "std": float(self.ta_params['bbands_std'])}, {"kind": "atr", "length": self.ta_params['atr_length']}, {"kind": "obv"}, {"kind": "stoch", "k": self.ta_params['stoch_k'], "d": self.ta_params['stoch_d'], "smooth_k": self.ta_params.get('stoch_smooth_k', 3)}]
            df_bars.ta.strategy(ta.Strategy(name="TAStrategy", ta=custom_ta_list))
        except Exception as e: self.logger.error(f"({ticker}) 지표 계산 오류: {e}"); traceback.print_exc(); return None
        latest_features = {}; latest_bar = df_bars.iloc[-1]
        for col in ['open', 'high', 'low', 'close', 'volume']: latest_features[col] = latest_bar.get(col)
        latest_features['amount'] = latest_bar.get('amount', latest_bar['close'] * latest_bar['volume'] if pd.notna(latest_bar.get('close')) and pd.notna(latest_bar.get('volume')) else 0)
        for length in self.ta_params['sma_lengths']: latest_features[f'SMA_{length}'] = latest_bar.get(f'SMA_{length}')
        latest_features[f'RSI_{self.ta_params["rsi_length"]}'] = latest_bar.get(f'RSI_{self.ta_params["rsi_length"]}')
        macd_suffix = f'{self.ta_params["macd_fast"]}_{self.ta_params["macd_slow"]}_{self.ta_params["macd_signal"]}'
        latest_features[f'MACD_{macd_suffix}'] = latest_bar.get(f'MACD_{macd_suffix}'); latest_features[f'MACDs_{macd_suffix}'] = latest_bar.get(f'MACDS_{macd_suffix}'); latest_features[f'MACDh_{macd_suffix}'] = latest_bar.get(f'MACDH_{macd_suffix}')
        bb_std_str = f"{float(self.ta_params['bbands_std']):.1f}"; bb_suffix = f'{self.ta_params["bbands_length"]}_{bb_std_str}'
        for b_col in ['BBL', 'BBM', 'BBU', 'BBB', 'BBP']: latest_features[f'{b_col}_{bb_suffix}'] = latest_bar.get(f'{b_col}_{bb_suffix}')
        atr_df_col = f'ATRr_{self.ta_params["atr_length"]}'; latest_features['ATRr_14'] = latest_bar.get(atr_df_col if atr_df_col in df_bars.columns else f'ATR_{self.ta_params["atr_length"]}') # pandas-ta 0.3.14 uses ATR_ not ATRr_
        latest_features['OBV'] = latest_bar.get('OBV')
        stoch_suffix_df = f'{self.ta_params["stoch_k"]}_{self.ta_params["stoch_d"]}_{self.ta_params.get("stoch_smooth_k",3)}'
        latest_features['STOCHk_14_3_3'] = latest_bar.get(f'STOCHk_{stoch_suffix_df}'); latest_features['STOCHd_14_3_3'] = latest_bar.get(f'STOCHd_{stoch_suffix_df}')
        latest_features['PBR'] = 0.0; latest_features['PER'] = 0.0; latest_features['USD_KRW'] = 0.0 # TODO: 실제 값으로 채워야 함 (예: API 조회)
        latest_features['is_month_end'] = 1 if latest_bar.name.is_month_end else 0

        feature_vector = np.full(NUM_FEATURES, np.nan)
        for i, col_norm_name in enumerate(self.feature_order):
            original_name_key = col_norm_name[:-5] if col_norm_name.endswith('_norm') else col_norm_name
            # pandas-ta 컬럼명 매핑 (버전/설정에 따라 다를 수 있음)
            if original_name_key == 'MACD_12_26_9': original_name_key = f'MACD_{macd_suffix}'
            elif original_name_key == 'MACDs_12_26_9': original_name_key = f'MACDs_{macd_suffix}'
            elif original_name_key == 'MACDh_12_26_9': original_name_key = f'MACDh_{macd_suffix}'
            elif original_name_key == 'BBL_20_2.0': original_name_key = f'BBL_{bb_suffix}'
            elif original_name_key == 'BBM_20_2.0': original_name_key = f'BBM_{bb_suffix}'
            elif original_name_key == 'BBU_20_2.0': original_name_key = f'BBU_{bb_suffix}'
            elif original_name_key == 'BBB_20_2.0': original_name_key = f'BBB_{bb_suffix}'
            elif original_name_key == 'BBP_20_2.0': original_name_key = f'BBP_{bb_suffix}'
            elif original_name_key == 'STOCHk_14_3_3': original_name_key = f'STOCHk_{stoch_suffix_df}'
            elif original_name_key == 'STOCHd_14_3_3': original_name_key = f'STOCHd_{stoch_suffix_df}'
            elif original_name_key == 'ATRr_14': original_name_key = f'ATRr_{self.ta_params["atr_length"]}'
            elif original_name_key == 'RSI_14': original_name_key = f'RSI_{self.ta_params["rsi_length"]}'

            value = latest_features.get(original_name_key)
            if value is not None and pd.notna(value): feature_vector[i] = value # NaN 값 제외
            elif col_norm_name == 'is_month_end': feature_vector[i] = latest_features.get('is_month_end', 0) # 기본값 처리
            # else: # NaN 값 발생 시 로깅 (디버깅용)
            #     self.logger.debug(f"({ticker}) NaN in feature calculation for {original_name_key} (col: {col_norm_name})")


        if np.isnan(feature_vector).any():
            nan_indices = np.where(np.isnan(feature_vector))[0]
            nan_features = [self.feature_order[i] for i in nan_indices]
            self.logger.warning(f"({ticker}) 최종 피처 NaN 포함 ({len(nan_features)}개). 예측 건너뜀. NaN 피처: {nan_features}. Vec(first 10): {feature_vector[:10]}")
            return None
        self.logger.debug(f"({ticker}) 계산된 원본 피처 (첫 5개): {np.nan_to_num(feature_vector[:5], nan=-999.0)}")
        return feature_vector


    def normalize_features(self, ticker, feature_vector):
        # ... (이전 코드와 동일, 정규화 로직) ...
        scalers_for_ticker = self.load_scalers(ticker)
        if not scalers_for_ticker: self.logger.warning(f"({ticker}) 스케일러 없음. 정규화 불가."); return None
        normalized_vector = np.zeros_like(feature_vector, dtype=float)
        for i, col_norm_name in enumerate(self.feature_order):
            raw_value = feature_vector[i] # calculate_features에서 NaN 처리 후 호출됨
            # if np.isnan(raw_value): normalized_vector[i] = 0.0; continue # calculate_features에서 NaN 처리 했으므로 불필요

            if col_norm_name.endswith('_norm'):
                original_name = col_norm_name[:-5]; scaler = scalers_for_ticker.get(original_name)
                if scaler:
                    try:
                        # 스케일러 입력 전 NaN/inf 재확인 및 처리
                        if not np.isfinite(raw_value):
                           self.logger.warning(f"({ticker}) '{original_name}' 정규화 전 유효하지 않은 값 ({raw_value}). 0.0 처리.");
                           normalized_vector[i] = 0.0
                        else:
                            normalized_vector[i] = scaler.transform([[raw_value]])[0][0]
                            # 정규화 후 값 범위 체크 (옵션)
                            # if not (-5.0 < normalized_vector[i] < 5.0):
                            #     self.logger.warning(f"({ticker}) '{original_name}' 정규화 값 범위 의심: {normalized_vector[i]}")
                    except Exception as e:
                         self.logger.error(f"({ticker}) '{original_name}' 정규화 오류: {e}. 원본값: {raw_value}. 0.0 처리.");
                         normalized_vector[i] = 0.0
                else:
                    self.logger.warning(f"({ticker}) '{original_name}' 스케일러 없음. 0.0 처리.");
                    normalized_vector[i] = 0.0 # 스케일러 없으면 0.0으로 처리 (학습 시와 일관성)
            else: # '_norm' 아닌 경우 (예: is_month_end는 이미 0 또는 1)
                normalized_vector[i] = raw_value

        if np.isnan(normalized_vector).any(): self.logger.error(f"({ticker}) 정규화 후 NaN 발생. 예측 건너뜀."); return None
        self.logger.debug(f"({ticker}) 정규화된 피처 (첫 5개): {normalized_vector[:5]}")
        return normalized_vector


    async def process_new_bar(self, ticker, bar_data):
        if not hasattr(self, 'logger'): print(f"CRITICAL: Logger not in process_new_bar for {ticker}"); return
        try:
            self.logger.info(f"1분봉 완성 ({ticker}): T={bar_data.get('time')}, O={bar_data.get('open'):.0f}, H={bar_data.get('high'):.0f}, L={bar_data.get('low'):.0f}, C={bar_data.get('close'):.0f}, V={bar_data.get('volume'):.0f}")
            if ticker not in self.completed_bars:
                # ... (deque 생성 로직 동일) ...
                 all_lengths = self.ta_params.get('sma_lengths', []) + [self.ta_params.get('rsi_length', 0)+1, self.ta_params.get('bbands_length', 0), self.ta_params.get('macd_slow', 0)+self.ta_params.get('macd_signal', 0)+1, self.ta_params.get('atr_length',0)+1, self.ta_params.get('stoch_k',0)+self.ta_params.get('stoch_d',0)+self.ta_params.get('stoch_smooth_k', 0)+1]
                 max_lookback = max(all_lengths) if all_lengths else 0
                 self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + max_lookback + 150)
            self.completed_bars[ticker].append(bar_data)

            raw_feature_vector = self.calculate_features(ticker)
            if raw_feature_vector is None: return # 지표 계산 불가 시 종료
            normalized_features = self.normalize_features(ticker, raw_feature_vector)
            if normalized_features is None: return # 정규화 불가 시 종료

            if ticker not in self.feature_sequences: self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS)
            self.feature_sequences[ticker].append(normalized_features)
            self.logger.debug(f"({ticker}) 피처 시퀀스 업데이트. 크기: {len(self.feature_sequences[ticker])}/{TIME_STEPS}")

            # === 모델 예측 및 주문 로직 실행 ===
            if len(self.feature_sequences[ticker]) == TIME_STEPS:
                model = self.load_model(ticker)
                if model:
                    input_seq = np.expand_dims(np.array(list(self.feature_sequences[ticker])).astype(np.float32), axis=0)
                    if input_seq.shape == (1, TIME_STEPS, NUM_FEATURES):
                        try:
                            # 1. 모델 예측 수행
                            prediction = model.predict(input_seq, verbose=0)
                            predicted_norm_value = prediction[0][0] # 모델 출력 구조에 따라 조정 필요
                            self.logger.info(f" ★★★ 예측 ({ticker}): {predicted_norm_value:.6f} ★★★")

                            # 2. 포지션 상태 확인
                            current_price = bar_data['close'] # 현재가 (해당 봉의 종가)
                            holding_info = self.positions.get(ticker) # 보유 정보 가져오기

                            # 3. 주문 로직 실행
                            if holding_info: # 해당 종목 보유 중인 경우 (매도 또는 유지 결정)
                                entry_price = holding_info['entry_price']
                                quantity = holding_info['quantity']
                                current_return = (current_price - entry_price) / entry_price if entry_price != 0 else 0

                                # 3-1. 손절매 조건 확인 [cite: 107, 212]
                                if current_return <= STOP_LOSS_PCT:
                                    self.logger.info(f"!!! 손절매 실행 ({ticker}) !!! 목표: {STOP_LOSS_PCT*100:.1f}%, 현재: {current_return*100:.1f}%, 매입가: {entry_price:.0f}, 현재가: {current_price:.0f}")
                                    # 시장가 매도 주문 [cite: 214]
                                    order_result = await self.send_order(order_type='sell', stock_code=ticker, quantity=quantity, trde_tp="03") # 시장가 매도
                                    if order_result and order_result.get('return_code') == 0:
                                        # 주문 성공 시 포지션 정리 (실제 체결은 ChejanData로 확인해야 더 정확)
                                        del self.positions[ticker]
                                        self.logger.info(f"({ticker}) 포지션 정리 (손절매 주문 성공)")
                                    else:
                                        self.logger.error(f"({ticker}) 손절매 주문 실패 또는 오류: {order_result}")

                                # 3-2. 예측 기반 매도 신호 확인 (손절매 조건 미충족 시)
                                elif predicted_norm_value < SELL_THRESHOLD:
                                    self.logger.info(f"--- 매도 신호 ({ticker}) --- 예측값: {predicted_norm_value:.4f} < 임계값: {SELL_THRESHOLD}, 수익률: {current_return*100:.1f}%")
                                    # 시장가 매도 주문
                                    order_result = await self.send_order(order_type='sell', stock_code=ticker, quantity=quantity, trde_tp="03")
                                    if order_result and order_result.get('return_code') == 0:
                                        del self.positions[ticker]
                                        self.logger.info(f"({ticker}) 포지션 정리 (매도 신호 주문 성공)")
                                    else:
                                        self.logger.error(f"({ticker}) 매도 신호 주문 실패 또는 오류: {order_result}")
                                else:
                                    # 매도 조건 미충족 시 보유 유지
                                    self.logger.info(f"--- 보유 유지 ({ticker}) --- 예측값: {predicted_norm_value:.4f}, 수익률: {current_return*100:.1f}%")

                            else: # 미보유 중인 경우 (매수 또는 대기 결정)
                                # 3-3. 예측 기반 매수 신호 확인
                                if predicted_norm_value > BUY_THRESHOLD:
                                    self.logger.info(f"+++ 매수 신호 ({ticker}) +++ 예측값: {predicted_norm_value:.4f} > 임계값: {BUY_THRESHOLD}")
                                    # 시장가 매수 주문 [cite: 193]
                                    order_result = await self.send_order(order_type='buy', stock_code=ticker, quantity=ORDER_QUANTITY, trde_tp="03") # 시장가 매수
                                    if order_result and order_result.get('return_code') == 0:
                                        # 주문 성공 시 포지션 기록 (단순화: 현재가로 매입 기록)
                                        self.positions[ticker] = {'quantity': ORDER_QUANTITY, 'entry_price': current_price}
                                        self.logger.info(f"({ticker}) 신규 포지션 진입. 수량: {ORDER_QUANTITY}, 진입가(추정): {current_price:.0f}")
                                    else:
                                        self.logger.error(f"({ticker}) 매수 신호 주문 실패 또는 오류: {order_result}")
                                else:
                                    # 매수 조건 미충족 시 대기
                                    self.logger.info(f"--- 매수 대기 ({ticker}) --- 예측값: {predicted_norm_value:.4f}")

                        except Exception as e: self.logger.error(f"({ticker}) 예측/주문 처리 오류: {e}"); traceback.print_exc()
                    else: self.logger.error(f"({ticker}) 모델 입력 형태 불일치: {input_seq.shape} vs (1, {TIME_STEPS}, {NUM_FEATURES})")
                else: self.logger.warning(f"({ticker}) 모델 미로드. 예측 불가.")
        except Exception as e: self.logger.error(f"process_new_bar 오류 ({ticker}): {e}"); traceback.print_exc()

    async def _aggregate_bar(self, code, trade_time_str, trade_price_str, trade_volume_str):
        # ... (이전 코드와 동일, 1분봉 집계 로직) ...
        try:
            trade_time_str = str(trade_time_str).strip(); price_str_cleaned = str(trade_price_str).strip().replace('+','').replace('-',''); volume_str_cleaned = str(trade_volume_str).strip().replace('+','').replace('-','')
            if not all([trade_time_str, price_str_cleaned, volume_str_cleaned]): self.logger.debug(f"({code}) 집계 데이터 누락 T:'{trade_time_str}', P:'{trade_price_str}', V:'{volume_str_cleaned}'"); return
            if len(trade_time_str) != 6: self.logger.warning(f"({code}) 잘못된 체결시간 형식: {trade_time_str}"); return # HHMMSS

            now_dt = datetime.now(); current_dt = now_dt.replace(hour=int(trade_time_str[:2]), minute=int(trade_time_str[2:4]), second=int(trade_time_str[4:6]), microsecond=0)
            price = float(price_str_cleaned); volume = abs(int(volume_str_cleaned)) # 거래량은 항상 양수
            if price <= 0: self.logger.warning(f"({code}) 유효하지 않은 가격: {price}"); return

            current_minute_start = current_dt.replace(second=0, microsecond=0); bar_start_minute = (current_minute_start.minute // self.bar_interval_minutes) * self.bar_interval_minutes; current_interval_start = current_minute_start.replace(minute=bar_start_minute)

            if code not in self.current_bars or self.current_bars[code]['time'] < current_interval_start:
                if code in self.current_bars and self.current_bars[code].get('open') is not None:
                    # ★★★ 이전 봉 완성 처리 ★★★
                    await self.process_new_bar(code, self.current_bars[code].copy())
                self.current_bars[code] = {'time': current_interval_start, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume, 'amount': price * volume}
                self.logger.debug(f"({code}) 새 {self.bar_interval_minutes}분봉 시작: {self.current_bars[code]}")
            else:
                bar = self.current_bars[code]; bar['high'] = max(bar.get('high', price), price); bar['low'] = min(bar.get('low', price), price); bar['close'] = price; bar['volume'] += volume; bar['amount'] += price * volume
        except ValueError as ve: self.logger.error(f"({code}) 집계 값 변환 오류: {ve}. T='{trade_time_str}', P='{trade_price_str}', V='{volume_str_cleaned}'")
        except Exception as e: self.logger.error(f"({code}) {self.bar_interval_minutes}분봉 집계 오류: {e}"); traceback.print_exc()


    async def receive_messages(self):
        """ 서버로부터 메시지 비동기 수신 및 처리 """
        # ... (이전 코드와 거의 동일, PING 반송 / REAL 파싱 수정된 버전) ...
        while self.keep_running:
            if not self.connected:
                await asyncio.sleep(0.1)
                continue
            try:
                response_str = await self.websocket.recv()
                self.logger.debug(f"Kiwoom RAW: {response_str[:300]}")
                response = json.loads(response_str)
                trnm = response.get('trnm')

                if trnm == 'LOGIN':
                    if response.get('return_code') == 0:
                        self.logger.info('*** WebSocket 로그인 성공 ***')
                        self.login_success = True
                        await self.register_realtime()
                    else:
                        self.logger.error(f'!!! WebSocket 로그인 실패: {response.get("return_msg")} !!!')
                        await self.disconnect(reconnect=False)
                        break
                elif trnm == 'PING':
                    # 키움증권 문서 예제 방식: 수신한 PING 메시지를 그대로 다시 전송하여 Heartbeat 역할 수행
                    self.logger.info(f"Kiwoom PING 수신. Heartbeat 응답으로 수신한 PING 메시지 반송: {response}")
                    await self.send_message(response) # 수정된 부분: 수신한 PING 객체를 그대로 전송
                elif trnm == 'REG':
                    return_code = response.get('return_code')
                    if return_code != 0:
                        error_mapping = {
                            105111: "잘못된 FID 형식 또는 권한 부족 (REG)",
                            105100: "토큰 만료 (REG)",
                        }
                        error_msg = error_mapping.get(return_code, "알 수 없는 REG 오류")
                        self.logger.error(f"실시간 등록 실패 응답 ({return_code}): {error_msg} - {response.get('return_msg')}")
                    else:
                        self.logger.info(f"실시간 등록 성공 응답: {response}")
                elif trnm == 'REAL':
                    data_list = response.get('data')
                    if isinstance(data_list, list):
                        for real_data in data_list: # real_data는 dict. 예: {"item": "001450", "type": "0B", "values": {"10": "12345", ...}}
                            code = real_data.get('item')       # 종목코드
                            value_map = real_data.get('values') # 'values'는 이미 FID:값 형태의 딕셔너리임

                            if not isinstance(value_map, dict):
                                self.logger.warning(f"REAL 데이터의 'values'가 딕셔너리 타입이 아님: {real_data}")
                                continue
                            if not code:
                                self.logger.warning(f"REAL 데이터에 'item'(종목코드) 누락: {real_data}")
                                continue

                            required_fids = ['20', '10', '15'] # 체결시간, 현재가, 거래량
                            missing_fids = [fid for fid in required_fids if fid not in value_map]
                            if missing_fids:
                                self.logger.warning(f"({code}) REAL 데이터 필수 FID({','.join(required_fids)}) 중 일부 누락: {missing_fids}. 수신된 FIDs: {list(value_map.keys())}. RAW Values: {value_map}")
                                continue
                            
                            # ★★★ 봉 집계 함수 호출 ★★★
                            # 이 함수 내부에서 봉 완성 시 process_new_bar가 호출됨
                            await self._aggregate_bar(
                                code,
                                value_map.get('20'),
                                value_map.get('10'),
                                value_map.get('15')
                            )
                    else:
                        self.logger.warning(f"Kiwoom REAL 'data' 형식 오류 (리스트 아님): {response}")
                elif trnm == 'SYSTEM': # SYSTEM 메시지 세분화
                    sys_code = str(response.get('code'))
                    if sys_code == 'R10004': # 토큰 관련 시스템 오류
                        self.logger.error(f"Kiwoom 시스템 오류 (R10004 - 토큰 관련): {response.get('message')}. 연결 종료.")
                        await self.disconnect(reconnect=False); break
                    elif sys_code == 'R10002': # Heartbeat 오류
                        self.logger.warning(f"Kiwoom 시스템 Heartbeat 오류 수신 (R10002): {response.get('message')}. PING 응답으로 해결 기대 중.")
                    else:
                        self.logger.info(f'Kiwoom 시스템 응답 (trnm=SYSTEM, code={sys_code}): {response}')
                else:
                    if response.get('trnm') == 'PONG' and response.get('return_code') == 105108:
                         self.logger.warning(f"서버로부터 PONG TRNM 처리 불가 응답 수신 (무시 가능): {response}")
                    else:
                        self.logger.info(f'Kiwoom 기타 응답 수신 (trnm={trnm}): {response}')

            except websockets.ConnectionClosedOK: self.logger.info('WebSocket 연결 정상 종료됨.'); self.connected = False; self.login_success = False; break
            except websockets.ConnectionClosedError as e: self.logger.error(f'WebSocket 연결 비정상 종료됨: {e}'); self.connected = False; self.login_success = False; break
            except json.JSONDecodeError: self.logger.error(f'수신 데이터 JSON 파싱 오류: {response_str}')
            except Exception as e: self.logger.error(f'메시지 처리 중 예외 발생: {e}'); traceback.print_exc(); await asyncio.sleep(1)


    async def register_realtime(self):
        # ... (이전 코드와 동일, 수정된 REG 요청 생성 로직) ...
        if not self.login_success:
            self.logger.warning("로그인 상태 아님. 실시간 등록 불가.")
            return
        self.logger.info("실시간 데이터 등록 요청 (Kiwoom)...")

        kiwoom_data_list_for_request = []
        for reg_config in REGISTER_LIST:
            payload_object = {
                "item": reg_config.get("item", []),
                "type": reg_config.get("type", [])
            }
            if not payload_object["item"] or not payload_object["type"]:
                self.logger.warning(f"잘못된 등록 설정 건너뜀: {reg_config}")
                continue
            kiwoom_data_list_for_request.append(payload_object)

        if kiwoom_data_list_for_request:
            grp_no = "1"
            refresh_opt = "1"

            kiwoom_reg_param = {
                "trnm": "REG",
                "grp_no": grp_no,
                "refresh": refresh_opt,
                "data": kiwoom_data_list_for_request
            }
            await self.send_message(kiwoom_reg_param)
            self.logger.debug(f"Kiwoom REG 상세 (수정됨): {json.dumps(kiwoom_reg_param, indent=2)}")
            await asyncio.sleep(0.3)
        self.logger.info("Kiwoom 실시간 데이터 등록 요청 완료.")

    def _send_order_sync_kiwoom(self, tr_id, headers, body):
        # ... (이전 코드와 동일, 동기 주문 전송 로직) ...
        url = KIWOOM_API_URL_ORDER
        self.logger.info(f"Kiwoom 주문 API 요청 (가이드 기반): URL={url}, TR_ID={tr_id}")
        self.logger.debug(f"Headers: {headers}"); self.logger.debug(f"Body: {json.dumps(body)}")
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
            res.raise_for_status(); response_json = res.json(); self.logger.info(f"주문 응답: {response_json}"); return response_json
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"주문 HTTP 오류: {http_err} - {res.status_code if 'res' in locals() else 'N/A'} - {res.text if 'res' in locals() else 'N/A'}")
            try: return res.json() if res.content else {"return_code": str(res.status_code), "return_msg": "Empty response on HTTP error", "output":{}}
            except json.JSONDecodeError: return {"return_code": str(res.status_code), "return_msg": f"HTTPError: {http_err}, Non-JSON", "output":{}}
        except requests.exceptions.RequestException as e: self.logger.error(f"주문 요청 오류: {e}"); return {"return_code": "-1", "return_msg": f"RequestException: {e}", "output":{}}
        except Exception as e: self.logger.error(f"주문 처리 예외: {e}"); traceback.print_exc(); return {"return_code": "-1", "return_msg": str(e), "output":{}}

    async def send_order(self, order_type, stock_code, quantity, price=0, trde_tp="3"):
        # ... (이전 코드와 동일, 비동기 주문 래퍼) ...
        self.logger.info(f"--- Kiwoom {stock_code} {order_type.upper()} 주문 (가이드 기준, trde_tp:{trde_tp}) --- Qty:{quantity}, Price:{price}")
        order_token = self.token
        if not order_token: self.logger.error("주문 불가: 주문용 토큰 없음."); return None
        if quantity <= 0: self.logger.error(f"주문 불가: 주문 수량({quantity})은 0보다 커야 함."); return None # 주문 수량 0 체크

        tr_id = KIWOOM_ORDER_TR_ID_BUY if order_type.lower()=='buy' else KIWOOM_ORDER_TR_ID_SELL
        headers = {"Content-Type":"application/json;charset=UTF-8", "Authorization":f"Bearer {order_token}", "api-id": tr_id}
        price_str = "" if trde_tp == "3" else str(int(price)) # 시장가일 경우 가격 빈 문자열
        body = {"dmst_stex_tp": "KRX", "stk_cd": stock_code, "ord_qty": str(int(quantity)), "ord_uv": price_str, "trde_tp": trde_tp}

        loop = asyncio.get_running_loop()
        try:
            # ★★★★★ 중요 ★★★★★
            # 실제 주문 전송 전 로그 남기기 (모의투자 시뮬레이션용으로도 활용 가능)
            self.logger.info(f"[실제 주문 전송 시도] 유형:{order_type}, 종목:{stock_code}, 수량:{quantity}, 가격:{price if price else '시장가'}({trde_tp})")

            # 실제 주문 API 호출 (실전투자 시 활성화)
            # response = await loop.run_in_executor(None, functools.partial(self._send_order_sync_kiwoom, tr_id, headers, body))

            # --- 모의 투자 또는 초기 테스트용: 주문 전송 시뮬레이션 ---
            # 실제 주문 API를 호출하지 않고 성공한 것처럼 가정하여 로깅 및 포지션 업데이트
            # 이렇게 하면 실제 주문 없이 로직 검증 가능
            mock_order_no = f"mock_{int(time.time())%10000:04d}"
            response = {"return_code": 0, "return_msg": "모의 주문 성공", "ord_no": mock_order_no, "output":{}}
            self.logger.info(f"=== 모의 주문 성공 처리 ===")
            # --- 시뮬레이션 끝 ---


            if response and response.get('return_code') == 0:
                ord_no = response.get('ord_no', 'N/A');
                self.logger.info(f" ★★★ Kiwoom 주문 요청 성공! 주문번호: {ord_no} (msg: {response.get('return_msg')}) ★★★")
            else:
                self.logger.error(f" !!! Kiwoom 주문 요청 실패 !!! 응답: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Kiwoom 주문 비동기 처리 오류: {e}"); traceback.print_exc(); return {"return_code": -1, "return_msg": str(e)}


    async def run(self):
        # ... (이전 코드와 동일, 메인 실행 루프) ...
        while self.keep_running:
            if not self.connected:
                self.logger.info(f"Kiwoom 연결 재시도 ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})...")
                await self.connect()
                if not self.connected:
                    wait_time = min(30, 2 ** self.reconnect_attempts) # Exponential backoff
                    self.logger.info(f"연결 실패. {wait_time}초 후 재시도.")
                    await asyncio.sleep(wait_time); self.reconnect_attempts += 1
                    if self.reconnect_attempts >= self.max_reconnect_attempts: self.logger.error("최대 재연결 시도 실패. 프로그램 종료."); self.keep_running = False; break
                    continue
                else: # 연결 성공
                    await asyncio.sleep(1.5) # 로그인 및 초기 응답 대기

            if self.connected: # 연결된 상태에서만 메시지 수신 시도
                try:
                    await self.receive_messages() # 이 함수는 연결이 끊기면 내부적으로 종료됨
                except websockets.exceptions.ConnectionClosed: # 명시적으로 처리
                    self.logger.warning("Kiwoom WS 연결 종료됨 (run 루프). 재연결 시도."); self.connected = False; self.login_success = False
                except Exception as e:
                    self.logger.error(f"Kiwoom run 루프 중 예외: {e}"); traceback.print_exc()
                    self.connected = False; self.login_success = False
                    await asyncio.sleep(5) # 예외 발생 시 잠시 대기 후 재연결 로직으로

        self.logger.info("WebSocketClient run 루프 최종 종료.")
        await self.disconnect(reconnect=False) # run 루프 종료 시 확실히 연결 해제

    async def disconnect(self, reconnect=True):
        # ... (이전 코드와 동일, 연결 해제 로직) ...
        if not reconnect: self.keep_running = False
        if self.connected and self.websocket:
            self.logger.info("Kiwoom WebSocket 연결 종료 시도...")
            try: await self.websocket.close()
            except Exception as e: self.logger.error(f"Kiwoom WS 닫는 중 오류: {e}")
            finally: self.websocket = None; self.connected = False; self.login_success = False
        elif not self.connected:
             self.logger.info("Kiwoom WS 이미 연결 해제됨 (disconnect 호출 시점).")


# --- main 함수 및 실행 부분 ---
async def main():
    # ... (이전 코드와 동일) ...
    if not all([KIWOOM_ACCOUNT_NO_STR, KIWOOM_TOKEN]):
        print("!!! 설정 오류: 키움증권 필수 정보 (토큰, 계좌번호)를 확인해주세요. !!!"); return

    client = None
    try:
        client = WebSocketClient(
            socket_uri=KIWOOM_WEBSOCKET_URL,
            token=KIWOOM_TOKEN,
            account_no=KIWOOM_ACCOUNT_NO_STR,
            app_key=KIWOOM_APP_KEY,
            app_secret=KIWOOM_APP_SECRET,
            user_id=KIWOOM_USER_ID,
            is_dev_mode=IS_MOCK_TRADING,
            monitored_tickers_list=[reg["item"][0] for reg in REGISTER_LIST if reg.get("item") and reg["item"]],
            log_file_path="trading_log_kiwoom_final.log"
        )
        await client.run()
    except KeyboardInterrupt: print("\nCtrl+C 감지 (main). Kiwoom 클라이언트 종료 중...")
    except Exception as e: print(f"main 함수 실행 중 예외 발생: {e}"); traceback.print_exc()
    finally:
        if client: await client.disconnect(reconnect=False)
        print("Kiwoom 메인 프로그램 루틴 최종 종료.")

if __name__ == '__main__':
    # ... (이전 코드와 동일) ...
    log_dir_main = os.path.dirname("trading_log_kiwoom_final.log")
    if log_dir_main and not os.path.exists(log_dir_main):
        try: os.makedirs(log_dir_main)
        except Exception as e: print(f"로그 디렉터리 생성 실패: {e}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt: print("프로그램 강제 종료 (Ctrl+C in __main__)")
    except RuntimeError as e:
        if "Event loop is closed" in str(e): print("오류: 이벤트 루프가 이미 닫혔습니다. (__main__)")
        elif "Cannot run the event loop while another loop is running" in str(e): print("오류: 다른 이벤트 루프가 이미 실행 중입니다. (__main__)")
        else: print(f"런타임 오류 발생 (__main__): {e}"); traceback.print_exc()
    except Exception as e: print(f"최상위 예외 발생 (__main__): {e}"); traceback.print_exc()
    finally:
        print("Kiwoom WebSocket 클라이언트 프로그램 최종적으로 모두 종료됨.")