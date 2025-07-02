# -*- coding: utf-8 -*-
# 파일명: websocket_client_kiwoom_final.py (키움증권 API 기준)
# 실행 환경: 64비트 Python (myenv)
# 필요 라이브러리: pip install websockets tensorflow pandas numpy requests pandas_ta joblib httpx

import logging
import logging.handlers
import asyncio
import websockets
import json
import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque
import requests # 동기 REST API 호출용
import httpx    # 비동기 REST API 호출용 (토큰 발급 등)
import traceback
import pprint
import functools
from datetime import datetime, timedelta
import joblib
import pandas_ta as ta

# --- 설정 (키움증권 API 기준) ---
KIWOOM_WEBSOCKET_LOGIN_TOKEN = "5G59I5s9zCgsRjI5MtnPyG-qOqlYrBwH-xkRy-1nnjKfYyNEmURVVpucxB5QlZowxIl1kG307O1y5RhvvM9CRw" # 웹소켓 LOGIN TRNM용 토큰
KIWOOM_ACCOUNT_NO_STR = "8102624701" # 계좌번호 전체 (예: "8012345601" - 앞8자리+뒤2자리)
KIWOOM_APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss" # REST API용 App Key
KIWOOM_APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw" # REST API용 App Secret
KIWOOM_USER_ID_FOR_LOGIN = "" # 웹소켓 LOGIN TRNM에 사용자 ID가 필요하다면 여기에 설정 (보통 토큰에 포함되거나 불필요)

IS_MOCK_TRADING = True # 모의투자 사용 여부

# --- 경로 설정 ---
MODEL_SAVE_PATH = r'G:\내 드라이브\lstm_models_per_stock_v2'
SCALER_SAVE_PATH = r'G:\내 드라이브\processed_stock_data_full_v2\scalers'
KIWOOM_REST_TOKEN_FILE = "KIWOOM_REST_API_TOKEN.json"

# --- 모델 및 피처 설정 ---
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

# --- API URL 설정 (키움증권 기준) ---
if IS_MOCK_TRADING:
    KIWOOM_WEBSOCKET_URL = "wss://mockapi.kiwoom.com:10000/api/dostk/websocket"
    KIWOOM_REST_BASE_URL = "https://mockapi.kiwoom.com"
    print(">> 키움 모의투자 환경으로 설정되었습니다.")
else:
    KIWOOM_WEBSOCKET_URL = "wss://openapi.kiwoom.com:10000/api/dostk/websocket" # [실제 키움 API 문서 확인 필요]
    KIWOOM_REST_BASE_URL = "https://openapi.kiwoom.com" # [실제 키움 API 문서 확인 필요]
    print(">> 키움 실전투자 환경으로 설정되었습니다.")

KIWOOM_API_URL_OAUTH_TOKEN = f"{KIWOOM_REST_BASE_URL}/oauth2/tokenP" # 키움 OAuth2 토큰 발급 (일반적으로 'P' 접미사)
KIWOOM_API_URL_ORDER = f"{KIWOOM_REST_BASE_URL}/uapi/domestic-stock/v1/trading/order-cash" # 키움 국내주식 현금주문 TR

REGISTER_LIST = [ {"item": ["005930"], "type": ["0B"]} ] # "0B": 주식체결

class KiwoomWsRestApiClient:
    def __init__(self, socket_uri, ws_login_token,
                 rest_app_key, rest_app_secret, account_no_str,
                 kiwoom_user_id, # ★★★ 수정된 부분: 인자 이름 일치 ★★★
                 is_dev_mode=True, monitored_tickers_list=None,
                 custom_ta_params=None, model_dir=MODEL_SAVE_PATH, scaler_dir=SCALER_SAVE_PATH,
                 log_file_path="trading_log_kiwoom_final.log"):

        self.logger = logging.getLogger("KiwoomWsRestApiClient")
        self.logger.setLevel(logging.DEBUG if is_dev_mode else logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(ch_formatter)
            self.logger.addHandler(ch)
            try:
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
                fh = logging.handlers.TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=30, encoding='utf-8')
                fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                fh.setFormatter(fh_formatter)
                self.logger.addHandler(fh)
            except Exception as e:
                print(f"파일 로거 설정 실패: {e}")

        self.logger.info(f"KiwoomWsRestApiClient 초기화 시작. 모의투자: {IS_MOCK_TRADING}")
        self.logger.info(f"접속 URI: {socket_uri}")

        self.uri = socket_uri
        self.ws_login_token = ws_login_token

        self.rest_app_key = rest_app_key
        self.rest_app_secret = rest_app_secret
        self.rest_access_token = None
        self._load_rest_token_from_file()

        self.account_no_str = account_no_str
        self.kiwoom_user_id = kiwoom_user_id # ★★★ 전달받은 인자 사용 ★★★

        self.websocket = None
        self.connected = False
        self.login_success_ws = False
        self.keep_running = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        self.bar_interval_minutes = 1
        self.current_bars = {}
        self.completed_bars = {}
        self.feature_sequences = {}
        self.scalers = {}
        self.models = {}
        self.trade_triggered = {}
        self.feature_order = FEATURE_COLUMNS_NORM

        self.model_path_root = model_dir
        self.scaler_path_root = scaler_dir

        default_ta_params = {
            'sma_lengths': [5, 20, 60, 120], 'rsi_length': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bbands_length': 20, 'bbands_std': 2.0, 'atr_length': 14,
            'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth_k': 3
        }
        self.ta_params = custom_ta_params if custom_ta_params else default_ta_params

        self.tickers_to_monitor = monitored_tickers_list if monitored_tickers_list else [item["item"][0] for item in REGISTER_LIST if item.get("item")]
        for ticker in self.tickers_to_monitor:
            all_lengths = self.ta_params.get('sma_lengths', []) + \
                          [self.ta_params.get('rsi_length', 0) + 1,
                           self.ta_params.get('bbands_length', 0),
                           self.ta_params.get('macd_slow', 0) + self.ta_params.get('macd_signal', 0) + 1,
                           self.ta_params.get('atr_length',0) + 1,
                           self.ta_params.get('stoch_k',0) + self.ta_params.get('stoch_d',0) + self.ta_params.get('stoch_smooth_k',0) +1 ] # smooth_k 포함
            max_lookback = max(all_lengths) if all_lengths else 0
            self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + max_lookback + 150)
            self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS)
            self.trade_triggered[ticker] = False
            if self.model_path_root: self.load_model(ticker)
            if self.scaler_path_root: self.load_scalers(ticker)
        self.logger.info(f"KiwoomWsRestApiClient 초기화 완료. 모니터링 종목: {self.tickers_to_monitor}")

    def _load_rest_token_from_file(self):
        if os.path.exists(KIWOOM_REST_TOKEN_FILE):
            try:
                with open(KIWOOM_REST_TOKEN_FILE, 'r') as f: token_data = json.load(f)
                issued_at_ts = token_data.get("issued_at_ts")
                expires_in_seconds = token_data.get("expires_in")
                if issued_at_ts and expires_in_seconds:
                    expire_time_ts = issued_at_ts + expires_in_seconds
                    if time.time() < expire_time_ts - 600: # 만료 10분 전
                        self.rest_access_token = token_data.get("access_token")
                        if self.rest_access_token: self.logger.info("저장된 Kiwoom REST API 토큰 로드 및 유효성 확인 성공.")
                        else: self.logger.warning("토큰 파일에 'access_token' 없음.")
                    else: self.logger.info("Kiwoom REST API 토큰 만료 또는 임박.")
                else: self.logger.warning("토큰 파일에 만료 정보 부족.")
            except Exception as e: self.logger.error(f"Kiwoom REST API 토큰 파일 로드 오류: {e}")
        else: self.logger.info("저장된 Kiwoom REST API 토큰 파일 없음.")
        if not self.rest_access_token : self.logger.info("유효한 저장된 REST 토큰 없음.")


    async def _issue_rest_access_token(self):
        headers = {"content-type": "application/json; charset=utf-8"}
        body = {"grant_type": "client_credentials", "appkey": self.rest_app_key, "appsecret": self.rest_app_secret}
        url = KIWOOM_API_URL_OAUTH_TOKEN
        self.logger.info(f"Kiwoom REST API 토큰 발급 요청: URL={url}")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                 res = await client.post(url, headers=headers, json=body)
            res.raise_for_status()
            token_data = res.json()
            self.rest_access_token = token_data.get("access_token")
            if self.rest_access_token:
                expires_in = int(token_data.get("expires_in", 21600))
                token_data["issued_at_ts"] = time.time()
                token_data["expires_in"] = expires_in
                token_data["expire_at_readable"] = (datetime.now() + timedelta(seconds=expires_in)).strftime("%Y-%m-%d %H:%M:%S")
                with open(KIWOOM_REST_TOKEN_FILE, 'w') as f: json.dump(token_data, f)
                self.logger.info(f"Kiwoom REST API 토큰 신규 발급 성공. 만료 예정: {token_data['expire_at_readable']}")
                return self.rest_access_token
            else: self.logger.error(f"토큰 발급 응답에 'access_token' 없음: {token_data}"); return None
        except httpx.HTTPStatusError as e: self.logger.error(f"토큰 발급 HTTP 오류: {e.response.status_code} - {e.response.text}")
        except Exception as e: self.logger.error(f"토큰 발급 중 예외: {e}"); traceback.print_exc()
        return None

    async def get_valid_rest_token(self):
        self._load_rest_token_from_file()
        if not self.rest_access_token:
            self.logger.info("REST API 토큰 없음/만료. 신규 발급 시도.")
            await self._issue_rest_access_token()
        return self.rest_access_token

    async def connect(self):
        self.logger.info(f"Kiwoom WebSocket 서버 연결 시도: {self.uri}")
        try:
            self.websocket = await websockets.connect(self.uri, ping_interval=30, ping_timeout=10, close_timeout=10)
            self.connected = True; self.logger.info("Kiwoom WebSocket 서버 기본 연결 성공."); self.reconnect_attempts = 0
            if self.ws_login_token:
                login_packet = {"trnm": "LOGIN", "token": self.ws_login_token}
                if self.kiwoom_user_id: login_packet["user_id"] = self.kiwoom_user_id
                self.logger.info(f"Kiwoom LOGIN 패킷 전송 시도: {login_packet}")
                await self.send_message(login_packet)
            else: self.logger.error("LOGIN 불가: 웹소켓 접속용 토큰(ws_login_token) 없음."); self.connected = False; self.login_success_ws = False
        except Exception as e: self.logger.error(f'Kiwoom WebSocket 연결/LOGIN 오류: {e}'); self.connected = False; self.login_success_ws = False

    async def send_message(self, message_dict):
        if self.connected and self.websocket:
            try:
                await self.websocket.send(json.dumps(message_dict))
            except websockets.exceptions.ConnectionClosed as e: self.logger.error(f"Kiwoom 메시지 전송 중 연결 끊김: {e}"); self.connected = False; self.login_success_ws = False
            except Exception as e: self.logger.error(f"Kiwoom 메시지 전송 오류: {e}")
        else: self.logger.warning("Kiwoom WebSocket 미연결. 메시지 전송 불가.")

    def load_model(self, ticker):
        if ticker in self.models and self.models[ticker] is not None: return self.models[ticker]
        model_file_path = os.path.join(self.model_path_root, f"{ticker}.keras")
        if os.path.exists(model_file_path):
            try:
                self.models[ticker] = tf.keras.models.load_model(model_file_path, compile=False)
                self.logger.info(f"모델 로딩 성공: {ticker} ({model_file_path})"); return self.models[ticker]
            except Exception as e: self.logger.error(f"모델 로딩 실패 ({ticker}): {e}"); self.models[ticker] = None; return None
        else: self.logger.warning(f"모델 파일 없음: {model_file_path}"); self.models[ticker] = None; return None

    def load_scalers(self, ticker):
        if ticker in self.scalers and self.scalers[ticker]: return self.scalers[ticker]
        scaler_file_path = os.path.join(self.scaler_path_root, f'{ticker}_scalers.joblib')
        if os.path.exists(scaler_file_path):
            try:
                self.scalers[ticker] = joblib.load(scaler_file_path)
                self.logger.info(f"스케일러 로딩 성공 ({ticker}): {len(self.scalers[ticker])}개 피처"); return self.scalers[ticker]
            except Exception as e: self.logger.error(f"스케일러 로딩 실패 ({ticker}): {e}"); self.scalers[ticker] = {}; return {}
        else: self.logger.warning(f"스케일러 파일 없음 ({ticker}): {scaler_file_path}"); self.scalers[ticker] = {}; return {}

    def calculate_features(self, ticker):
        all_lengths = self.ta_params.get('sma_lengths', []) + \
                      [self.ta_params.get('rsi_length', 0) + 1,
                       self.ta_params.get('bbands_length', 0),
                       self.ta_params.get('macd_slow', 0) + self.ta_params.get('macd_signal', 0) + 1,
                       self.ta_params.get('atr_length',0) + 1,
                       self.ta_params.get('stoch_k',0) + self.ta_params.get('stoch_d',0) + self.ta_params.get('stoch_smooth_k', 3) +1 ]
        min_data_length = (max(all_lengths) if all_lengths else 0) + 15

        if ticker not in self.completed_bars or len(self.completed_bars[ticker]) < min_data_length:
            self.logger.debug(f"({ticker}) 지표 계산 데이터 부족: {len(self.completed_bars.get(ticker, []))} / {min_data_length}"); return None
        df_bars = pd.DataFrame(list(self.completed_bars[ticker])).set_index('time')
        if not all(col in df_bars.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            self.logger.error(f"({ticker}) 봉 데이터 OHLCV 컬럼 없음."); return None
        self.logger.debug(f"({ticker}) 지표 계산 시작. 데이터: {len(df_bars)}개")
        try:
            custom_ta_list = [ {"kind": "sma", "length": l} for l in self.ta_params['sma_lengths']] + \
                             [ {"kind": "rsi", "length": self.ta_params['rsi_length']},
                               {"kind": "macd", "fast": self.ta_params['macd_fast'], "slow": self.ta_params['macd_slow'], "signal": self.ta_params['macd_signal']},
                               {"kind": "bbands", "length": self.ta_params['bbands_length'], "std": float(self.ta_params['bbands_std'])},
                               {"kind": "atr", "length": self.ta_params['atr_length']}, {"kind": "obv"},
                               {"kind": "stoch", "k": self.ta_params['stoch_k'], "d": self.ta_params['stoch_d'], "smooth_k": self.ta_params.get('stoch_smooth_k', 3)} ]
            df_bars.ta.strategy(ta.Strategy(name="TAStrategy", ta=custom_ta_list))
        except Exception as e: self.logger.error(f"({ticker}) 지표 계산 오류: {e}"); traceback.print_exc(); return None
        
        latest_features = {}; latest_bar = df_bars.iloc[-1]
        for col in ['open', 'high', 'low', 'close', 'volume']: latest_features[col] = latest_bar.get(col)
        latest_features['amount'] = latest_bar.get('amount', latest_bar['close'] * latest_bar['volume'] if pd.notna(latest_bar.get('close')) and pd.notna(latest_bar.get('volume')) else 0)
        for length in self.ta_params['sma_lengths']: latest_features[f'SMA_{length}'] = latest_bar.get(f'SMA_{length}')
        latest_features[f'RSI_{self.ta_params["rsi_length"]}'] = latest_bar.get(f'RSI_{self.ta_params["rsi_length"]}')
        macd_s = f'{self.ta_params["macd_fast"]}_{self.ta_params["macd_slow"]}_{self.ta_params["macd_signal"]}'
        latest_features[f'MACD_{macd_s}'] = latest_bar.get(f'MACD_{macd_s}'); latest_features[f'MACDs_{macd_s}'] = latest_bar.get(f'MACDS_{macd_s}'); latest_features[f'MACDh_{macd_s}'] = latest_bar.get(f'MACDH_{macd_s}')
        bb_s = f'{self.ta_params["bbands_length"]}_{float(self.ta_params["bbands_std"]):.1f}'
        for b_col in ['BBL', 'BBM', 'BBU', 'BBB', 'BBP']: latest_features[f'{b_col}_{bb_s}'] = latest_bar.get(f'{b_col}_{bb_s}')
        atr_df_col = f'ATRr_{self.ta_params["atr_length"]}'; latest_features['ATRr_14'] = latest_bar.get(atr_df_col if atr_df_col in df_bars.columns else f'ATR_{self.ta_params["atr_length"]}')
        latest_features['OBV'] = latest_bar.get('OBV')
        stoch_df_s = f'{self.ta_params["stoch_k"]}_{self.ta_params["stoch_d"]}_{self.ta_params.get("stoch_smooth_k",3)}'
        latest_features['STOCHk_14_3_3'] = latest_bar.get(f'STOCHk_{stoch_df_s}'); latest_features['STOCHd_14_3_3'] = latest_bar.get(f'STOCHd_{stoch_df_s}')
        latest_features['PBR'] = 0.0; latest_features['PER'] = 0.0; latest_features['USD_KRW'] = 0.0
        latest_features['is_month_end'] = 1 if latest_bar.name.is_month_end else 0

        feature_vector = np.full(NUM_FEATURES, np.nan)
        for i, col_norm_name in enumerate(self.feature_order):
            original_name_key = col_norm_name[:-5] if col_norm_name.endswith('_norm') else col_norm_name
            value = latest_features.get(original_name_key)
            if value is not None: feature_vector[i] = value
            elif col_norm_name == 'is_month_end': feature_vector[i] = 0
        if np.isnan(feature_vector).any(): self.logger.warning(f"({ticker}) 최종 피처 벡터 NaN 포함. 예측 건너뜀."); return None
        self.logger.debug(f"({ticker}) 계산된 원본 피처 (첫 5개): {np.nan_to_num(feature_vector[:5], nan=-999.0)}")
        return feature_vector

    def normalize_features(self, ticker, feature_vector):
        scalers_for_ticker = self.load_scalers(ticker)
        if not scalers_for_ticker: self.logger.warning(f"({ticker}) 스케일러 없음. 정규화 불가."); return None
        normalized_vector = np.zeros_like(feature_vector, dtype=float)
        for i, col_norm_name in enumerate(self.feature_order):
            raw_value = feature_vector[i]
            if np.isnan(raw_value): normalized_vector[i] = 0.0; continue
            if col_norm_name.endswith('_norm'):
                original_name = col_norm_name[:-5]; scaler = scalers_for_ticker.get(original_name)
                if scaler:
                    try: normalized_vector[i] = scaler.transform([[raw_value]])[0][0]
                    except Exception as e: self.logger.error(f"({ticker}) '{original_name}' 정규화 오류: {e}. 0.0 처리."); normalized_vector[i] = 0.0
                else: self.logger.warning(f"({ticker}) '{original_name}' 스케일러 없음. 원본값 사용."); normalized_vector[i] = raw_value
            else: normalized_vector[i] = raw_value
        if np.isnan(normalized_vector).any(): self.logger.error(f"({ticker}) 정규화 후 NaN 발생. 예측 건너뜀."); return None
        self.logger.debug(f"({ticker}) 정규화된 피처 (첫 5개): {normalized_vector[:5]}")
        return normalized_vector

    def process_new_bar(self, ticker, bar_data):
        if not hasattr(self, 'logger'): print(f"CRITICAL: Logger not in process_new_bar for {ticker}"); return
        self.logger.info(f"1분봉 완성 ({ticker}): T={bar_data['time']}, O={bar_data['open']:.0f}, H={bar_data['high']:.0f}, L={bar_data['low']:.0f}, C={bar_data['close']:.0f}, V={bar_data['volume']:.0f}")
        if ticker not in self.completed_bars:
            max_len_ta = max(self.ta_params.get('sma_lengths', [0]) + [self.ta_params.get('rsi_length',0), self.ta_params.get('bbands_length',0), self.ta_params.get('macd_slow',0) + self.ta_params.get('macd_signal',0)])
            self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + max_len_ta + 150)
        self.completed_bars[ticker].append(bar_data)

        raw_feature_vector = self.calculate_features(ticker)
        if raw_feature_vector is None: self.logger.debug(f"({ticker}) 원본 피처 계산 실패 (process_new_bar)"); return
        normalized_features = self.normalize_features(ticker, raw_feature_vector)
        if normalized_features is None: self.logger.debug(f"({ticker}) 피처 정규화 실패 (process_new_bar)"); return

        if ticker not in self.feature_sequences: self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS)
        self.feature_sequences[ticker].append(normalized_features)
        self.logger.debug(f"({ticker}) 피처 시퀀스 업데이트. 크기: {len(self.feature_sequences[ticker])}/{TIME_STEPS}")

        if len(self.feature_sequences[ticker]) == TIME_STEPS:
            model = self.load_model(ticker)
            if model:
                input_seq = np.expand_dims(np.array(list(self.feature_sequences[ticker])).astype(np.float32), axis=0)
                if input_seq.shape == (1, TIME_STEPS, NUM_FEATURES):
                    try:
                        prediction = model.predict(input_seq, verbose=0); predicted_norm_value = prediction[0][0]
                        self.logger.info(f" ★★★ 예측 ({ticker}): {predicted_norm_value:.6f} ★★★")
                        # 주문 로직 예시
                        # if ticker == '005930' and not self.trade_triggered.get(ticker, False):
                        #     if predicted_norm_value > 0.65:
                        #         self.logger.info(f"★★★ 매수 조건 ({ticker}): 예측={predicted_norm_value:.4f} > 0.65 ★★★")
                        #         # asyncio.create_task(self.send_order("buy", ticker, 1, price=0, kiwoom_order_cond="03"))
                        #         # self.trade_triggered[ticker] = True
                    except Exception as e: self.logger.error(f"({ticker}) 예측/주문 오류: {e}"); traceback.print_exc()
                else: self.logger.error(f"({ticker}) 모델 입력 형태 불일치: {input_seq.shape}")
            else: self.logger.warning(f"({ticker}) 모델 미로드. 예측 불가.")

    async def _aggregate_bar(self, code, trade_time_str, trade_price_str, trade_volume_str):
        try:
            trade_time_str = str(trade_time_str).strip()
            price_str_cleaned = str(trade_price_str).strip().replace('+', '').replace('-', '')
            volume_str_cleaned = str(trade_volume_str).strip().replace('+', '').replace('-', '')

            if not all([trade_time_str, price_str_cleaned, volume_str_cleaned]): return
            if len(trade_time_str) != 6: return

            now_dt = datetime.now(); current_dt = now_dt.replace(hour=int(trade_time_str[:2]), minute=int(trade_time_str[2:4]), second=int(trade_time_str[4:6]), microsecond=0)
            price = float(price_str_cleaned); volume = abs(int(volume_str_cleaned))
            if price <= 0: return

            current_minute_start = current_dt.replace(second=0, microsecond=0)
            bar_start_minute = (current_minute_start.minute // self.bar_interval_minutes) * self.bar_interval_minutes
            current_interval_start = current_minute_start.replace(minute=bar_start_minute)

            if code not in self.current_bars or self.current_bars[code]['time'] < current_interval_start:
                if code in self.current_bars and self.current_bars[code].get('open') is not None:
                    self.process_new_bar(code, self.current_bars[code].copy())
                self.current_bars[code] = {'time': current_interval_start, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume, 'amount': price * volume}
                # self.logger.debug(f"({code}) 새 분봉 시작(집계): {self.current_bars[code]['time']}") # 로그 너무 많음
            else:
                bar = self.current_bars[code]
                bar['high'] = max(bar['high'], price); bar['low'] = min(bar['low'], price)
                bar['close'] = price; bar['volume'] += volume; bar['amount'] += price * volume
        except ValueError as ve: self.logger.error(f"({code}) 값 변환 오류(집계): {ve}. T='{trade_time_str}', P='{trade_price_str}', V='{trade_volume_str}'")
        except Exception as e: self.logger.error(f"({code}) 1분봉 집계 중 오류: {e}"); traceback.print_exc()

    async def receive_messages(self):
        while self.keep_running:
            if not self.connected:
                self.logger.info(f"Kiwoom 연결 끊김. {self.reconnect_attempts + 1}번째 재연결 시도...")
                await asyncio.sleep(min(30, 2 ** self.reconnect_attempts)); self.reconnect_attempts += 1
                if self.reconnect_attempts > self.max_reconnect_attempts: self.logger.error("Kiwoom 최대 재연결 실패."); self.keep_running = False; break
                await self.connect(); continue
            try:
                response_str = await asyncio.wait_for(self.websocket.recv(), timeout=70.0)
                self.logger.debug(f"Kiwoom RAW 수신: {response_str[:200]}")
                if not response_str.strip(): continue
                response = json.loads(response_str); trnm = response.get('trnm')
                return_code = response.get('return_code'); msg = response.get('return_msg', response.get('msg', ''))

                if trnm == 'LOGIN':
                    if str(return_code) == '0': self.logger.info(f"*** Kiwoom WS LOGIN 성공: {msg} ***"); self.login_success_ws = True; self.reconnect_attempts = 0; await self.register_realtime()
                    else: self.logger.error(f"!!! Kiwoom WS LOGIN 실패: Code={return_code}, Msg={msg} !!!"); await self.disconnect(reconnect=False); break
                elif trnm == 'PING': self.logger.info(f"Kiwoom PING: {response}. PONG 전송."); await self.send_message({"trnm": "PONG"})
                elif trnm == 'REG': self.logger.info(f"Kiwoom REG 응답: Code={return_code}, Msg={msg}, Data={response.get('data')}")
                elif trnm == 'REAL':
                    data_list = response.get('data')
                    if isinstance(data_list, list):
                        for real_data in data_list:
                            code = real_data.get('item'); data_type = real_data.get('type'); values = real_data.get('values')
                            if not all([code, data_type, values]): self.logger.warning(f"Kiwoom REAL 필드 누락: {real_data}"); continue
                            if data_type == '0B': await self._aggregate_bar(code, values.get('20'), values.get('10'), values.get('15'))
                    else: self.logger.warning(f"Kiwoom REAL 'data' 형식 오류: {response}")
                else: self.logger.info(f"Kiwoom 기타 응답 (trnm={trnm}): {response}")
            except asyncio.TimeoutError: self.logger.warning("Kiwoom 메시지 수신 타임아웃."); await self.send_message({"trnm":"PING"})
            except websockets.exceptions.ConnectionClosed: self.logger.warning("Kiwoom WS 연결 종료됨 (recv 중)."); self.connected = False; self.login_success_ws = False
            except json.JSONDecodeError: self.logger.error(f"Kiwoom JSON 파싱 오류: {response_str[:200]}")
            except Exception as e: self.logger.error(f"Kiwoom 메시지 처리 중 예외: {e}"); traceback.print_exc(); await asyncio.sleep(1)

    async def register_realtime(self):
        if not self.login_success_ws: self.logger.warning("Kiwoom 로그인 실패. 실시간 등록 불가."); return
        self.logger.info("실시간 데이터 등록 요청 (Kiwoom)...")
        for reg_config in REGISTER_LIST:
            data_payload = [{"item": item_code, "type": type_code} for item_code in reg_config["item"] for type_code in reg_config["type"]]
            if data_payload:
                kiwoom_reg_param = {"trnm": "REG", "grp_no": "1", "refresh": "1", "data": data_payload}
                await self.send_message(kiwoom_reg_param)
                self.logger.info(f"Kiwoom 실시간 등록 요청: {kiwoom_reg_param}")
                await asyncio.sleep(0.3)
        self.logger.info("Kiwoom 실시간 데이터 등록 요청 완료.")

    def _send_order_sync_kiwoom(self, tr_id, order_headers, order_body):
        self.logger.info(f"Kiwoom 주문 요청: URL={KIWOOM_API_URL_ORDER}, TR_ID={tr_id}")
        self.logger.debug(f"Headers: {order_headers}"); self.logger.debug(f"Body: {json.dumps(order_body)}")
        try:
            res = requests.post(KIWOOM_API_URL_ORDER, headers=order_headers, data=json.dumps(order_body), timeout=10)
            res.raise_for_status(); return res.json()
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"주문 HTTP 오류: {http_err} - {res.text if 'res' in locals() else ''}")
            try: return res.json()
            except: return {"rt_cd": str(res.status_code), "msg1": res.text}
        except Exception as e: self.logger.error(f"주문 처리 예외: {e}"); traceback.print_exc(); return {"rt_cd": "-1", "msg1": str(e)}

    async def send_order(self, order_type, stock_code, quantity, price=0, kiwoom_order_cond="00"):
        self.logger.info(f"--- Kiwoom {stock_code} {order_type.upper()} 주문 (조건:{kiwoom_order_cond}) --- Qty:{quantity}, Price:{price}")
        valid_rest_token = await self.get_valid_rest_token()
        if not valid_rest_token: self.logger.error("주문 불가: Kiwoom REST API 토큰 없음."); return None

        tr_id = "VTTC0802U" if order_type.lower() == 'buy' and IS_MOCK_TRADING else \
                "VTTC0801U" if order_type.lower() == 'sell' and IS_MOCK_TRADING else \
                "TTTC0802U" if order_type.lower() == 'buy' and not IS_MOCK_TRADING else \
                "TTTC0801U" if order_type.lower() == 'sell' and not IS_MOCK_TRADING else None
        if not tr_id: self.logger.error(f"주문 TRID 설정 오류: {order_type}"); return None

        headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {valid_rest_token}",
                   "appkey": self.rest_app_key, "appsecret": self.rest_app_secret, "tr_id": tr_id, "custtype": "P"}
        ord_dvsn = "01" if kiwoom_order_cond == "03" else "00"; price_str = "0" if ord_dvsn == "01" else str(int(price))
        
        cano_prefix = self.account_no_str[:8]
        acnt_prdt_cd = self.account_no_str[8:] if len(self.account_no_str) > 8 else "01" # 기본 상품코드
        if len(self.account_no_str) < 9: self.logger.warning(f"계좌번호 형식이 짧습니다: {self.account_no_str}. 접미사를 '01'로 가정합니다.")


        body = {"CANO": cano_prefix, "ACNT_PRDT_CD": acnt_prdt_cd, "PDNO": stock_code,
                "ORD_DVSN": ord_dvsn, "ORD_QTY": str(quantity), "ORD_UNPR": price_str}
        
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, functools.partial(self._send_order_sync_kiwoom, tr_id, headers, body))
            if response and response.get('rt_cd') == '0':
                ord_no = response.get('output', {}).get('ORD_NO', 'N/A')
                self.logger.info(f" ★★★ Kiwoom 주문 성공! 주문번호: {ord_no} (msg1: {response.get('msg1')}) ★★★")
            else: self.logger.error(f" !!! Kiwoom 주문 실패 !!! 응답: {response}")
            return response
        except Exception as e: self.logger.error(f"Kiwoom 주문 비동기 처리 오류: {e}"); traceback.print_exc(); return {"rt_cd": "E96", "msg1": str(e)}

    async def run(self):
        if not IS_MOCK_TRADING and not await self.get_valid_rest_token():
             self.logger.error("실전 투자 REST API 토큰 발급 실패. 주문 불가."); # 웹소켓은 계속 시도
        
        while self.keep_running:
            if not self.connected or not self.login_success_ws:
                self.logger.info(f"Kiwoom 연결/로그인 재시도 ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})...")
                await self.connect()
                if not self.connected or not self.login_success_ws:
                    await asyncio.sleep(min(30, 2 ** self.reconnect_attempts)); self.reconnect_attempts += 1
                    if self.reconnect_attempts >= self.max_reconnect_attempts: self.logger.error("최대 재연결 실패."); self.keep_running = False; break
                    continue
            try: await self.receive_messages()
            except websockets.exceptions.ConnectionClosed: self.logger.warning("Kiwoom recv 중 연결 끊김."); self.connected = False; self.login_success_ws = False
            except Exception as e: self.logger.error(f"Kiwoom recv 루프 중 예외: {e}"); traceback.print_exc(); self.connected = False; self.login_success_ws = False; await asyncio.sleep(5)
        self.logger.info("KiwoomWsRestApiClient run 루프 종료.")
        await self.disconnect(reconnect=False)

    async def disconnect(self, reconnect=True):
        self.logger.info(f"Kiwoom 클라이언트 disconnect 호출 (reconnect={reconnect})...")
        if not reconnect: self.keep_running = False
        if self.websocket and self.connected:
            self.logger.info("Kiwoom WebSocket 연결 종료 시도...")
            try: await self.websocket.close()
            except Exception as e: self.logger.error(f"Kiwoom WS 닫는 중 오류: {e}")
            finally: self.websocket = None; self.connected = False; self.login_success_ws = False
        else: self.logger.info("Kiwoom WS 이미 연결 해제됨.")

async def main():
    if not all([KIWOOM_WEBSOCKET_LOGIN_TOKEN, KIWOOM_ACCOUNT_NO_STR, KIWOOM_APP_KEY, KIWOOM_APP_SECRET]):
        print("!!! 설정 오류: 키움증권 필수 정보 (토큰, 계좌번호, 앱키, 앱시크릿)를 확인해주세요. !!!"); return

    client = None
    try:
        client = KiwoomWsRestApiClient(
            socket_uri=KIWOOM_WEBSOCKET_URL,
            ws_login_token=KIWOOM_WEBSOCKET_LOGIN_TOKEN,
            rest_app_key=KIWOOM_APP_KEY,
            rest_app_secret=KIWOOM_APP_SECRET,
            account_no_str=KIWOOM_ACCOUNT_NO_STR, # ★★★ 단일 계좌번호 문자열 전달 ★★★
            kiwoom_user_id=KIWOOM_USER_ID_FOR_LOGIN, # ★★★ 올바른 인자명으로 전달 ★★★
            is_dev_mode=IS_MOCK_TRADING,
            monitored_tickers_list=[reg["item"][0] for reg in REGISTER_LIST if reg.get("item")],
            log_file_path="trading_log_kiwoom_final.log"
        )
        await client.run()
    except KeyboardInterrupt: print("\nCtrl+C 감지 (main). Kiwoom 클라이언트 종료 중...")
    except Exception as e: print(f"main 함수 실행 중 예외 발생: {e}"); traceback.print_exc()
    finally:
        if client: await client.disconnect(reconnect=False)
        print("Kiwoom 메인 프로그램 루틴 최종 종료.")

if __name__ == '__main__':
    try: import httpx
    except ImportError: print("httpx 라이브러리가 필요합니다. pip install httpx"); exit()
    log_dir_main = os.path.dirname("trading_log_kiwoom_final.log")
    if log_dir_main and not os.path.exists(log_dir_main):
        try: os.makedirs(log_dir_main)
        except Exception as e: print(f"로그 디렉터리 생성 실패: {e}")
    try: asyncio.run(main())
    except KeyboardInterrupt: print("프로그램 강제 종료 (Ctrl+C in __main__)")
    except RuntimeError as e:
        if "Event loop is closed" in str(e): print("오류: 이벤트 루프가 이미 닫혔습니다. (__main__)")
        elif "Cannot run the event loop while another loop is running" in str(e): print("오류: 다른 이벤트 루프가 이미 실행 중입니다. (__main__)")
        else: print(f"런타임 오류 발생 (__main__): {e}"); traceback.print_exc()
    except Exception as e: print(f"최상위 예외 발생 (__main__): {e}"); traceback.print_exc()
    finally: print("Kiwoom WebSocket 클라이언트 프로그램 최종적으로 모두 종료됨.")