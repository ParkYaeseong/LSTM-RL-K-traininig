# -*- coding: utf-8 -*-
# 파일명: websocket_client_kiwoom_final.py (키움증권 API 기준)
# 실행 환경: 64비트 Python (myenv)
# 필요 라이브러리: pip install websockets tensorflow pandas numpy requests pandas_ta joblib

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
import requests
import traceback
import pprint
import functools
from datetime import datetime, timedelta
import joblib
import pandas_ta as ta

# --- 설정 ---
# 키움증권 REST API용 (OAuth2 토큰 발급 및 API 호출 시 사용)
ACCESS_TOKEN = "5G59I5s9zCgsRjI5MtnPyG-qOqlYrBwH-xkRy-1nnjKfYyNEmURVVpucxB5QlZowxIl1kG307O1y5RhvvM9CRw" # 실제 발급받은 토큰으로 변경
ACCOUNT_NO = 81026247 
APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss"  
APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw"

# 키움증권 계좌 정보
KIWOOM_ACCOUNT_NO_PREFIX = "81026247" # 계좌번호 앞 8자리
KIWOOM_ACCOUNT_NO_SUFFIX = "01"     # 계좌번호 뒤 2자리 (일반적으로 "01", 상품별로 다를 수 있음)

# 키움증권 웹소켓 접속 및 LOGIN TRNM용 토큰 (이 토큰의 발급 방식은 키움 API 정책 확인 필요)
# 만약 REST API용 Access Token과 동일하다면 그것을 사용. 다르다면 별도 발급/저장 로직 필요.
# 제공된 로그에서는 이 토큰을 LOGIN 시 사용.
KIWOOM_WEBSOCKET_LOGIN_TOKEN = "5G59I5s9zCgsRjI5MtnPyG-qOqlYrBwH-xkRy-1nnjKfYyNEmURVVpucxB5QlZowxIl1kG307O1y5RhvvM9CRw"


IS_MOCK_TRADING = True # 모의투자 사용 여부

# --- 경로 설정 ---
MODEL_SAVE_PATH = r'G:\내 드라이브\lstm_models_per_stock_v2'
SCALER_SAVE_PATH = r'G:\내 드라이브\processed_stock_data_full_v2\scalers'
TOKEN_FILE_PATH = "KIWOOM_REST_TOKEN.json" # REST API 용 Access Token 저장 파일

# --- 모델 및 피처 설정 (이전과 동일) ---
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
    # 웹소켓 URL (로그 기반)
    KIWOOM_SOCKET_URL = "wss://mockapi.kiwoom.com:10000/api/dostk/websocket"
    # REST API 기본 URL (로그 기반)
    KIWOOM_REST_BASE_URL = "https://mockapi.kiwoom.com"
    print(">> 모의투자 환경으로 설정되었습니다.")
else:
    # 웹소켓 URL (실전 투자 - 실제 URL 확인 필요)
    KIWOOM_SOCKET_URL = "wss://openapi.kiwoom.com:10000/api/dostk/websocket" # [실제 키움 API 문서 확인 필요]
    # REST API 기본 URL (실전 투자 - 실제 URL 확인 필요)
    KIWOOM_REST_BASE_URL = "https://openapi.kiwoom.com" # [실제 키움 API 문서 확인 필요]
    print(">> 실전투자 환경으로 설정되었습니다.")

# REST API 엔드포인트 (키움 OpenAPI+ 문서 기준)
KIWOOM_API_URL_TOKEN = f"{KIWOOM_REST_BASE_URL}/oauth2/tokenP"
KIWOOM_API_URL_ORDER = f"{KIWOOM_REST_BASE_URL}/uapi/domestic-stock/v1/trading/order-cash" # 주식 현금 주문

# --- 실시간 등록 설정 (키움증권 형식) ---
# REGISTER_LIST는 웹소켓을 통해 'REG' TRNM으로 전송됨
REGISTER_LIST = [ {"item": ["005930"], "type": ["0B"]} ] # 0B: 주식체결. 호가(0A) 등 추가 가능

class KiwoomStockWSClient:
    def __init__(self, socket_uri, ws_login_token,
                 rest_app_key, rest_app_secret, account_prefix, account_suffix,
                 is_dev_mode=True, monitored_tickers_list=None,
                 custom_ta_params=None, model_dir=MODEL_SAVE_PATH, scaler_dir=SCALER_SAVE_PATH,
                 log_file_path="trading_log_kiwoom.log"):

        self.logger = logging.getLogger("KiwoomStockWSClient")
        self.logger.setLevel(logging.DEBUG if is_dev_mode else logging.INFO)
        if not self.logger.handlers:
            # StreamHandler 설정
            ch = logging.StreamHandler()
            ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(ch_formatter)
            self.logger.addHandler(ch)
            # FileHandler 설정
            try:
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
                fh = logging.handlers.TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=30, encoding='utf-8')
                fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                fh.setFormatter(fh_formatter)
                self.logger.addHandler(fh)
            except Exception as e:
                print(f"파일 로거 설정 실패: {e}") # 콘솔에는 계속 출력

        self.logger.info(f"KiwoomStockWSClient 초기화 시작. 모의투자: {IS_MOCK_TRADING}")

        self.uri = socket_uri
        self.ws_login_token = ws_login_token # 웹소켓 LOGIN trnm용 토큰

        # REST API 호출용 인증 정보
        self.rest_app_key = rest_app_key
        self.rest_app_secret = rest_app_secret
        self.rest_access_token = None # REST API용 Access Token (필요시 발급)

        self.account_no_prefix = account_prefix
        self.account_no_suffix = account_suffix

        self.websocket = None
        self.connected = False
        self.login_success_ws = False # 웹소켓 LOGIN 성공 여부
        self.keep_running = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

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
            max_lookback = max(self.ta_params['sma_lengths'] + [self.ta_params['rsi_length'], self.ta_params['bbands_length'], self.ta_params['macd_slow'] + self.ta_params['macd_signal'], self.ta_params['atr_length'], self.ta_params['stoch_k'] + self.ta_params['stoch_d']])
            self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + max_lookback + 100) # 버퍼 증가
            self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS)
            self.trade_triggered[ticker] = False
            self.load_model(ticker)
            self.load_scalers(ticker)
        self.logger.info("KiwoomStockWSClient 초기화 완료.")

    def _get_rest_access_token(self):
        """키움 REST API용 Access Token 발급 및 관리"""
        if self.rest_access_token:
            # 토큰 유효성 검사 (만료 시간 등) 로직 추가 가능
            # 여기서는 단순화를 위해 기존 토큰이 있으면 반환
            # 실제로는 만료 시간을 체크하고 필요시 재발급해야 함
            try:
                with open(TOKEN_FILE_PATH, 'r') as f:
                    token_data = json.load(f)
                expire_time = datetime.strptime(token_data.get("expire_at", "1970-01-01T00:00:00"), "%Y-%m-%dT%H:%M:%S")
                if datetime.now() < expire_time - timedelta(minutes=10) : # 만료 10분 전 갱신
                    self.logger.info("기존 REST API 토큰 사용.")
                    self.rest_access_token = token_data["access_token"]
                    return self.rest_access_token
                else:
                    self.logger.info("REST API 토큰 만료 또는 임박. 신규 발급.")
            except FileNotFoundError:
                self.logger.info("저장된 REST API 토큰 파일 없음. 신규 발급.")
            except Exception as e:
                self.logger.error(f"저장된 REST API 토큰 로드 오류: {e}")
        
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.rest_app_key,
            "appsecret": self.rest_app_secret
        }
        self.logger.info(f"키움 REST API 토큰 발급 요청: URL={KIWOOM_API_URL_TOKEN}, Body={body}")
        try:
            res = requests.post(KIWOOM_API_URL_TOKEN, headers=headers, data=json.dumps(body))
            res.raise_for_status()
            token_data = res.json()
            self.rest_access_token = token_data.get("access_token")
            if self.rest_access_token:
                expires_in_seconds = int(token_data.get("expires_in", 43200)) # 기본 12시간
                expire_at = datetime.now() + timedelta(seconds=expires_in_seconds)
                token_data["expire_at"] = expire_at.strftime("%Y-%m-%dT%H:%M:%S")
                with open(TOKEN_FILE_PATH, 'w') as f:
                    json.dump(token_data, f)
                self.logger.info(f"키움 REST API 토큰 발급 성공. 만료 예정: {expire_at}")
                return self.rest_access_token
            else:
                self.logger.error(f"키움 REST API 토큰 발급 실패: 'access_token' 필드 없음. 응답: {token_data}")
                return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"키움 REST API 토큰 발급 요청 오류: {e}")
        except Exception as e:
            self.logger.error(f"키움 REST API 토큰 발급 중 예외 발생: {e}")
        return None


    async def connect(self):
        self.logger.info(f"Kiwoom WebSocket 서버 연결 시도: {self.uri}")
        try:
            self.websocket = await websockets.connect(self.uri, ping_interval=20, ping_timeout=10) # ping 간격 조정
            self.connected = True
            self.logger.info("Kiwoom WebSocket 서버 연결 성공.")
            self.reconnect_attempts = 0

            if self.ws_login_token:
                # 제공된 로그 형식의 LOGIN trnm 사용
                login_packet = {
                    "trnm": "LOGIN",
                    "token": self.ws_login_token
                    # 키움 API 명세에 따라 user_id 등 추가 정보 필요시 여기에 포함
                }
                self.logger.info(f"LOGIN 패킷 전송 시도: {login_packet}")
                await self.send_message(login_packet)
            else:
                self.logger.error("LOGIN 불가: 웹소켓 접속용 토큰(ws_login_token)이 없습니다.")
                self.connected = False
                self.login_success_ws = False

        except Exception as e:
            self.logger.error(f'Kiwoom WebSocket 연결 또는 LOGIN 시도 중 오류: {e}')
            self.connected = False
            self.login_success_ws = False


    async def send_message(self, message_dict):
        if self.connected and self.websocket:
            try:
                message_str = json.dumps(message_dict)
                await self.websocket.send(message_str)
                # self.logger.debug(f"Sent to Kiwoom: {message_str}") # 상세 로깅 필요시 주석 해제
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.error(f"Kiwoom 메시지 전송 중 연결 끊김: {e}")
                self.connected = False
            except Exception as e:
                self.logger.error(f"Kiwoom 메시지 전송 오류: {e}")
        else:
            self.logger.warning("Kiwoom WebSocket 미연결 상태. 메시지 전송 불가.")

    # load_model, load_scalers, calculate_features, normalize_features, process_new_bar 는 이전과 동일하게 유지
    # (내부 로깅 메시지나 미세 조정은 필요할 수 있음)
    def load_model(self, ticker):
        if ticker in self.models: return self.models[ticker]
        model_file_path = os.path.join(self.model_path_root, f"{ticker}.keras")
        if os.path.exists(model_file_path):
            try:
                self.models[ticker] = tf.keras.models.load_model(model_file_path, compile=False)
                self.logger.info(f"모델 로딩 성공: {ticker} from {model_file_path}")
                return self.models[ticker]
            except Exception as e:
                self.logger.error(f"모델 로딩 실패 ({ticker}): {e} at {model_file_path}")
                return None
        else:
            self.logger.warning(f"모델 파일 없음: {model_file_path}")
            return None

    def load_scalers(self, ticker):
        if ticker in self.scalers: return self.scalers[ticker]
        scaler_file_path = os.path.join(self.scaler_path_root, f'{ticker}_scalers.joblib')
        if os.path.exists(scaler_file_path):
            try:
                self.logger.debug(f"스케일러 로딩 시도: {scaler_file_path}")
                loaded_scaler_dict = joblib.load(scaler_file_path)
                self.scalers[ticker] = loaded_scaler_dict
                self.logger.info(f"스케일러 로딩 성공 ({ticker}): {len(loaded_scaler_dict)}개 피처 from {scaler_file_path}")
                return self.scalers[ticker]
            except Exception as e:
                self.logger.error(f"스케일러 로딩 실패 ({ticker}): {e} at {scaler_file_path}")
                self.scalers[ticker] = {}; return {}
        else:
            self.logger.warning(f"스케일러 파일 없음 ({ticker}): {scaler_file_path}")
            self.scalers[ticker] = {}; return {}

    def calculate_features(self, ticker):
        # 이 함수는 self.logger를 사용하므로, __init__에서 self.logger가 먼저 할당되어야 함
        required_lengths_for_ta = [
            max(self.ta_params['sma_lengths']), self.ta_params['rsi_length'] + 1,
            self.ta_params['bbands_length'], self.ta_params['macd_slow'] + self.ta_params['macd_signal'] + 1,
            self.ta_params['atr_length'] +1, # ATR은 보통 (length-1)개의 이전 데이터 + 현재 데이터
            self.ta_params['stoch_k'] + self.ta_params['stoch_d'] + self.ta_params.get('stoch_smooth_k', 3) -2 +1 # Stochastic은 k,d,smooth_k의 합에서 중복분을 빼고 + alpha
        ]
        min_data_length = max(required_lengths_for_ta) + 10 # 넉넉한 버퍼

        if ticker not in self.completed_bars or len(self.completed_bars[ticker]) < min_data_length:
            if hasattr(self, 'logger'): # logger가 있는지 확인 후 사용
                self.logger.debug(f"({ticker}) 기술적 지표 계산을 위한 데이터 부족: 현재 {len(self.completed_bars.get(ticker, []))}개 / 필요 {min_data_length}개")
            return None

        recent_bars_list = list(self.completed_bars[ticker])
        df_bars = pd.DataFrame(recent_bars_list).set_index('time')
        if not all(col in df_bars.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            self.logger.error(f"({ticker}) 봉 데이터에 OHLCV 컬럼이 없습니다.")
            return None

        self.logger.debug(f"  >> 기술적 지표 계산 시작 ({ticker})... 데이터 개수: {len(df_bars)}")
        try:
            custom_ta_list = [
                {"kind": "sma", "length": l} for l in self.ta_params['sma_lengths']
            ] + [
                {"kind": "rsi", "length": self.ta_params['rsi_length']},
                {"kind": "macd", "fast": self.ta_params['macd_fast'], "slow": self.ta_params['macd_slow'], "signal": self.ta_params['macd_signal']},
                {"kind": "bbands", "length": self.ta_params['bbands_length'], "std": float(self.ta_params['bbands_std'])},
                {"kind": "atr", "length": self.ta_params['atr_length']},
                {"kind": "obv"},
                {"kind": "stoch", "k": self.ta_params['stoch_k'], "d": self.ta_params['stoch_d'], "smooth_k": self.ta_params.get('stoch_smooth_k', 3)}
            ]
            custom_ta_strategy = ta.Strategy(name="TradingStrategyTA", ta=custom_ta_list)
            df_bars.ta.strategy(custom_ta_strategy)
        except Exception as e:
            self.logger.error(f"  >> 오류: 기술적 지표 계산 중 오류 발생 ({ticker}): {e}")
            traceback.print_exc(); return None

        latest_features = {}; latest_bar = df_bars.iloc[-1]
        for col in ['open', 'high', 'low', 'close', 'volume']: latest_features[col] = latest_bar.get(col)
        latest_features['amount'] = latest_bar.get('amount', latest_bar['close'] * latest_bar['volume'] if latest_bar.get('close') is not None and latest_bar.get('volume') is not None else 0)

        for length in self.ta_params['sma_lengths']: latest_features[f'SMA_{length}'] = latest_bar.get(f'SMA_{length}')
        latest_features[f'RSI_{self.ta_params["rsi_length"]}'] = latest_bar.get(f'RSI_{self.ta_params["rsi_length"]}')
        macd_suffix = f'{self.ta_params["macd_fast"]}_{self.ta_params["macd_slow"]}_{self.ta_params["macd_signal"]}'
        latest_features[f'MACD_{macd_suffix}'] = latest_bar.get(f'MACD_{macd_suffix}')
        latest_features[f'MACDs_{macd_suffix}'] = latest_bar.get(f'MACDS_{macd_suffix}') # pandas_ta는 MACDS_ 접두사 사용
        latest_features[f'MACDh_{macd_suffix}'] = latest_bar.get(f'MACDH_{macd_suffix}') # pandas_ta는 MACDH_ 접두사 사용

        bb_suffix = f'{self.ta_params["bbands_length"]}_{self.ta_params["bbands_std"]:.1f}' # pandas_ta 컬럼명 형식에 맞춤
        latest_features[f'BBL_{bb_suffix}'] = latest_bar.get(f'BBL_{bb_suffix}')
        latest_features[f'BBM_{bb_suffix}'] = latest_bar.get(f'BBM_{bb_suffix}')
        latest_features[f'BBU_{bb_suffix}'] = latest_bar.get(f'BBU_{bb_suffix}')
        latest_features[f'BBB_{bb_suffix}'] = latest_bar.get(f'BBB_{bb_suffix}')
        latest_features[f'BBP_{bb_suffix}'] = latest_bar.get(f'BBP_{bb_suffix}')
        
        latest_features[f'ATRr_{self.ta_params["atr_length"]}'] = latest_bar.get(f'ATRr_{self.ta_params["atr_length"]}')
        if pd.isna(latest_features[f'ATRr_{self.ta_params["atr_length"]}']): # ATRr_14가 없을 경우 ATR_14 시도
            latest_features[f'ATRr_{self.ta_params["atr_length"]}'] = latest_bar.get(f'ATR_{self.ta_params["atr_length"]}')


        latest_features['OBV'] = latest_bar.get('OBV')
        
        # Stochastic 컬럼명 pandas_ta 기본값: STOCHk_K_D_smoothK, STOCHd_K_D_smoothK
        stoch_col_suffix_in_df = f'{self.ta_params["stoch_k"]}_{self.ta_params["stoch_d"]}_{self.ta_params.get("stoch_smooth_k", 3)}'
        latest_features[f'STOCHk_14_3_3'] = latest_bar.get(f'STOCHk_{stoch_col_suffix_in_df}') # feature_columns_norm에 맞춤
        latest_features[f'STOCHd_14_3_3'] = latest_bar.get(f'STOCHd_{stoch_col_suffix_in_df}') # feature_columns_norm에 맞춤

        latest_features['PBR'] = 0.0; latest_features['PER'] = 0.0; latest_features['USD_KRW'] = 0.0
        latest_features['is_month_end'] = 1 if latest_bar.name.is_month_end else 0

        feature_vector = np.full(NUM_FEATURES, np.nan); missing_keys_log = []
        for i, col_norm_name in enumerate(self.feature_order):
            original_name_key = col_norm_name[:-5] if col_norm_name.endswith('_norm') else col_norm_name
            value_from_latest = latest_features.get(original_name_key)
            if value_from_latest is not None: feature_vector[i] = value_from_latest
            else: missing_keys_log.append(original_name_key)
        
        if missing_keys_log: self.logger.warning(f"({ticker}) 원본 피처 latest_features에 누락: {', '.join(missing_keys_log)}")
        
        temp_features_preview = np.nan_to_num(feature_vector, nan=-999.0) # 전체 벡터 미리보기
        self.logger.debug(f"  >> 계산된 원본 피처 벡터 ({ticker}) (NaN은 -999.0으로 대체): {temp_features_preview[:5]}...") # 첫 5개만 출력

        if np.isnan(feature_vector).any():
            nan_indices = np.where(np.isnan(feature_vector))[0]
            nan_feature_names = [self.feature_order[i] for i in nan_indices]
            self.logger.warning(f"({ticker}) 최종 원본 피처 벡터에 NaN 포함. NaN 피처: {nan_feature_names}. 예측 건너뜀.")
            return None
        return feature_vector

    def normalize_features(self, ticker, feature_vector):
        scalers_for_ticker = self.load_scalers(ticker)
        if not scalers_for_ticker: self.logger.warning(f"({ticker}) 스케일러 없음. 정규화 불가."); return None
        normalized_vector = np.zeros_like(feature_vector, dtype=float); missing_scalers_log = []
        for i, col_norm_name in enumerate(self.feature_order):
            raw_value = feature_vector[i]
            if np.isnan(raw_value):
                self.logger.warning(f"({ticker}) 정규화 입력값 NaN 발견: {col_norm_name}. 0.0으로 처리.")
                normalized_vector[i] = 0.0; continue

            if col_norm_name.endswith('_norm'):
                original_name = col_norm_name[:-5]; scaler = scalers_for_ticker.get(original_name)
                if scaler:
                    try: normalized_vector[i] = scaler.transform([[raw_value]])[0][0]
                    except Exception as e: self.logger.error(f"'{original_name}' 정규화 오류 ({ticker}): {e}. 값: {raw_value}. 0.0으로 처리."); normalized_vector[i] = 0.0
                else: missing_scalers_log.append(original_name); normalized_vector[i] = raw_value # 스케일러 없으면 원본값 (또는 0)
            else: normalized_vector[i] = raw_value # 정규화 안하는 피처 (is_month_end 등)
        
        if missing_scalers_log: self.logger.warning(f"({ticker}) 스케일러 누락 피처 (원본값 사용): {', '.join(missing_scalers_log)}")
        if np.isnan(normalized_vector).any():
            self.logger.error(f"({ticker}) 정규화 후 NaN 발생. 예측 건너뜀. 벡터: {normalized_vector}"); return None
        self.logger.debug(f"({ticker}) 정규화된 피처 (첫 5개): {normalized_vector[:5]}")
        return normalized_vector

    def process_new_bar(self, ticker, bar_data):
        # 이 함수는 self.logger를 사용하므로, __init__에서 self.logger가 먼저 할당되어야 함
        if not hasattr(self, 'logger'):
            print(f"CRITICAL: Logger not initialized in process_new_bar for {ticker}")
            return # 로거 없으면 더 이상 진행 불가

        self.logger.info(f"  >> 1분봉 완성 ({ticker}): Time={bar_data['time']}, O={bar_data['open']}, H={bar_data['high']}, L={bar_data['low']}, C={bar_data['close']}, V={bar_data['volume']}")
        if ticker not in self.completed_bars:
            max_lookback = max(self.ta_params['sma_lengths'] + [self.ta_params['rsi_length'], self.ta_params['bbands_length'], self.ta_params['macd_slow'] + self.ta_params['macd_signal'], self.ta_params['atr_length'], self.ta_params['stoch_k'] + self.ta_params['stoch_d']])
            self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + max_lookback + 100)
        self.completed_bars[ticker].append(bar_data)
        self.logger.debug(f"({ticker}) completed_bars에 새 봉 추가. 현재 길이: {len(self.completed_bars[ticker])}")

        raw_feature_vector = self.calculate_features(ticker)
        if raw_feature_vector is None: return

        normalized_features = self.normalize_features(ticker, raw_feature_vector)
        if normalized_features is None: return

        if ticker not in self.feature_sequences: self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS)
        self.feature_sequences[ticker].append(normalized_features)
        self.logger.debug(f"({ticker}) 피처 시퀀스 업데이트. 현재 크기: {len(self.feature_sequences[ticker])}/{TIME_STEPS}")

        if len(self.feature_sequences[ticker]) == TIME_STEPS:
            model = self.load_model(ticker)
            if model:
                input_seq = np.array(list(self.feature_sequences[ticker])).astype(np.float32)
                input_seq = np.expand_dims(input_seq, axis=0)
                if input_seq.shape == (1, TIME_STEPS, NUM_FEATURES):
                    try:
                        prediction = model.predict(input_seq, verbose=0)
                        predicted_norm_value = prediction[0][0]
                        self.logger.info(f" ★★★ 예측 결과 ({ticker}): {predicted_norm_value:.6f} ★★★")
                        # 예시: 삼성전자(005930) 및 임계값 조건 매매
                        if ticker == '005930' and not self.trade_triggered.get(ticker, False):
                            current_price = bar_data['close'] # 현재 종가
                            if predicted_norm_value > 0.65: # 매수 조건 (임계값 조정 필요)
                                self.logger.info(f"★★★ 매수 조건 충족 ({ticker}): 예측값={predicted_norm_value:.6f} > 0.65 ★★★")
                                # asyncio.create_task(self.send_order("buy", ticker, 1, price=0, kiwoom_order_cond="03")) # 시장가 매수 (키움 모의는 지정가만 가능할 수 있음)
                                # self.trade_triggered[ticker] = True
                            elif predicted_norm_value < 0.35: # 매도 조건 (임계값 조정 필요)
                                self.logger.info(f"★★★ 매도 조건 충족 ({ticker}): 예측값={predicted_norm_value:.6f} < 0.35 ★★★")
                                # asyncio.create_task(self.send_order("sell", ticker, 1, price=0, kiwoom_order_cond="03")) # 시장가 매도
                                # self.trade_triggered[ticker] = True
                    except Exception as e: self.logger.error(f"({ticker}) 예측/주문 오류: {e}"); traceback.print_exc()
                else: self.logger.error(f"({ticker}) 모델 입력 형태 불일치: {input_seq.shape}, 필요: (1, {TIME_STEPS}, {NUM_FEATURES})")
            else: self.logger.warning(f"({ticker}) 모델 미로드. 예측 불가.")
        else: self.logger.debug(f"({ticker}) 예측 위한 피처 시퀀스 부족: {len(self.feature_sequences[ticker])}/{TIME_STEPS}")

    async def _aggregate_bar(self, code, trade_time_str, trade_price_str, trade_volume_str):
        try:
            # 이 함수는 receive_messages 내부에서 호출되므로 self.logger는 이미 존재한다고 가정
            trade_time_str = str(trade_time_str).strip()
            price_str_cleaned = str(trade_price_str).strip().replace('+', '').replace('-', '')
            volume_str_cleaned = str(trade_volume_str).strip().replace('+', '').replace('-', '')

            if not all([trade_time_str, price_str_cleaned, volume_str_cleaned]):
                self.logger.warning(f"({code}) 집계용 필수 데이터 누락: T='{trade_time_str}', P='{trade_price_str}', V='{trade_volume_str}'")
                return
            if len(trade_time_str) != 6:
                self.logger.warning(f"({code}) 집계용 시간 형식 오류: '{trade_time_str}'"); return

            now_dt = datetime.now()
            current_dt = now_dt.replace(hour=int(trade_time_str[:2]), minute=int(trade_time_str[2:4]), second=int(trade_time_str[4:6]), microsecond=0)
            price = float(price_str_cleaned); volume = abs(int(volume_str_cleaned)) # 거래량은 항상 양수

            if price <= 0: self.logger.warning(f"({code}) 유효하지 않은 가격: {price}"); return
            if volume < 0: self.logger.warning(f"({code}) 유효하지 않은 거래량: {volume}"); return # 0은 허용

            current_minute_start = current_dt.replace(second=0, microsecond=0)
            minute_val = current_minute_start.minute
            bar_start_minute = (minute_val // self.bar_interval_minutes) * self.bar_interval_minutes
            current_interval_start = current_minute_start.replace(minute=bar_start_minute)

            if code not in self.current_bars or self.current_bars[code]['time'] < current_interval_start:
                if code in self.current_bars and self.current_bars[code].get('open') is not None:
                    completed_bar_data = self.current_bars[code].copy()
                    self.process_new_bar(code, completed_bar_data)
                
                self.current_bars[code] = {'time': current_interval_start, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume, 'amount': price * volume}
                self.logger.debug(f"({code}) 새 분봉 시작: {self.current_bars[code]}")
            else:
                bar = self.current_bars[code]
                bar['high'] = max(bar['high'], price); bar['low'] = min(bar['low'], price)
                bar['close'] = price; bar['volume'] += volume; bar['amount'] += price * volume
        except ValueError as ve: self.logger.error(f"({code}) 실시간 값 변환 오류 (집계): {ve}. Values: T='{trade_time_str}', P='{trade_price_str}', V='{trade_volume_str}'")
        except Exception as e: self.logger.error(f"({code}) 1분봉 집계 중 알 수 없는 오류: {e}"); traceback.print_exc()

    async def receive_messages(self):
        while self.keep_running:
            if not self.connected:
                self.logger.info(f"연결 끊김. {self.reconnect_attempts + 1}번째 재연결 시도...")
                await asyncio.sleep(min(30, 2 ** self.reconnect_attempts))
                self.reconnect_attempts += 1
                if self.reconnect_attempts > self.max_reconnect_attempts:
                    self.logger.error("최대 재연결 시도 실패. 프로그램 종료."); self.keep_running = False; break
                await self.connect(); continue

            try:
                response_str = await asyncio.wait_for(self.websocket.recv(), timeout=70.0)
                self.logger.debug(f"Kiwoom RAW 수신: {response_str[:200]}")
                if not response_str.strip(): continue

                response = json.loads(response_str)
                trnm = response.get('trnm')
                # 키움증권은 header/body 구조 대신 직접 필드 사용
                return_code = response.get('return_code') # LOGIN 응답용
                msg = response.get('return_msg', response.get('msg', '')) # LOGIN, REG 응답용

                if trnm == 'LOGIN':
                    if str(return_code) == '0': # 성공 코드 문자열 '0'
                        self.logger.info(f"*** Kiwoom WebSocket LOGIN 성공: {msg} ***")
                        self.login_success_ws = True; self.reconnect_attempts = 0
                        await self.register_realtime()
                    else:
                        self.logger.error(f"!!! Kiwoom WebSocket LOGIN 실패: Code={return_code}, Msg={msg} !!!")
                        await self.disconnect(reconnect=False); break
                elif trnm == 'PING':
                    self.logger.info(f"Kiwoom PING 수신: {response}. PONG 전송.")
                    await self.send_message({"trnm": "PONG"}) # 키움 형식에 맞춘 PONG
                elif trnm == 'REG':
                    self.logger.info(f"Kiwoom 실시간 등록 응답: Code={return_code}, Msg={msg}, Data={response.get('data')}")
                elif trnm == 'REAL':
                    data_list = response.get('data')
                    if isinstance(data_list, list):
                        for real_data in data_list:
                            item_code = real_data.get('item')
                            data_type = real_data.get('type') # "0B" 또는 "0A"
                            values = real_data.get('values')
                            if not item_code or not data_type or not values:
                                self.logger.warning(f"Kiwoom REAL 데이터 필드 누락: {real_data}"); continue
                            
                            self.logger.debug(f"REAL 데이터 상세: item={item_code}, type={data_type}, values_keys={list(values.keys()) if isinstance(values, dict) else 'N/A'}")

                            if data_type == '0B': # 주식 체결
                                raw_trade_time = values.get('20'); raw_trade_price = values.get('10'); raw_trade_volume = values.get('15')
                                self.logger.debug(f"RAW Fields ({item_code}): Time='{raw_trade_time}', Price='{raw_trade_price}', Vol='{raw_trade_volume}'")
                                await self._aggregate_bar(item_code, raw_trade_time, raw_trade_price, raw_trade_volume)
                            elif data_type == '0A': # 주식 호가
                                self.logger.debug(f"Kiwoom 호가 ({item_code}): {values}") # 호가 처리 로직 (필요시)
                    else: self.logger.warning(f"Kiwoom REAL 데이터 형식 오류 ('data' 필드): {response}")
                else: self.logger.info(f"Kiwoom 기타 응답 (trnm={trnm}): {response}")

            except asyncio.TimeoutError: self.logger.warning("Kiwoom 메시지 수신 타임아웃."); await self.send_message({"trnm":"PING"}) # 타임아웃 시 PING
            except websockets.exceptions.ConnectionClosedOK: self.logger.info("Kiwoom WebSocket 정상 종료."); self.connected = False; self.login_success_ws = False
            except websockets.exceptions.ConnectionClosedError as e: self.logger.error(f"Kiwoom WebSocket 비정상 종료: {e}"); self.connected = False; self.login_success_ws = False
            except json.JSONDecodeError: self.logger.error(f"Kiwoom JSON 파싱 오류: {response_str[:200]}")
            except Exception as e: self.logger.error(f"Kiwoom 메시지 처리 중 예외: {e}"); traceback.print_exc(); await asyncio.sleep(1)


    async def register_realtime(self):
        if not self.login_success_ws: self.logger.warning("Kiwoom 로그인 실패. 실시간 등록 불가."); return
        self.logger.info("실시간 데이터 등록 요청 (Kiwoom)...")
        for reg_config in REGISTER_LIST:
            # 키움 API는 data 필드에 list of dictionaries 형태로 item과 type을 전달
            data_payload = [{"item": reg_config["item"], "type": reg_config["type"]}]
            kiwoom_reg_param = {"trnm": "REG", "grp_no": "1", "refresh": "1", "data": data_payload}
            await self.send_message(kiwoom_reg_param)
            self.logger.info(f"Kiwoom 실시간 등록 요청: {kiwoom_reg_param}")
            await asyncio.sleep(0.2) # API 요청 간격
        self.logger.info("Kiwoom 실시간 데이터 등록 요청 완료.")

    async def _send_order_async_wrapper(self, order_type, stock_code, quantity, price, kiwoom_order_cond):
        # REST API용 토큰 가져오기 또는 발급
        current_rest_token = self._get_rest_access_token()
        if not current_rest_token:
            self.logger.error("주문 불가: REST API 접근 토큰을 가져올 수 없습니다.")
            return {"rt_cd": "E99", "msg1": "REST API 토큰 없음"}

        # 키움 OpenAPI+ 현금 주문 TRID
        tr_id_rest = ""
        if IS_MOCK_TRADING:
            if order_type.lower() == 'buy':  tr_id_rest = "VTTC0802U"  # 모의투자 매수
            elif order_type.lower() == 'sell': tr_id_rest = "VTTC0801U" # 모의투자 매도
        else: # 실전 투자
            if order_type.lower() == 'buy':  tr_id_rest = "TTTC0802U"  # 실전투자 매수 [실제 TRID 확인 필요]
            elif order_type.lower() == 'sell': tr_id_rest = "TTTC0801U" # 실전투자 매도 [실제 TRID 확인 필요]

        if not tr_id_rest:
            self.logger.error(f"유효하지 않은 주문 TRID 설정: order_type={order_type}, IS_MOCK_TRADING={IS_MOCK_TRADING}")
            return {"rt_cd": "E95", "msg1": "주문 TRID 설정 오류"}

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {current_rest_token}",
            "appkey": self.rest_app_key,
            "appsecret": self.rest_app_secret,
            "tr_id": tr_id_rest,
            "custtype": "P"
        }
        ord_dvsn_code = "01" if kiwoom_order_cond == "03" else "00" # 시장가 "01", 지정가 "00" (키움 기준)
        order_price = "0" if ord_dvsn_code == "01" else str(price)

        body = {
            "CANO": self.account_no_prefix,
            "ACNT_PRDT_CD": self.account_no_suffix,
            "PDNO": stock_code,
            "ORD_DVSN": ord_dvsn_code,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": order_price,
        }
        loop = asyncio.get_running_loop()
        response_data = await loop.run_in_executor(None, functools.partial(self._send_order_sync, tr_id_rest, headers, body))
        return response_data

    async def send_order(self, order_type, stock_code, quantity, price=0, kiwoom_order_cond="00"):
        self.logger.info(f"--- Kiwoom {stock_code} {order_type.upper()} 주문 ({kiwoom_order_cond}) 전송 준비 --- Qty:{quantity}, Price:{price}")
        try:
            response_data = await self._send_order_async_wrapper(order_type, stock_code, quantity, price, kiwoom_order_cond)
            if response_data and response_data.get('rt_cd') == '0':
                order_no = response_data.get('output', {}).get('ODNO', 'N/A')
                self.logger.info(f" ★★★ Kiwoom 주문 성공! 주문번호: {order_no} (Msg: {response_data.get('msg1')}) ★★★")
            else:
                self.logger.error(f" !!! Kiwoom 주문 실패 !!! 응답: {response_data}")
            return response_data
        except Exception as e:
            self.logger.error(f"Kiwoom 주문 전송 중 최상위 오류: {e}"); traceback.print_exc()
            return {"rt_cd": "E94", "msg1": str(e)}

    async def run(self):
        # REST API 토큰 초기 발급 시도
        if not self._get_rest_access_token() and not IS_MOCK_TRADING : # 실전 투자 시 토큰 필수
             self.logger.error("실전 투자 환경에서 REST API 토큰 발급 실패. 프로그램 종료.")
             return

        while self.keep_running:
            if not self.connected or not self.login_success_ws:
                await self.connect() # 연결 및 로그인 시도
                if not self.connected or not self.login_success_ws: # 여전히 실패 시
                    self.logger.info(f"연결/로그인 실패. {min(30, 2 ** self.reconnect_attempts)}초 후 재시도...")
                    await asyncio.sleep(min(30, 2 ** self.reconnect_attempts))
                    self.reconnect_attempts = min(self.reconnect_attempts + 1, 6)
                    if self.reconnect_attempts > self.max_reconnect_attempts :
                         self.logger.error("최대 재연결 시도 실패. 프로그램 종료.")
                         self.keep_running = False
                    continue # 루프 처음으로 돌아가 재시도
            
            # 연결 및 로그인 성공 상태
            try:
                await self.receive_messages()
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("receive_messages 중 연결 끊김.")
                self.connected = False; self.login_success_ws = False # 상태 초기화
            except Exception as e:
                self.logger.error(f"receive_messages 루프 중 예외: {e}")
                traceback.print_exc()
                self.connected = False; self.login_success_ws = False # 오류 발생 시 연결 상태 재설정
                await asyncio.sleep(5) # 짧은 대기 후 재연결 시도

        self.logger.info("KiwoomStockWSClient run 루프 종료.")
        await self.disconnect(reconnect=False)

    async def disconnect(self, reconnect=True):
        self.logger.info(f"Kiwoom 클라이언트 disconnect 호출 (reconnect={reconnect})...")
        if not reconnect: self.keep_running = False
        
        if self.websocket and self.connected:
            self.logger.info("Kiwoom WebSocket 연결 종료 시도...")
            try:
                # 실시간 등록 해제 (선택 사항, API 명세에 따라 필요할 수 있음)
                # for reg_config in REGISTER_LIST:
                #     for stock_code in reg_config["item"]:
                #         for reg_type in reg_config["type"]:
                #             unreg_param = {"trnm": "REMOVE", "grp_no": "1", "data": [{"item": [stock_code], "type": [reg_type]}]}
                #             await self.send_message(unreg_param)
                #             self.logger.info(f"Kiwoom 실시간 해제 요청: {unreg_param}")
                await self.websocket.close()
                self.logger.info('Kiwoom WebSocket 연결이 정상적으로 닫혔습니다.')
            except Exception as e: self.logger.error(f"Kiwoom WebSocket 연결 닫는 중 오류: {e}")
            finally: self.websocket = None; self.connected = False; self.login_success_ws = False
        else: self.logger.info("Kiwoom WebSocket이 이미 연결 해제되었거나 없습니다.")


# --- main 함수 및 실행 부분 ---
async def main():
    # 필수 설정값 검사 (웹소켓 로그인 토큰, 계좌번호, REST API용 앱키/시크릿)
    if KIWOOM_WEBSOCKET_LOGIN_TOKEN == "5G59I5s9zCgsRjI5MtnPyG-qOqlYrBwH-xkRy-1nnjKfYyNEmURVVpucxB5QlZowxIl1kG307O1y5RhvvM9CRw": # 기본값이라면 경고
        print("경고: KIWOOM_WEBSOCKET_LOGIN_TOKEN이 기본값입니다. 실제 토큰으로 변경해주세요.")
    # 나머지 설정값은 이미 코드 상단에 정의되어 있음

    client = None
    try:
        client = KiwoomStockWSClient(
            socket_uri=KIWOOM_SOCKET_URL,
            ws_login_token=KIWOOM_WEBSOCKET_LOGIN_TOKEN,
            rest_app_key=KIWOOM_APP_KEY,
            rest_app_secret=KIWOOM_APP_SECRET,
            account_prefix=KIWOOM_ACCOUNT_NO_PREFIX,
            account_suffix=KIWOOM_ACCOUNT_NO_SUFFIX,
            kiwoom_user_id="", # 키움 OpenAPI+ 웹소켓 LOGIN TRNM에 user_id가 명시적으로 필요 없다면 빈 문자열 또는 None
            is_dev_mode=IS_MOCK_TRADING,
            monitored_tickers_list=[reg["item"][0] for reg in REGISTER_LIST if reg.get("item")],
            log_file_path="trading_log_kiwoom.log"
        )
        await client.run()
    except KeyboardInterrupt:
        print("\nCtrl+C 감지 (main). Kiwoom 클라이언트 종료 중...")
        if client: await client.disconnect(reconnect=False)
    except Exception as e:
        print(f"main 함수 실행 중 예외 발생: {e}")
        traceback.print_exc()
        if client: await client.disconnect(reconnect=False)
    finally:
        print("Kiwoom 메인 프로그램 루틴 최종 종료.")

if __name__ == '__main__':
    log_dir_main = os.path.dirname("trading_log_kiwoom.log")
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
    finally: print("Kiwoom WebSocket 클라이언트 프로그램 최종적으로 모두 종료됨.")