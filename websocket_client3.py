# -*- coding: utf-8 -*-
# 파일명: websocket_client.py (상세 피처 계산 및 정규화)
# 실행 환경: 64비트 Python (myenv)
# 필요 라이브러리: pip install websockets tensorflow pandas numpy requests pandas_ta joblib

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
import signal
import functools
from datetime import datetime, timedelta
import joblib # ★★★ 스케일러 로딩 위해 추가 ★★★
import pandas_ta as ta # ★★★ 기술적 지표 계산 위해 추가 ★★★

# --- 설정 ---
ACCESS_TOKEN = "5G59I5s9zCgsRjI5MtnPyG-qOqlYrBwH-xkRy-1nnjKfYyNEmURVVpucxB5QlZowxIl1kG307O1y5RhvvM9CRw" # 실제 발급받은 토큰으로 변경
ACCOUNT_NO = 81026247 
APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss"  
APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw"

IS_MOCK_TRADING = True # 모의투자 사용 여부

# --- 경로 설정 ---
MODEL_SAVE_PATH = r'G:\내 드라이브\lstm_models_per_stock_v1'
SCALER_SAVE_PATH = r'G:\내 드라이브\processed_stock_data_full_v1\scalers' # ★★★ 스케일러 저장 경로 ★★★
# DRIVE_DATA_PATH = r'G:\내 드라이브\processed_stock_data_full_v1\processed_parquet' # 과거 데이터 로딩 필요시

# --- 모델 및 피처 설정 ---
TIME_STEPS = 20
FEATURE_COLUMNS_NORM = [ # 모델 입력 피처 (정규화된 이름)
    'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 'amount_norm',
    'SMA_5_norm', 'SMA_20_norm', 'SMA_60_norm', 'SMA_120_norm', 'RSI_14_norm',
    'MACD_12_26_9_norm', 'MACDs_12_26_9_norm', 'MACDh_12_26_9_norm',
    'BBL_20_2.0_norm', 'BBM_20_2.0_norm', 'BBU_20_2.0_norm', 'BBB_20_2.0_norm', 'BBP_20_2.0_norm',
    'ATRr_14_norm', 'OBV_norm', 'STOCHk_14_3_3_norm', 'STOCHd_14_3_3_norm',
    'PBR_norm', 'PER_norm', 'USD_KRW_norm', 'is_month_end'
]
# 정규화가 필요한 원본 피처 이름 목록 (스케일러 참조 및 계산용)
# '_norm' 접미사를 제거하고, 원본 OHLCV 및 계산될 지표 이름 포함
# amount, is_month_end 등 일부는 정규화 안 할 수 있음
FEATURES_TO_CALC_AND_NORM = [
    'open', 'high', 'low', 'close', 'volume', 'amount', # 기본 OHLCV, 거래대금
    'SMA_5', 'SMA_20', 'SMA_60', 'SMA_120', 'RSI_14', # SMA, RSI
    'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', # MACD
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', # Bollinger Bands
    'ATRr_14', # ATR
    'OBV', # OBV
    'STOCHk_14_3_3', 'STOCHd_14_3_3', # Stochastic
    'PBR', 'PER', # 재무 (Placeholder)
    'USD_KRW' # 거시 (Placeholder)
    # 'is_month_end' # 정규화 안 할 피처는 여기서 제외하거나 아래 로직에서 처리
]
NUM_FEATURES = len(FEATURE_COLUMNS_NORM) # 최종 모델 입력 피처 개수

# --- API URL 설정 ---
if IS_MOCK_TRADING:
    SOCKET_URL = "wss://mockapi.kiwoom.com:10000/api/dostk/websocket"
    REST_BASE_URL = "https://mockapi.kiwoom.com"
    print(">> 모의투자 환경으로 설정되었습니다.")
else:
    SOCKET_URL = "wss://api.kiwoom.com:10000/api/dostk/websocket"
    REST_BASE_URL = "https://api.kiwoom.com"
    print(">> 실전투자 환경으로 설정되었습니다.")

API_URL_ORDER = f"{REST_BASE_URL}/api/dostk/ordr"

# --- 실시간 등록 설정 ---
REGISTER_LIST = [ {"item": ["005930"], "type": ["0A", "0B"]} ]
# -----------------------

class WebSocketClient:
    def __init__(self, uri, token, account_no, app_key, app_secret):
        self.uri = uri; self.token = token; self.account_no = account_no;
        self.app_key = app_key; self.app_secret = app_secret;
        self.websocket = None; self.connected = False; self.keep_running = True
        self.login_success = False; self.models = {}; self.trade_triggered = {}

        # --- 데이터 처리용 변수 초기화 ---
        self.bar_interval_minutes = 1
        self.current_bars = {}
        # ★★★ 완료된 봉 저장 deque 크기 늘림 (지표 계산 위해) ★★★
        self.completed_bars = {} # {'ticker': deque(maxlen=200)} # 예: 120 SMA + 알파
        self.feature_sequences = {} # {'ticker': deque(maxlen=TIME_STEPS)}
        self.scalers = {} # 로드된 스케일러 저장 {'ticker': {'feature_name': scaler_object, ...}}
        self.feature_order = FEATURE_COLUMNS_NORM # 최종 피처 순서

        # 기술적 지표 계산 파라미터
        self.ta_params = {
            'sma_lengths': [5, 20, 60, 120], 'rsi_length': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bbands_length': 20, 'bbands_std': 2, 'atr_length': 14,
            'stoch_k': 14, 'stoch_d': 3
        }
        # -------------------------------

    # ... (connect, send_message 메소드는 이전과 동일) ...
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri, ping_interval=60, ping_timeout=20)
            self.connected = True; print("WebSocket 서버 연결 성공.")
            login_param = {'trnm': 'LOGIN', 'token': self.token}; print('로그인 패킷 전송 시도...')
            await self.send_message(login_param)
        except Exception as e: print(f'WebSocket 연결 오류: {e}'); self.connected = False

    async def send_message(self, message):
        if self.connected and self.websocket:
            if not isinstance(message, str): message_str = json.dumps(message)
            else: message_str = message
            await self.websocket.send(message_str)
        else: print("WebSocket 미연결 상태")

    def load_model(self, ticker):
        # ... (이전과 동일) ...
        if ticker in self.models: return self.models[ticker]
        model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}.keras");
        if os.path.exists(model_path):
            try: self.models[ticker] = tf.keras.models.load_model(model_path, compile=False); print(f"모델 로딩 성공: {ticker}"); return self.models[ticker]
            except Exception as e: print(f"모델 로딩 실패 ({ticker}): {e}"); return None
        else: print(f"모델 파일 없음: {model_path}"); return None

    def load_scalers(self, ticker):
        """ 저장된 스케일러 객체 로드 """
        if ticker in self.scalers:
            return self.scalers[ticker]
        else:
            scaler_path = os.path.join(SCALER_SAVE_PATH, f'{ticker}_scalers.joblib')
            if os.path.exists(scaler_path):
                try:
                    print(f"  >> 스케일러 로딩 시도: {scaler_path}")
                    loaded_scaler_dict = joblib.load(scaler_path)
                    self.scalers[ticker] = loaded_scaler_dict
                    print(f"  >> 스케일러 로딩 성공 ({ticker}): {len(loaded_scaler_dict)}개 피처")
                    return self.scalers[ticker]
                except Exception as e:
                    print(f"  >> 오류: 스케일러 로딩 실패 ({ticker}): {e}")
                    self.scalers[ticker] = {} # 빈 딕셔너리로 저장하여 반복 로딩 시도 방지
                    return {}
            else:
                print(f"  >> 경고: 스케일러 파일 없음 ({ticker}): {scaler_path}")
                self.scalers[ticker] = {}
                return {}

    def calculate_features(self, ticker):
        """ 완성된 분봉 데이터를 기반으로 모든 기술적 지표 계산 """
        if ticker not in self.completed_bars or len(self.completed_bars[ticker]) < max(self.ta_params['sma_lengths'], self.ta_params['bbands_length'], self.ta_params['stoch_k']) + 5: # 대략적인 최소 요구 길이
            # print(f"  >> 피처 계산 위한 봉 데이터 부족 ({ticker}): {len(self.completed_bars.get(ticker, []))}")
            return None # 데이터 부족

        recent_bars_list = list(self.completed_bars[ticker])
        df_bars = pd.DataFrame(recent_bars_list).set_index('time')
        if not all(col in df_bars.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            print("  >> 오류: 봉 데이터에 OHLCV 컬럼이 없습니다.")
            return None

        print(f"  >> 기술적 지표 계산 시작 ({ticker})...")
        try:
            # Pandas TA Strategy 정의 (한번에 여러 지표 계산)
            custom_ta = ta.Strategy(
                name="TradingStrategyTA",
                ta=[
                    {"kind": "sma", "length": l} for l in self.ta_params['sma_lengths']
                ] + [
                    {"kind": "rsi", "length": self.ta_params['rsi_length']},
                    {"kind": "macd", "fast": self.ta_params['macd_fast'], "slow": self.ta_params['macd_slow'], "signal": self.ta_params['macd_signal']},
                    {"kind": "bbands", "length": self.ta_params['bbands_length'], "std": self.ta_params['bbands_std']},
                    {"kind": "atr", "length": self.ta_params['atr_length']},
                    {"kind": "obv"},
                    {"kind": "stoch", "k": self.ta_params['stoch_k'], "d": self.ta_params['stoch_d']}
                ]
            )
            # 지표 계산 적용
            df_bars.ta.strategy(custom_ta)
            # print(f"  >> 계산된 지표 컬럼: {df_bars.columns.tolist()}") # 디버깅용

        except Exception as e:
            print(f"  >> 오류: 기술적 지표 계산 중 오류 발생 ({ticker}): {e}")
            traceback.print_exc()
            return None

        # --- 최신 피처 값 추출 및 벡터 생성 ---
        latest_features = {}
        latest_bar = df_bars.iloc[-1] # 가장 최근 완성된 봉 데이터

        # OHLCV 값
        for col in ['open', 'high', 'low', 'close', 'volume']:
            latest_features[col] = latest_bar[col]
        # 거래대금 (만약 분봉 데이터에 있다면)
        latest_features['amount'] = latest_bar.get('amount', 0) # 없으면 0 (또는 다르게 처리)

        # 계산된 기술적 지표 값 (pandas_ta가 생성한 컬럼 이름 확인 필요!)
        for length in self.ta_params['sma_lengths']: latest_features[f'SMA_{length}'] = latest_bar.get(f'SMA_{length}')
        latest_features[f'RSI_{self.ta_params["rsi_length"]}'] = latest_bar.get(f'RSI_{self.ta_params["rsi_length"]}')
        # MACD 컬럼 이름 확인 (예: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9)
        macd_suffix = f'{self.ta_params["macd_fast"]}_{self.ta_params["macd_slow"]}_{self.ta_params["macd_signal"]}'
        latest_features[f'MACD_{macd_suffix}'] = latest_bar.get(f'MACD_{macd_suffix}')
        latest_features[f'MACDs_{macd_suffix}'] = latest_bar.get(f'MACDs_{macd_suffix}')
        latest_features[f'MACDh_{macd_suffix}'] = latest_bar.get(f'MACDh_{macd_suffix}')
        # BBands 컬럼 이름 확인 (예: BBL_20_2.0, BBM_20_2.0 등)
        bb_suffix = f'{self.ta_params["bbands_length"]}_{float(self.ta_params["bbands_std"])}' # std가 float일 수 있음
        latest_features[f'BBL_{bb_suffix}'] = latest_bar.get(f'BBL_{bb_suffix}')
        latest_features[f'BBM_{bb_suffix}'] = latest_bar.get(f'BBM_{bb_suffix}')
        latest_features[f'BBU_{bb_suffix}'] = latest_bar.get(f'BBU_{bb_suffix}')
        latest_features[f'BBB_{bb_suffix}'] = latest_bar.get(f'BBB_{bb_suffix}')
        latest_features[f'BBP_{bb_suffix}'] = latest_bar.get(f'BBP_{bb_suffix}')
        # ATR 컬럼 이름 확인 (예: ATRt_14, ATRf_14, ATRr_14) - ATRr (True Range?) 사용 가정
        latest_features[f'ATRr_{self.ta_params["atr_length"]}'] = latest_bar.get(f'ATRr_{self.ta_params["atr_length"]}')
        # OBV 컬럼 이름 확인 (예: OBV)
        latest_features['OBV'] = latest_bar.get('OBV')
        # Stochastic 컬럼 이름 확인 (예: STOCHk_14_3_3, STOCHd_14_3_3)
        stoch_suffix = f'{self.ta_params["stoch_k"]}_{self.ta_params["stoch_d"]}_3' # 마지막은 보통 3(smooth_k)
        latest_features[f'STOCHk_{stoch_suffix}'] = latest_bar.get(f'STOCHk_{stoch_suffix}')
        latest_features[f'STOCHd_{stoch_suffix}'] = latest_bar.get(f'STOCHd_{stoch_suffix}')

        # ★★★ 외부 피처 (PBR, PER, USD_KRW) 는 현재 임시값(0) 사용 ★★★
        #     실제 값은 별도 메커니즘으로 가져와서 업데이트 필요
        latest_features['PBR'] = 0.0 # 임시
        latest_features['PER'] = 0.0 # 임시
        latest_features['USD_KRW'] = 0.0 # 임시

        # ★★★ 이벤트 피처 (is_month_end) ★★★
        latest_features['is_month_end'] = 1 if latest_bar.name.is_month_end else 0

        # --- 최종 피처 벡터 생성 (self.feature_order 순서 기준) ---
        feature_vector = np.zeros(NUM_FEATURES) * np.nan # NaN으로 초기화
        for i, col_norm_name in enumerate(self.feature_order):
            if col_norm_name.endswith('_norm'):
                original_name = col_norm_name[:-5] # '_norm' 제거
                value = latest_features.get(original_name)
                if value is not None and not np.isnan(value): # None이나 NaN이 아닌 경우
                    feature_vector[i] = value
                else:
                    # print(f"  >> 경고: 피처 값 누락 ({ticker}) - {original_name}. NaN으로 설정.")
                    pass # 이미 NaN으로 초기화됨
            elif col_norm_name == 'is_month_end': # 정규화 안 하는 피처
                feature_vector[i] = latest_features.get('is_month_end', 0)
            # 다른 정규화 안하는 피처가 있다면 여기에 추가

        print(f"  >> 계산된 원본 피처 벡터 ({ticker}): {np.nan_to_num(feature_vector[:5], nan=-999):.2f}...") # NaN은 -999로 표시

        # NaN 값이 남아있는지 확인 (지표 계산 불가 등의 이유)
        if np.isnan(feature_vector).any():
             print(f"  >> 경고: 최종 피처 벡터에 NaN 포함됨 ({ticker}). 예측 건너<0xEB><0x8A>.")
             return None

        return feature_vector # 정규화 전의 원본 피처 벡터 반환

    def normalize_features(self, ticker, feature_vector):
        """ 저장된 스케일러를 사용하여 피처 벡터 정규화 """
        scalers_for_ticker = self.load_scalers(ticker)
        if not scalers_for_ticker: # 스케일러 파일 없거나 로딩 실패
            print(f"  >> 경고: 스케일러 없음 ({ticker}). 정규화 불가.")
            return None # 정규화 불가 시 None 반환

        normalized_vector = np.zeros_like(feature_vector)

        for i, col_norm_name in enumerate(self.feature_order):
            if col_norm_name.endswith('_norm'):
                original_name = col_norm_name[:-5]
                scaler = scalers_for_ticker.get(original_name)
                if scaler:
                    try:
                        raw_value = feature_vector[i]
                        # scaler.transform은 2D 배열을 기대 -> reshape
                        # NaN 값이 없다고 가정 (calculate_features에서 처리됨)
                        normalized_value = scaler.transform([[raw_value]])[0][0]
                        normalized_vector[i] = normalized_value
                    except Exception as e:
                        print(f"  >> 오류: '{original_name}' 정규화 중 오류 ({ticker}): {e}")
                        normalized_vector[i] = 0 # 오류 시 0으로 대체 (또는 다른 값)
                else:
                    # print(f"  >> 경고: '{original_name}' 스케일러 없음 ({ticker}). 0으로 설정.")
                    normalized_vector[i] = 0 # 스케일러 없으면 0으로
            elif col_norm_name == 'is_month_end': # 정규화 안 하는 피처
                normalized_vector[i] = feature_vector[i]
            else: # 그 외 정규화 안하는 피처 (있다면)
                 normalized_vector[i] = feature_vector[i] # 일단 원본값 복사

        print(f"  >> 정규화된 피처 벡터 ({ticker}): {normalized_vector[:5]:.4f}...")
        return normalized_vector

    def process_new_bar(self, ticker, bar_data):
         """ 완성된 1분봉 데이터 처리 파이프라인 """
         print(f"  >> 1분봉 완성 ({ticker}): Time={bar_data['time']}, O={bar_data['open']}, H={bar_data['high']}, L={bar_data['low']}, C={bar_data['close']}, V={bar_data['volume']}")

         if ticker not in self.completed_bars:
              self.completed_bars[ticker] = deque(maxlen=TIME_STEPS + 120) # 지표 계산 위해 충분히 길게
         self.completed_bars[ticker].append(bar_data)

         # 1. 원본 피처 벡터 계산
         raw_feature_vector = self.calculate_features(ticker)
         if raw_feature_vector is None: return

         # 2. 피처 정규화
         normalized_features = self.normalize_features(ticker, raw_feature_vector)
         if normalized_features is None: return

         # 3. 모델 입력 시퀀스 버퍼에 추가
         if ticker not in self.feature_sequences:
             self.feature_sequences[ticker] = deque(maxlen=TIME_STEPS)
         self.feature_sequences[ticker].append(normalized_features)
         # print(f"  >> 피처 시퀀스 버퍼 업데이트 ({ticker}): 크기={len(self.feature_sequences[ticker])}/{TIME_STEPS}")

         # 4. 시퀀스 완성 시 예측 수행
         if len(self.feature_sequences[ticker]) == TIME_STEPS:
             model = self.load_model(ticker)
             if model:
                 input_seq = np.array(list(self.feature_sequences[ticker])).astype(np.float32)
                 input_seq = np.expand_dims(input_seq, axis=0)

                 if input_seq.shape == (1, TIME_STEPS, NUM_FEATURES):
                     try:
                         prediction = model.predict(input_seq, verbose=0)
                         predicted_norm_value = prediction[0][0]
                         print(f"  ★★★ 예측 결과 ({ticker}): {predicted_norm_value:.6f} ★★★")

                         # 5. 매매 로직 및 주문 실행 (예시)
                         if ticker == '005930' and not self.trade_triggered.get(ticker):
                              if predicted_norm_value > 0.7: # ★ 임시 조건 수정 (예: 0.7) ★
                                   print(f"★★★ 매수 조건 충족 ({ticker}): 예측값={predicted_norm_value:.6f} > 0.7 ★★★")
                                   asyncio.create_task(self.send_order("buy", ticker, 1, order_cond="03"))
                                   self.trade_triggered[ticker] = True
                     except Exception as e: print(f"  >> 예측 또는 주문 호출 중 오류: {e}")
                 else: print(f"  >> 오류: 모델 입력 형태 불일치 - {input_seq.shape}")


    async def receive_messages(self):
        """ 서버로부터 메시지 비동기 수신 및 처리 """
        while self.keep_running:
            # ... (연결 끊김 처리 로직) ...
            if not self.connected: print("연결 끊김 감지..."); await asyncio.sleep(5); await self.connect(); continue

            try:
                response_str = await self.websocket.recv()
                response = json.loads(response_str)
                trnm = response.get('trnm')

                if trnm == 'LOGIN': # ... (로그인 처리) ...
                    if response.get('return_code') == 0: print('*** WebSocket 로그인 성공 ***'); self.login_success = True; await self.register_realtime()
                    else: print(f'!!! WebSocket 로그인 실패: {response.get("return_msg")} !!!'); await self.disconnect()
                elif trnm == 'PING': await self.send_message(response)
                elif trnm == 'REAL':
                    if 'data' in response and isinstance(response['data'], list):
                        for real_data in response['data']:
                            code = real_data.get('item'); data_type = real_data.get('type')
                            values = real_data.get('values')
                            if not code or not data_type or not values: continue

                            # --- 1분봉 집계 로직 ---
                            if data_type == '0B': # 주식 체결 데이터만 사용
                                try:
                                    trade_time_str = values.get('908'); trade_price_str = values.get('10'); trade_vol_str = values.get('15')
                                    if not trade_time_str or not trade_price_str or not trade_vol_str: continue
                                    now = datetime.now(); hms_len=len(trade_time_str)
                                    if hms_len != 6: print(f"경고: 시간 형식 오류 {trade_time_str}"); continue # 시간형식 오류 방지
                                    current_dt = now.replace(hour=int(trade_time_str[:2]), minute=int(trade_time_str[2:4]), second=int(trade_time_str[4:6]), microsecond=0)
                                    price = float(trade_price_str.replace('+','').replace('-','')) # 부호 제거
                                    volume = int(trade_vol_str)
                                    if price <= 0 or volume <= 0: continue # 유효하지 않은 값 제외

                                    current_minute_start = current_dt.replace(second=0, microsecond=0)
                                    minute_interval = self.bar_interval_minutes
                                    current_interval_start = current_minute_start - timedelta(minutes=current_minute_start.minute % minute_interval)

                                    if code not in self.current_bars or self.current_bars[code]['time'] < current_interval_start:
                                         if code in self.current_bars and self.current_bars[code].get('open') is not None:
                                             # ★★★ 이전 봉 완료 처리 ★★★
                                             self.process_new_bar(code, self.current_bars[code])
                                         # 새 봉 시작
                                         self.current_bars[code] = {'time': current_interval_start, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume}
                                    else:
                                         # 현재 봉 업데이트
                                         bar = self.current_bars[code]
                                         bar['high'] = max(bar['high'], price); bar['low'] = min(bar['low'], price)
                                         bar['close'] = price; bar['volume'] += volume
                                except ValueError: print(f"실시간 체결 데이터 값 변환 오류: {values}"); continue
                                except Exception as e: print(f"1분봉 집계 중 오류: {e}"); traceback.print_exc()
                            # --- 1분봉 집계 로직 끝 ---
                            # --- 시세('0A') 데이터 처리 로직 (필요시 추가) ---
                            # 예: elif data_type == '0A': current_price = values.get('10')...
                            # --------------------------------------------
                    else: print(f"기타 REAL 데이터: {response}")
                elif trnm == 'REG' or trnm == 'REMOVE': print(f"응답 [{trnm}]: Code={response.get('return_code')}, Msg={response.get('return_msg')}")
                else: print(f'기타 응답 수신: {response}')
            # ... (except 블록들 이전과 동일) ...
            except websockets.ConnectionClosedOK: print('WebSocket 연결 정상 종료됨.'); self.connected = False; break # 루프 종료
            except websockets.ConnectionClosedError as e: print(f'WebSocket 연결 비정상 종료됨: {e}'); self.connected = False; break # 루프 종료
            except json.JSONDecodeError: print(f'수신 데이터 JSON 파싱 오류: {response_str}')
            except Exception as e: print(f'메시지 처리 중 예외 발생: {e}'); traceback.print_exc(); await asyncio.sleep(1)


    # ... (register_realtime, _send_order_sync, send_order, run, disconnect 메소드는 이전과 동일) ...
    async def register_realtime(self):
        if not self.login_success: print("로그인 실패로 실시간 등록 불가"); return
        print("\n실시간 데이터 등록 요청 전송..."); time_now_debug = time.time()
        for reg_item in REGISTER_LIST:
             reg_data_list = [{"item": reg_item["item"], "type": reg_item["type"]}]
             reg_param = {'trnm': 'REG', 'grp_no': '1', 'refresh': '1', 'data': reg_data_list}
             await self.send_message(reg_param); # print(f"  >> 등록 요청 보냄: {reg_param}")
             await asyncio.sleep(0.1)
        print(f"실시간 데이터 등록 요청 완료 (소요시간: {time.time()-time_now_debug:.2f}초)")

    def _send_order_sync(self, tr_id, headers, body):
        try:
            res = requests.post(API_URL_ORDER, headers=headers, data=json.dumps(body))
            res.raise_for_status(); return res.json()
        except requests.exceptions.RequestException as e: print(f"  > 주문 HTTP 요청 오류: {e}"); error_response = {"return_code": -1, "return_msg": f"RequestException: {e}"}; return error_response
        except Exception as e: print(f"  > 주문 처리 예외 (_send_order_sync): {e}"); traceback.print_exc(); return {"return_code": -1, "return_msg": f"Exception: {e}"}

    async def send_order(self, order_type, stock_code, quantity, price=0, order_cond="03"):
        if not self.connected: print("API 미연결 상태. 주문 불가."); return None
        tr_id = ""; body = {}
        if order_type.lower() == 'buy': tr_id = "kt10000"
        elif order_type.lower() == 'sell': tr_id = "kt10001"
        else: print(f"오류: 주문 유형 오류: {order_type}"); return None
        print(f"--- {stock_code} {order_type.upper()} 주문 ({tr_id}) 전송 준비 ---")
        headers = {"Content-Type": "application/json;charset=UTF-8", "Authorization": f"Bearer {self.token}", "api-id": tr_id}
        body = {"dmst_stex_tp": "KRX", "stk_cd": stock_code, "ord_qty": str(quantity), "ord_uv": str(price), "trde_tp": order_cond}
        print(f"  > 주문 요청 Body: {json.dumps(body)}")
        loop = asyncio.get_running_loop()
        try:
            response_data = await loop.run_in_executor(None, functools.partial(self._send_order_sync, tr_id, headers, body))
            print(f"  > 주문 응답 수신: Code={response_data.get('return_code')}, Msg={response_data.get('return_msg')}")
            if response_data.get('return_code') == 0: order_no = response_data.get('output', {}).get('ord_no', response_data.get('ord_no', 'N/A')); print(f"  ★★★ 주문 성공! 주문번호: {order_no} ★★★")
            else: print(f"  !!! 주문 실패 !!!"); pprint.pprint(response_data)
            return response_data
        except Exception as e: print(f"  > 주문 전송 비동기 처리 오류: {e}"); traceback.print_exc(); return {"return_code": -1, "return_msg": str(e)}

    async def run(self):
        await self.connect()
        if self.connected: await self.receive_messages()

    async def disconnect(self):
        self.keep_running = False
        if self.connected and self.websocket: print("WebSocket 연결 종료 시도..."); await self.websocket.close(); self.connected = False; print('WebSocket 연결 종료됨.')
        # ZeroMQ 정리 등 추가 가능
        if hasattr(self, 'model_server_socket') and self.model_server_socket and not self.model_server_socket.closed: self.model_server_socket.close()
        if hasattr(self, 'zmq_context') and self.zmq_context and not self.zmq_context.closed: self.zmq_context.term()
        print("ZeroMQ 연결 종료됨 (disconnect).")


# --- main 함수 및 실행 부분 ---
async def main():
    # ... (설정값 확인 동일) ...
    if any(val in ["YOUR_ACCESS_TOKEN_HERE", ""] for val in [ACCESS_TOKEN]): print("!!! 토큰 필요 !!!"); return
    if any(val in ["YOUR_ACCOUNT_NUMBER_HERE", ""] for val in [ACCOUNT_NO]): print("!!! 계좌번호 필요 !!!"); return
    if any(val in ["YOUR_APP_KEY", ""] for val in [APP_KEY]): print("!!! 앱키 필요 !!!"); return
    if any(val in ["YOUR_APP_SECRET", ""] for val in [APP_SECRET]): print("!!! 시크릿키 필요 !!!"); return

    client = WebSocketClient(SOCKET_URL, ACCESS_TOKEN, ACCOUNT_NO, APP_KEY, APP_SECRET)
    # 시그널 핸들러 설정 (비동기 환경에서는 다르게 처리 필요할 수 있음)
    loop = asyncio.get_running_loop()
    def shutdown_handler():
         print('\n종료 요청 감지! disconnect 호출...')
         # 비동기 disconnect를 직접 호출하기 어려울 수 있으므로 플래그 설정
         client.keep_running = False
         # asyncio.create_task(client.disconnect()) # 새 태스크로 실행 시도 (안전하지 않을 수 있음)

    # loop.add_signal_handler(signal.SIGINT, shutdown_handler) # Windows에서는 미지원
    # loop.add_signal_handler(signal.SIGTERM, shutdown_handler) # Windows에서는 미지원
    # Windows에서는 Ctrl+C 처리가 복잡하므로 일단 KeyboardInterrupt로 처리

    try: await client.run()
    except KeyboardInterrupt: print("\nCtrl+C 감지 (main). 종료 중...")
    finally: await client.disconnect() # run 종료 시 정리

if __name__ == '__main__':
    try: asyncio.run(main())
    except RuntimeError as e:
         if "Cannot run the event loop while another loop is running" in str(e): print("오류: 다른 이벤트 루프가 이미 실행 중입니다.")
         else: print(f"런타임 오류 발생: {e}")
    except Exception as e: print(f"비동기 실행 중 오류 발생: {e}"); traceback.print_exc()
    print("WebSocket 클라이언트 프로그램 종료.")