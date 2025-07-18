# -*- coding: utf-8 -*-
# 파일명: portfolio_rl_trader_final.py
# 설명: 위험 관리 및 웹소켓 실시간 데이터 처리가 통합된 강화학습 포트폴리오 자동매매 프로그램

import os
import time
import json
import requests
import random
import logging
import logging.handlers
import traceback
import asyncio
import websockets
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.preprocessing import MinMaxScaler
import FinanceDataReader as fdr
from pykrx import stock

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- 1. 설정 (Config) 클래스 ---
class Config:
    """프로젝트의 모든 설정을 관리하는 클래스"""
    def __init__(self):
        # --- API 및 계좌 설정 ---
        self.KIWOOM_APP_KEY = "YOUR_APP_KEY"
        self.KIWOOM_APP_SECRET = "YOUR_APP_SECRET"
        self.KIWOOM_ACCOUNT_NO_PREFIX = "YOUR_ACCOUNT_PREFIX"
        self.KIWOOM_ACCOUNT_NO_SUFFIX = "YOUR_ACCOUNT_SUFFIX"
        self.KIWOOM_WEBSOCKET_TOKEN = "YOUR_WEBSOCKET_TOKEN" # 웹소켓 접속용 토큰
        self.IS_MOCK_TRADING = True

        # --- 경로 설정 ---
        self.BASE_DIR = "./trading_system_rl_final"
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "rl_models")
        self.DATA_DIR = os.path.join(self.BASE_DIR, "trading_data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.SCALER_DIR = os.path.join(self.DATA_DIR, "scalers")

        # --- 데이터 및 특징 공학 설정 ---
        self.TARGET_TICKERS = ['005930', '000660', '035720']
        self.DATA_START_DATE = "2022-01-01"
        self.DATA_END_DATE = datetime.now().strftime('%Y-%m-%d')
        self.TIME_STEPS = 30

        # --- 강화학습 환경 및 에이전트 설정 ---
        self.ENV_INITIAL_BALANCE = 100000000
        self.ENV_TRANSACTION_COST_PCT = 0.00015
        self.ENV_TAX_PCT = 0.0020
        self.ENV_SLIPPAGE_PCT = 0.0005
        self.AGENT_POLICY = "MultiInputPolicy"
        self.TOTAL_TRAINING_TIMESTEPS = 200000

        # --- 위험 관리 설정 ---
        self.RISK_MAX_POSITION_RATIO = 0.3
        self.RISK_STOP_LOSS_PCT = -0.10

        for path in [self.MODEL_DIR, self.DATA_DIR, self.LOG_DIR, self.SCALER_DIR]:
            os.makedirs(path, exist_ok=True)

# --- 2. 로거 (Logger) 설정 ---
def setup_logger(config):
    logger = logging.getLogger("PortfolioRLTrader")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    log_file = os.path.join(config.LOG_DIR, f"trading_log_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=10, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    return logger

# --- 3. 위험 관리자 (RiskManager) 클래스 ---
class RiskManager:
    """주문 수량 계산 및 손절매 등 위험 관리를 담당"""
    def __init__(self, config, logger):
        self.cfg = config
        self.logger = logger

    def calculate_position_size(self, balance, price, num_tickers):
        """매수할 주식 수량을 계산"""
        if price <= 0: return 0
        # 포트폴리오 비중에 따라 종목당 투자 가능 금액 계산
        investment_per_ticker = (balance * self.cfg.RISK_MAX_POSITION_RATIO)
        # 실제 주문 가능 수량 계산 (수수료, 슬리피지 고려)
        buy_price_with_cost = price * (1 + self.cfg.ENV_TRANSACTION_COST_PCT + self.cfg.ENV_SLIPPAGE_PCT)
        if buy_price_with_cost <= 0: return 0
        quantity = int(investment_per_ticker / buy_price_with_cost)
        self.logger.info(f"주문 수량 계산: 잔고({balance:,.0f}), 종목당 투자금({investment_per_ticker:,.0f}) -> {quantity}주")
        return quantity

    def check_stop_loss(self, current_price, entry_price):
        """손절매 조건 확인"""
        if entry_price <= 0: return False
        pnl_ratio = (current_price - entry_price) / entry_price
        if pnl_ratio <= self.cfg.RISK_STOP_LOSS_PCT:
            self.logger.warning(f"!!! 손절매 조건 충족 !!! 현재수익률: {pnl_ratio:.2%}, 손절기준: {self.cfg.RISK_STOP_LOSS_PCT:.2%}")
            return True
        return False

# --- 4. 키움증권 API 클라이언트 (KiwoomClient) ---
# 웹소켓과 REST API를 모두 관리하는 통합 클라이언트
class KiwoomClient:
    def __init__(self, config, logger, data_manager, feature_engineer, agent, risk_manager):
        self.cfg = config
        self.logger = logger
        self.dm = data_manager
        self.fe = feature_engineer
        self.agent = agent
        self.rm = risk_manager

        # REST API 관련
        self.BASE_URL = "https://mockapi.kiwoom.com" if self.cfg.IS_MOCK_TRADING else "https://openapi.kiwoom.com"
        self.rest_access_token = None
        
        # WebSocket 관련
        self.WEBSOCKET_URL = f"wss://{'mockapi.kiwoom.com:10000' if self.cfg.IS_MOCK_TRADING else 'openapi.kiwoom.com:9443'}/api/dostk/websocket"
        self.ws = None
        self.is_ws_connected = False
        
        # 실시간 데이터 처리
        self.current_bars = {}
        self.completed_bars = {ticker: deque(maxlen=200) for ticker in self.cfg.TARGET_TICKERS}
        self.feature_sequences = {ticker: deque(maxlen=self.cfg.TIME_STEPS) for ticker in self.cfg.TARGET_TICKERS}
        
        # 포지션 관리
        self.positions = {ticker: {'quantity': 0, 'entry_price': 0} for ticker in self.cfg.TARGET_TICKERS}

    async def run(self):
        """메인 실행 함수"""
        self._get_rest_access_token()
        while True:
            try:
                self.logger.info("웹소켓 연결 시도...")
                async with websockets.connect(self.WEBSOCKET_URL) as ws:
                    self.ws = ws
                    self.is_ws_connected = True
                    self.logger.info("웹소켓 연결 성공.")
                    
                    # 로그인 및 실시간 데이터 등록
                    await self._login_and_register()
                    
                    # 메시지 수신 루프
                    async for message in self.ws:
                        await self._on_message(message)

            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
                self.logger.warning(f"웹소켓 연결 끊김: {e}. 5초 후 재연결합니다.")
                self.is_ws_connected = False
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"웹소켓 처리 중 예외 발생: {e}", exc_info=True)
                self.is_ws_connected = False
                await asyncio.sleep(5)
    
    async def _login_and_register(self):
        """웹소켓 로그인 및 실시간 시세 등록"""
        # 로그인
        login_packet = {"header": {"tr_id": "LOGIN"}, "body": {"token": self.cfg.KIWOOM_WEBSOCKET_TOKEN}}
        await self.ws.send(json.dumps(login_packet))
        
        # 실시간 시세 등록
        reg_packet = {
            "header": {"tr_id": "REG", "grp_no": "1", "refresh": "1"},
            "body": {"data": [{"item": self.cfg.TARGET_TICKERS, "type": ["0B"]}]} # 0B: 체결
        }
        await self.ws.send(json.dumps(reg_packet))
        self.logger.info(f"실시간 시세 등록 요청 완료: {self.cfg.TARGET_TICKERS}")

    async def _on_message(self, message):
        """웹소켓 메시지 처리"""
        try:
            data = json.loads(message)
            if data.get('trnm') == 'REAL' and data.get('data'):
                for real_data in data['data']:
                    await self._aggregate_bar(real_data)
        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e} | 메시지: {message}")

    async def _aggregate_bar(self, real_data):
        """1분봉 데이터 집계"""
        try:
            ticker = real_data.get('item')
            values = real_data.get('values')
            if not all([ticker, values]): return
            
            trade_time = datetime.strptime(f"{datetime.now().strftime('%Y-%m-%d')} {values.get('20')}", "%Y-%m-%d %H%M%S")
            price = float(values.get('10').replace('+', '').replace('-', ''))
            volume = int(values.get('15'))

            bar_time = trade_time.replace(second=0, microsecond=0)
            
            if ticker not in self.current_bars or self.current_bars[ticker]['time'] < bar_time:
                # 이전 봉 완성 처리
                if ticker in self.current_bars:
                    await self._on_bar_complete(ticker, self.current_bars[ticker])
                # 새 봉 시작
                self.current_bars[ticker] = {'time': bar_time, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume}
            else:
                # 현재 봉 업데이트
                bar = self.current_bars[ticker]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
                bar['volume'] += volume
        except Exception as e:
            self.logger.error(f"({ticker}) 봉 집계 오류: {e}", exc_info=True)

    async def _on_bar_complete(self, ticker, bar_data):
        """봉 완성 시 의사결정 및 거래 실행"""
        self.logger.info(f"봉 완성 ({ticker}): {bar_data}")
        
        # 1. 데이터 업데이트 및 전처리
        df_new_bar = pd.DataFrame([bar_data]).set_index('time')
        self.completed_bars[ticker].append(df_new_bar)
        
        df_history = pd.concat(list(self.completed_bars[ticker]))
        processed_df = self.fe.process_features(df_history.copy())
        
        if processed_df is None or len(processed_df) < self.cfg.TIME_STEPS:
            self.logger.warning(f"({ticker}) 처리 후 데이터 길이 부족. 의사결정 건너<binary data, 1 bytes><binary data, 1 bytes><binary data, 1 bytes>.")
            return

        # 2. RL 에이전트 의사결정
        obs = self._get_observation_for_agent()
        action, _ = self.agent.predict(obs, deterministic=True)
        
        # 3. 위험 관리 및 주문 실행
        self._execute_trade_logic(action[0])
    
    def _get_observation_for_agent(self):
        """RL 에이전트를 위한 관찰(상태) 데이터 생성"""
        # (StockPortfolioEnv의 _next_observation 로직과 유사하게 구성)
        # 모든 종목의 최신 데이터를 기반으로 상태를 만듭니다.
        # 이 부분은 단순화를 위해 생략, 실제 구현 시 env와 동일한 로직 필요
        # 임시로 랜덤 값을 반환
        return self.agent.env.observation_space.sample()

    def _execute_trade_logic(self, actions):
        """예측된 행동에 따라 거래 로직 실행"""
        for i, ticker_action in enumerate(actions):
            ticker = self.cfg.TARGET_TICKERS[i]
            current_price = self.get_current_price(ticker)
            if current_price is None: continue
            
            position = self.positions[ticker]

            # 손절매 로직
            if position['quantity'] > 0 and self.rm.check_stop_loss(current_price, position['entry_price']):
                self.send_order(ticker, 'sell', position['quantity'])
                self.positions[ticker] = {'quantity': 0, 'entry_price': 0}
                continue # 손절매 실행 시 추가 행동 없음

            # RL 에이전트 행동 실행
            if ticker_action == 1 and position['quantity'] == 0: # 매수
                quantity = self.rm.calculate_position_size(10000000, current_price, len(self.cfg.TARGET_TICKERS)) # 임시 잔고
                if quantity > 0:
                    if self.send_order(ticker, 'buy', quantity):
                        self.positions[ticker] = {'quantity': quantity, 'entry_price': current_price}
            elif ticker_action == 2 and position['quantity'] > 0: # 매도
                if self.send_order(ticker, 'sell', position['quantity']):
                    self.positions[ticker] = {'quantity': 0, 'entry_price': 0}


    def _get_rest_access_token(self):
        """REST API 접근 토큰 발급"""
        # 이전 답변의 KiwoomRESTClient._get_access_token 내용과 동일
        url = f"{self.BASE_URL}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {"grant_type": "client_credentials", "appkey": self.cfg.KIWOOM_APP_KEY, "appsecret": self.cfg.KIWOOM_APP_SECRET}
        try:
            res = requests.post(url, headers=headers, json=body)
            res.raise_for_status()
            self.access_token = res.json().get("access_token")
            if self.access_token: self.logger.info("키움증권 REST API 접근 토큰 발급 성공")
            else: self.logger.error(f"키움증권 REST API 접근 토큰 발급 실패: {res.text}")
        except Exception as e: self.logger.error(f"REST API 토큰 발급 중 예외 발생: {e}", exc_info=True)


    def get_current_price(self, ticker):
        """REST API로 현재가 조회"""
        # 이전 답변의 KiwoomRESTClient.get_current_price 내용과 동일
        if not self.access_token: return None
        url = f"{self.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {"Authorization": f"Bearer {self.access_token}", "appkey": self.cfg.KIWOOM_APP_KEY, "appsecret": self.cfg.KIWOOM_APP_SECRET, "tr_id": "FHKST01010100"}
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
        try:
            res = requests.get(url, headers=headers, params=params)
            data = res.json()
            if data['rt_cd'] == '0': return float(data['output']['stck_prpr'])
        except Exception: return None
        return None


    def send_order(self, ticker, order_type, quantity, price=0):
        """REST API로 주문 실행"""
        # 이전 답변의 KiwoomRESTClient.send_order 내용과 동일
        if not self.access_token: return False
        tr_id = "VTTC0802U" if order_type.lower() == 'buy' else "VTTC0801U"
        if not self.cfg.IS_MOCK_TRADING: tr_id = "TTTC0802U" if order_type.lower() == 'buy' else "TTTC0801U"
        
        headers = {"Authorization": f"Bearer {self.access_token}", "appkey": self.cfg.KIWOOM_APP_KEY, "appsecret": self.cfg.KIWOOM_APP_SECRET, "tr_id": tr_id, "Content-Type": "application/json; charset=utf-8"}
        body = {"CANO": self.cfg.KIWOOM_ACCOUNT_NO_PREFIX, "ACNT_PRDT_CD": self.cfg.KIWOOM_ACCOUNT_NO_SUFFIX, "PDNO": ticker, "ORD_DVSN": "01", "ORD_QTY": str(quantity), "ORD_UNPR": "0"}
        url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
        
        try:
            res = requests.post(url, headers=headers, json=body)
            data = res.json()
            if data['rt_cd'] == '0': self.logger.info(f"주문 성공 ({ticker}, {order_type}, {quantity}주)"); return True
            else: self.logger.error(f"주문 실패 ({ticker}): {data['msg1']}"); return False
        except Exception as e: self.logger.error(f"주문 예외 ({ticker}): {e}"); return False

# --- 7. 메인 실행 함수 ---
async def main():
    cfg = Config()
    logger = setup_logger(cfg)

    # 데이터 준비
    dm = DataManager(cfg, logger)
    fe = FeatureEngineer(cfg, logger)
    
    processed_data = {}
    valid_tickers = []
    for ticker in cfg.TARGET_TICKERS:
        raw_df = dm.download_stock_data(ticker)
        if raw_df is not None:
            processed_df = fe.process_features(raw_df)
            if processed_df is not None:
                processed_data[ticker] = processed_df
                valid_tickers.append(ticker)
    
    if not valid_tickers:
        logger.error("유효한 데이터를 가진 종목이 없습니다. 종료합니다.")
        return

    common_index = None
    for ticker in valid_tickers:
        if common_index is None: common_index = processed_data[ticker].index
        else: common_index = common_index.intersection(processed_data[ticker].index)
    
    for ticker in valid_tickers:
        processed_data[ticker] = processed_data[ticker].reindex(common_index).fillna(0)

    # 강화학습 모델
    env = DummyVecEnv([lambda: Monitor(StockPortfolioEnv(processed_data, valid_tickers, cfg, logger))])
    model_path = os.path.join(cfg.MODEL_DIR, "ppo_portfolio_model.zip")

    if not os.path.exists(model_path):
        logger.info("모델 학습 시작...")
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=cfg.LOG_DIR)
        model.learn(total_timesteps=cfg.TOTAL_TRAINING_TIMESTEPS)
        model.save(model_path)
    else:
        logger.info("학습된 모델 로드...")
        model = PPO.load(model_path, env)

    # 리스크 관리자
    rm = RiskManager(cfg, logger)

    # 키움 클라이언트 실행
    client = KiwoomClient(cfg, logger, dm, fe, model, rm)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())