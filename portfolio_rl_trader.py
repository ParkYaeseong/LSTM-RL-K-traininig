# -*- coding: utf-8 -*-
# 파일명: portfolio_rl_trader.py
# 설명: 강화학습(PPO)과 키움증권 REST API를 이용한 포트폴리오 자동매매 프로그램

import os
import time
import json
import requests
import random
import logging
import logging.handlers
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.preprocessing import MinMaxScaler

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 1. 설정 (Config) 클래스 ---
class Config:
    """프로젝트의 모든 설정을 관리하는 클래스"""
    def __init__(self):
        # --- API 및 계좌 설정 ---
        self.KIWOOM_APP_KEY = "YOUR_APP_KEY"  # ★ 실제 앱 키로 교체
        self.KIWOOM_APP_SECRET = "YOUR_APP_SECRET" # ★ 실제 앱 시크릿으로 교체
        self.KIWOOM_ACCOUNT_NO_PREFIX = "YOUR_ACCOUNT_PREFIX" # ★ 실제 계좌번호 앞 8자리
        self.KIWOOM_ACCOUNT_NO_SUFFIX = "YOUR_ACCOUNT_SUFFIX" # ★ 실제 계좌번호 뒤 2자리
        self.IS_MOCK_TRADING = True # True: 모의투자, False: 실전투자

        # --- 경로 설정 ---
        self.BASE_DIR = "./trading_system_rl"
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "rl_models")
        self.DATA_DIR = os.path.join(self.BASE_DIR, "trading_data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")

        # --- 데이터 및 특징 공학 설정 ---
        self.TARGET_TICKERS = ['005930', '000660', '035720'] # ★ 거래할 종목 티커
        self.DATA_START_DATE = "2022-01-01"
        self.DATA_END_DATE = datetime.now().strftime('%Y-%m-%d')
        self.TIME_STEPS = 30 # 에이전트가 관찰할 과거 데이터 기간

        # --- 강화학습 환경 설정 ---
        self.ENV_INITIAL_BALANCE = 100000000 # 초기 자본금 (1억)
        self.ENV_TRANSACTION_COST_PCT = 0.00015 # 수수료 (0.015%)
        self.ENV_TAX_PCT = 0.0020 # 매도세 (0.2%)
        self.ENV_SLIPPAGE_PCT = 0.0005 # 슬리피지 (0.05%)

        # --- 강화학습 에이전트 설정 ---
        self.AGENT_TYPE = "PPO"
        self.AGENT_POLICY = "MlpPolicy"
        self.TOTAL_TRAINING_TIMESTEPS = 200000 # 총 학습 타임스텝

        # --- 위험 관리 설정 ---
        self.RISK_MAX_POSITION_RATIO = 0.3 # 종목당 최대 투자 비율 (30%)
        self.RISK_STOP_LOSS_PCT = -0.10 # 손절매 비율 (-10%)

        # --- 디렉토리 생성 ---
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

# --- 2. 로거 (Logger) 설정 ---
def setup_logger(config):
    """로깅 설정 함수"""
    logger = logging.getLogger("PortfolioRLTrader")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    # 파일 핸들러
    log_file = os.path.join(config.LOG_DIR, f"trading_log_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=10, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    return logger

# --- 3. 키움증권 API 클라이언트 (KiwoomRESTClient) 클래스 ---
class KiwoomRESTClient:
    """키움증권 REST API 통신을 담당하는 클래스"""
    def __init__(self, config, logger):
        self.cfg = config
        self.logger = logger
        self.BASE_URL = "https://mockapi.kiwoom.com" if self.cfg.IS_MOCK_TRADING else "https://openapi.kiwoom.com"
        self.access_token = None
        self._get_access_token()

    def _get_access_token(self):
        """접근 토큰 발급"""
        url = f"{self.BASE_URL}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {"grant_type": "client_credentials", "appkey": self.cfg.KIWOOM_APP_KEY, "appsecret": self.cfg.KIWOOM_APP_SECRET}
        try:
            res = requests.post(url, headers=headers, json=body)
            res.raise_for_status()
            self.access_token = res.json().get("access_token")
            if self.access_token:
                self.logger.info("키움증권 접근 토큰 발급 성공")
            else:
                self.logger.error(f"키움증권 접근 토큰 발급 실패: {res.text}")
        except Exception as e:
            self.logger.error(f"토큰 발급 중 예외 발생: {e}", exc_info=True)

    def get_current_price(self, ticker):
        """실시간 현재가 조회"""
        url = f"{self.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.access_token}",
            "appkey": self.cfg.KIWOOM_APP_KEY,
            "appsecret": self.cfg.KIWOOM_APP_SECRET,
            "tr_id": "FHKST01010100" # 주식 현재가 시세
        }
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
        try:
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()
            data = res.json()
            if data['rt_cd'] == '0':
                return float(data['output']['stck_prpr'])
            else:
                self.logger.warning(f"({ticker}) 현재가 조회 실패: {data['msg1']}")
                return None
        except Exception as e:
            self.logger.error(f"({ticker}) 현재가 조회 중 예외 발생: {e}", exc_info=True)
            return None

    def send_order(self, ticker, order_type, quantity, price=0):
        """주식 주문 실행 (시장가)"""
        url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "VTTC0802U" if order_type.lower() == 'buy' else "VTTC0801U" # 모의투자 기준
        if not self.cfg.IS_MOCK_TRADING:
             tr_id = "TTTC0802U" if order_type.lower() == 'buy' else "TTTC0801U"
        
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.access_token}",
            "appkey": self.cfg.KIWOOM_APP_KEY,
            "appsecret": self.cfg.KIWOOM_APP_SECRET,
            "tr_id": tr_id
        }
        body = {
            "CANO": self.cfg.KIWOOM_ACCOUNT_NO_PREFIX,
            "ACNT_PRDT_CD": self.cfg.KIWOOM_ACCOUNT_NO_SUFFIX,
            "PDNO": ticker,
            "ORD_DVSN": "01",  # 01: 시장가
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0"
        }
        try:
            res = requests.post(url, headers=headers, json=body)
            res.raise_for_status()
            data = res.json()
            if data['rt_cd'] == '0':
                self.logger.info(f"주문 성공 ({ticker}, {order_type}, {quantity}주): {data['msg1']}")
                return True
            else:
                self.logger.error(f"주문 실패 ({ticker}, {order_type}, {quantity}주): {data['msg1']}")
                return False
        except Exception as e:
            self.logger.error(f"주문 중 예외 발생 ({ticker}, {order_type}): {e}", exc_info=True)
            return False

# --- 4. 데이터 관리자 (DataManager) 및 특징 공학자 (FeatureEngineer) ---
# colab_stock.ipynb의 DataManager와 FeatureEngineer 클래스를 여기에 붙여넣습니다.
# (생략 - 위 파일 내용 참조)

# --- 5. 강화학습 환경 (StockPortfolioEnv) 클래스 ---
# colab_stock.ipynb의 StockTradingEnv를 포트폴리오 환경으로 수정한 버전
class StockPortfolioEnv(gym.Env):
    """다중 주식 포트폴리오 거래를 위한 강화학습 환경"""
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dict, tickers, config, logger):
        super(StockPortfolioEnv, self).__init__()
        self.data = data_dict
        self.tickers = tickers
        self.cfg = config
        self.logger = logger
        self.time_steps = self.cfg.TIME_STEPS
        
        # 상태 공간 정의 (특징 + 포트폴리오)
        # 각 종목의 특징 데이터 (TIME_STEPS x num_features) + 포트폴리오 상태 (종목 수 + 현금)
        self.num_features = self.data[self.tickers[0]].shape[1]
        self.shape = (len(self.tickers), self.time_steps, self.num_features)
        features_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        portfolio_space = spaces.Box(low=0, high=1, shape=(len(self.tickers) + 1,), dtype=np.float32)
        self.observation_space = spaces.Dict({'features': features_space, 'portfolio': portfolio_space})

        # 행동 공간 정의 (각 종목에 대해 -1: 매도, 0: 유지, 1: 매수)
        self.action_space = spaces.MultiDiscrete([3] * len(self.tickers))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.time_steps
        self.balance = self.cfg.ENV_INITIAL_BALANCE
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.balance
        self.done = False
        return self._next_observation(), self._get_info()

    def _next_observation(self):
        features = np.array([self.data[ticker].iloc[self.current_step-self.time_steps:self.current_step].values for ticker in self.tickers])
        
        portfolio = np.zeros(len(self.tickers) + 1)
        portfolio[0] = self.balance / self.portfolio_value
        for i, ticker in enumerate(self.tickers):
            price = self.data[ticker]['close'].iloc[self.current_step]
            portfolio[i+1] = (self.shares_held[ticker] * price) / self.portfolio_value
        
        return {'features': features.astype(np.float32), 'portfolio': portfolio.astype(np.float32)}

    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        
        # 행동 실행 (매수/매도/유지)
        for i, ticker_action in enumerate(action):
            ticker = self.tickers[i]
            current_price = self.data[ticker]['close'].iloc[self.current_step]
            
            if ticker_action == 2: # 매수
                # RiskManager 로직으로 구매 수량 결정 (여기서는 단순화)
                available_investment = self.balance * self.cfg.RISK_MAX_POSITION_RATIO / len(self.tickers)
                quantity_to_buy = int(available_investment / (current_price * (1 + self.cfg.ENV_SLIPPAGE_PCT)))
                cost = quantity_to_buy * current_price * (1 + self.cfg.ENV_TRANSACTION_COST_PCT + self.cfg.ENV_SLIPPAGE_PCT)
                if self.balance >= cost and quantity_to_buy > 0:
                    self.balance -= cost
                    self.shares_held[ticker] += quantity_to_buy
            
            elif ticker_action == 0: # 매도
                quantity_to_sell = self.shares_held[ticker]
                if quantity_to_sell > 0:
                    proceeds = quantity_to_sell * current_price * (1 - self.cfg.ENV_TRANSACTION_COST_PCT - self.cfg.ENV_SLIPPAGE_PCT - self.cfg.ENV_TAX_PCT)
                    self.balance += proceeds
                    self.shares_held[ticker] = 0

        # 다음 스텝으로 이동 및 포트폴리오 가치 업데이트
        self.current_step += 1
        current_portfolio_value = self.balance
        for ticker in self.tickers:
            current_portfolio_value += self.shares_held[ticker] * self.data[ticker]['close'].iloc[self.current_step]
        self.portfolio_value = current_portfolio_value

        # 보상 계산
        reward = self.portfolio_value - prev_portfolio_value
        
        # 종료 조건
        if self.current_step >= len(self.data[self.tickers[0]]) - 1:
            self.done = True
        
        obs = self._next_observation()
        info = self._get_info()
        
        return obs, reward, self.done, self.done, info

    def _get_info(self):
        return {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held
        }

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:,.0f}, Balance: {self.balance:,.0f}")
        print(f"Shares: {self.shares_held}")

# --- 6. 메인 실행 함수 ---
def main():
    cfg = Config()
    logger = setup_logger(cfg)

    # 데이터 준비
    # DataManager와 FeatureEngineer를 사용하여 데이터프레임 딕셔너리 생성
    # (생략 - 위 파일 내용 참조)
    
    # 임시 데이터 생성 (실제로는 DataManager 사용)
    data = {}
    for ticker in cfg.TARGET_TICKERS:
        dates = pd.date_range(start=cfg.DATA_START_DATE, end=cfg.DATA_END_DATE, freq='B')
        price = 100000 + np.random.randn(len(dates)).cumsum() * 500
        df = pd.DataFrame({'open': price, 'high': price, 'low': price, 'close': price, 'volume': 1000}, index=dates)
        # FeatureEngineer로 기술적 지표 추가
        df.ta.strategy(ta.Strategy(name="TA_ALL", ta=[{"kind": "all"}]))
        df.fillna(0, inplace=True)
        # 정규화
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        data[ticker] = df

    # 강화학습 환경 및 에이전트 생성
    env = DummyVecEnv([lambda: StockPortfolioEnv(data, cfg.TARGET_TICKERS, cfg, logger)])
    model_path = os.path.join(cfg.MODEL_DIR, f"ppo_portfolio_model.zip")

    if os.path.exists(model_path):
        logger.info(f"학습된 모델 로드: {model_path}")
        model = PPO.load(model_path, env)
    else:
        logger.info("새로운 모델 학습 시작...")
        model = PPO(cfg.AGENT_POLICY, env, verbose=1)
        model.learn(total_timesteps=cfg.TOTAL_TRAINING_TIMESTEPS)
        model.save(model_path)
        logger.info(f"모델 학습 완료 및 저장: {model_path}")

    # --- 실시간 거래 루프 ---
    logger.info("--- 실시간 거래 시작 ---")
    api = KiwoomRESTClient(cfg, logger)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # 실제 주문 실행
        for i, ticker_action in enumerate(action[0]):
            ticker = cfg.TARGET_TICKERS[i]
            if ticker_action == 2: # 매수
                api.send_order(ticker, 'buy', 1) # 수량은 RiskManager로 결정 필요
            elif ticker_action == 0: # 매도
                api.send_order(ticker, 'sell', 1) # 수량은 보유량만큼

        env.render()
        time.sleep(60) # 1분마다 실행

if __name__ == "__main__":
    main()