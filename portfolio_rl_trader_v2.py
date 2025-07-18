# -*- coding: utf-8 -*-
# 파일명: portfolio_rl_trader_v2.py
# 설명: 데이터 파이프라인이 통합된 강화학습 기반 포트폴리오 자동매매 프로그램

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
        self.KIWOOM_APP_KEY = "YOUR_APP_KEY"  # ★ 실제 앱 키로 교체
        self.KIWOOM_APP_SECRET = "YOUR_APP_SECRET" # ★ 실제 앱 시크릿으로 교체
        self.KIWOOM_ACCOUNT_NO_PREFIX = "YOUR_ACCOUNT_PREFIX" # ★ 실제 계좌번호 앞 8자리
        self.KIWOOM_ACCOUNT_NO_SUFFIX = "YOUR_ACCOUNT_SUFFIX" # ★ 실제 계좌번호 뒤 2자리
        self.IS_MOCK_TRADING = True

        # --- 경로 설정 ---
        self.BASE_DIR = "./trading_system_rl"
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "rl_models")
        self.DATA_DIR = os.path.join(self.BASE_DIR, "trading_data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.SCALER_DIR = os.path.join(self.DATA_DIR, "scalers") # 스케일러 저장 경로

        # --- 데이터 및 특징 공학 설정 ---
        self.TARGET_TICKERS = ['005930', '000660', '035720'] # 거래할 종목 티커
        self.DATA_START_DATE = "2022-01-01"
        self.DATA_END_DATE = datetime.now().strftime('%Y-%m-%d')
        self.TIME_STEPS = 30
        # 사용할 기술적 지표 목록
        self.TECHNICAL_INDICATORS = ['SMA_20', 'RSI_14', 'MACD_12_26_9', 'BB_upper', 'BB_lower']

        # --- 강화학습 환경 및 에이전트 설정 ---
        self.ENV_INITIAL_BALANCE = 100000000
        self.ENV_TRANSACTION_COST_PCT = 0.00015
        self.ENV_TAX_PCT = 0.0020
        self.ENV_SLIPPAGE_PCT = 0.0005
        self.AGENT_TYPE = "PPO"
        self.AGENT_POLICY = "MultiInputPolicy" # Dict Observation Space에는 MultiInputPolicy 사용
        self.TOTAL_TRAINING_TIMESTEPS = 200000

        # --- 위험 관리 설정 ---
        self.RISK_MAX_POSITION_RATIO = 0.3
        self.RISK_STOP_LOSS_PCT = -0.10

        # --- 디렉토리 생성 ---
        for path in [self.MODEL_DIR, self.DATA_DIR, self.LOG_DIR, self.SCALER_DIR]:
            os.makedirs(path, exist_ok=True)

# --- 2. 로거 (Logger) 설정 ---
def setup_logger(config):
    logger = logging.getLogger("PortfolioRLTrader")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    log_file = os.path.join(config.LOG_DIR, f"trading_log_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=10, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    return logger

# --- 3. 데이터 관리자 (DataManager) ---
class DataManager:
    """주가 데이터 다운로드 및 관리를 담당하는 클래스"""
    def __init__(self, config, logger):
        self.cfg = config
        self.logger = logger

    def download_stock_data(self, ticker):
        """단일 종목의 OHLCV 데이터를 다운로드"""
        self.logger.info(f"({ticker}) 데이터 다운로드 중... ({self.cfg.DATA_START_DATE} ~ {self.cfg.DATA_END_DATE})")
        try:
            df = fdr.DataReader(ticker, self.cfg.DATA_START_DATE, self.cfg.DATA_END_DATE)
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            if df.empty:
                self.logger.warning(f"({ticker}) 다운로드된 데이터가 없습니다.")
                return None
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            self.logger.error(f"({ticker}) 데이터 다운로드 실패: {e}", exc_info=True)
            return None

# --- 4. 특징 공학자 (FeatureEngineer) ---
class FeatureEngineer:
    """데이터에 기술적 보조지표 등 특징을 추가하는 클래스"""
    def __init__(self, config, logger):
        self.cfg = config
        self.logger = logger

    def process_features(self, df):
        """데이터프레임에 기술적 지표 추가 및 정규화"""
        if df is None or df.empty:
            return None

        # 기술적 지표 추가
        df.ta.sma(length=20, append=True, col_names=('SMA_20',))
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
        df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
        df.ta.bbands(length=20, std=2, append=True, col_names=('BB_lower', 'BB_middle', 'BB_upper', 'BB_bandwidth', 'BB_percent'))
        
        # 사용하지 않는 bbands 컬럼 제거
        df.drop(columns=['BB_middle', 'BB_bandwidth', 'BB_percent'], inplace=True, errors='ignore')
        
        df.fillna(0, inplace=True) # NaN 값을 0으로 채움
        
        # 정규화
        scaler = MinMaxScaler()
        # 원본 OHLCV도 학습에 사용될 수 있으므로 정규화 대상에 포함
        cols_to_scale = ['open', 'high', 'low', 'close', 'volume'] + [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        df_scaled = pd.DataFrame(scaler.fit_transform(df[cols_to_scale]), columns=df[cols_to_scale].columns, index=df.index)

        return df_scaled

# --- 5. 강화학습 환경 (StockPortfolioEnv) ---
# (이전 답변의 StockPortfolioEnv 클래스 코드와 동일)
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
        
        self.num_features = self.data[self.tickers[0]].shape[1]
        features_shape = (len(self.tickers), self.time_steps, self.num_features)
        
        # 관찰 공간 정의: Dict space 사용
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=features_shape, dtype=np.float32),
            'portfolio': spaces.Box(low=0, high=1, shape=(len(self.tickers) + 1,), dtype=np.float32)
        })

        # 행동 공간 정의 (각 종목에 대해 0:유지, 1:매수, 2:매도)
        self.action_space = spaces.MultiDiscrete([3] * len(self.tickers))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.time_steps
        self.balance = self.cfg.ENV_INITIAL_BALANCE
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_portfolio_value = self.balance
        self.done = False
        return self._next_observation(), self._get_info()

    def _next_observation(self):
        # 특징 데이터 생성
        features = np.array([self.data[ticker].iloc[self.current_step-self.time_steps:self.current_step].values for ticker in self.tickers])
        
        # 포트폴리오 상태 생성
        portfolio_values = np.array([self.shares_held[ticker] * self.data[ticker]['close'].iloc[self.current_step] for ticker in self.tickers])
        self.total_portfolio_value = self.balance + portfolio_values.sum()
        
        portfolio_state = np.zeros(len(self.tickers) + 1)
        if self.total_portfolio_value > 0:
            portfolio_state[0] = self.balance / self.total_portfolio_value
            portfolio_state[1:] = portfolio_values / self.total_portfolio_value

        return {'features': features.astype(np.float32), 'portfolio': portfolio_state.astype(np.float32)}

    def step(self, action):
        prev_portfolio_value = self.total_portfolio_value
        
        # 행동 실행
        for i, ticker_action in enumerate(action):
            ticker = self.tickers[i]
            current_price = self.data[ticker]['close'].iloc[self.current_step]
            
            if ticker_action == 1: # 매수
                trade_amount = self.balance * self.cfg.RISK_MAX_POSITION_RATIO
                if trade_amount > 10000: # 최소 거래 금액
                    quantity = trade_amount / (current_price * (1 + self.cfg.ENV_SLIPPAGE_PCT))
                    cost = quantity * current_price * (1 + self.cfg.ENV_TRANSACTION_COST_PCT)
                    if self.balance >= cost:
                        self.balance -= cost
                        self.shares_held[ticker] += quantity
            
            elif ticker_action == 2: # 매도
                quantity_to_sell = self.shares_held[ticker]
                if quantity_to_sell > 0:
                    proceeds = quantity_to_sell * current_price * (1 - self.cfg.ENV_TRANSACTION_COST_PCT - self.cfg.ENV_TAX_PCT - self.cfg.ENV_SLIPPAGE_PCT)
                    self.balance += proceeds
                    self.shares_held[ticker] = 0

        self.current_step += 1
        reward = (self.total_portfolio_value - prev_portfolio_value) / prev_portfolio_value

        if self.current_step >= len(self.data[self.tickers[0]]) - 1:
            self.done = True
        
        obs = self._next_observation()
        info = self._get_info()
        
        return obs, reward, self.done, self.done, info

    def _get_info(self):
        return {'portfolio_value': self.total_portfolio_value, 'balance': self.balance, 'shares_held': self.shares_held}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.total_portfolio_value:,.0f}")

# --- 6. 메인 실행 로직 ---
def main():
    cfg = Config()
    logger = setup_logger(cfg)

    # --- 데이터 파이프라인 실행 ---
    logger.info("--- 데이터 파이프라인 시작 ---")
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
            else:
                logger.warning(f"({ticker}) 특징 공학 처리 실패.")
        else:
            logger.warning(f"({ticker}) 데이터 다운로드 실패.")
    
    if not valid_tickers:
        logger.error("유효한 데이터를 가진 종목이 없습니다. 프로그램을 종료합니다.")
        return

    logger.info(f"--- 데이터 파이프라인 완료. 최종 학습/거래 대상 종목: {valid_tickers} ---")
    
    # 공통 날짜로 데이터 정렬
    common_index = None
    for ticker in valid_tickers:
        if common_index is None:
            common_index = processed_data[ticker].index
        else:
            common_index = common_index.intersection(processed_data[ticker].index)
    
    for ticker in valid_tickers:
        processed_data[ticker] = processed_data[ticker].reindex(common_index).fillna(0)


    # --- 강화학습 모델 학습 또는 로드 ---
    env = DummyVecEnv([lambda: Monitor(StockPortfolioEnv(processed_data, valid_tickers, cfg, logger))])
    model_path = os.path.join(cfg.MODEL_DIR, "ppo_portfolio_model.zip")

    if os.path.exists(model_path):
        logger.info(f"학습된 모델 로드: {model_path}")
        model = PPO.load(model_path, env)
    else:
        logger.info("새로운 모델 학습 시작...")
        model = PPO(cfg.AGENT_POLICY, env, verbose=1, tensorboard_log=cfg.LOG_DIR)
        model.learn(total_timesteps=cfg.TOTAL_TRAINING_TIMESTEPS)
        model.save(model_path)
        logger.info(f"모델 학습 완료 및 저장: {model_path}")
        
    # --- 실시간 거래 루프 ---
    logger.info("--- 실시간 거래 시작 ---")
    api = KiwoomRESTClient(cfg, logger)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        # obs, rewards, dones, info = env.step(action) # 실시간 거래에서는 step을 호출하지 않음
        
        # 예측된 행동에 따라 주문 실행
        for i, ticker_action in enumerate(action[0]):
            ticker = valid_tickers[i]
            if ticker_action == 1: # 매수
                api.send_order(ticker, 'buy', 1) 
            elif ticker_action == 2: # 매도
                # 보유 수량 확인 후 매도 로직 추가 필요
                api.send_order(ticker, 'sell', 1)

        time.sleep(60) # 1분마다 반복

if __name__ == "__main__":
    main()