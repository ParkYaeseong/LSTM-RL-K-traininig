#  LSTM-RL-K-traininig (LSTM & 강화학습 기반 주식 자동매매)

## 프로젝트 소개

이 프로젝트는 **LSTM(Long Short-Term Memory) 딥러닝 모델**과 \*\*강화학습(Reinforcement Learning)\*\*을 이용하여 주식 매매 전략을 수립하고, **키움증권 API**와 연동하여 자동으로 주문을 실행하는 시스템입니다.

두 가지 주요 접근 방식을 포함하고 있습니다:

1.  **LSTM 예측 기반 매매**: 실시간으로 수집한 데이터와 기술적 보조지표를 바탕으로 LSTM 모델이 주가를 예측하면, 그 결과에 따라 매매를 실행합니다.
2.  **강화학습 기반 매매**: **`Gymnasium`** 환경 내에서 \*\*`Stable Baselines3`\*\*의 **PPO** 에이전트가 거래를 학습하여, 최적의 매매 정책을 스스로 찾아내고 실행합니다.

## 주요 기능

  * **실시간 데이터 처리**: 키움증권 웹소켓 API를 통해 실시간 주식 체결 데이터를 수신하여 1분봉 데이터로 집계합니다.
  * **AI 기반 매매 결정**:
      * **LSTM 모델**: 다양한 기술적 보조지표를 피처로 사용하여 다음 캔들의 가격 등락을 예측합니다.
      * **강화학습 (PPO)**: 가상의 거래 환경(`StockTradingEnv`)에서 직접 거래를 시뮬레이션하며 누적 보상을 최대화하는 매매 정책을 학습합니다.
  * **자동 주문 실행**:
      * AI 모델의 결정에 따라 사전에 설정된 매매 전략(매수/매도 임계값, 손절매 등)을 기반으로 자동 주문을 실행합니다.
      * 거래 수수료, 세금, 슬리피지를 고려한 현실적인 주문 로직이 포함되어 있습니다.
  * **데이터 관리 및 전처리**: `pykrx`와 `FinanceDataReader`를 통해 주가 및 재무 데이터를 수집하고, `pandas-ta`로 기술적 지표를 생성하여 학습 데이터를 구축합니다.
  * **모듈화된 프레임워크**: 데이터 관리(`DataManager`), 특징 공학(`FeatureEngineer`), 위험 관리(`RiskManager`), 에이전트(`RLAgent`), 거래 환경(`StockTradingEnv`) 등 기능별로 클래스를 분리하여 관리 용이성을 높였습니다.

-----

## 시스템 아키텍처

이 시스템은 크게 **데이터 파이프라인**, **LSTM 모델링 파이프라인**, **강화학습 파이프라인** 세 부분으로 나뉩니다.

1.  **데이터 수집 및 전처리**

      * **스크립트**: `process_and_save_data_with_scalers.py`, `한국 주식 모든 종목.ipynb`
      * **프로세스**: KOSPI, KOSDAQ 시장의 OHLCV, 재무, 거시경제 데이터를 수집하고, 기술적 보조지표를 추가하여 정규화한 후 학습용 데이터셋(.parquet)과 스케일러(.joblib)를 생성합니다.

2.  **LSTM 모델링 및 실시간 매매**

      * **학습**: `모델링.ipynb`에서 전처리된 데이터를 사용하여 각 종목별 LSTM 모델(`.keras`)을 학습합니다.
      * **실행**: `websocket_client(final).py`가 실시간 데이터를 수집, 가공하여 학습된 LSTM 모델로 예측을 수행하고, 그 결과에 따라 주문을 실행합니다.

3.  **강화학습 모델링 및 실시간 매매**

      * **학습**: `final_주식.ipynb` 또는 `colab_stock.ipynb` 내에서 `StockTradingEnv`(거래 환경)와 `RLAgent`(에이전트)를 정의하고, PPO 알고리즘으로 학습을 진행하여 정책 모델(`.zip`)을 저장합니다.
      * **실행**: `websocket_client(멀티).py`가 실시간 데이터를 강화학습 환경의 '상태'로 변환하고, 학습된 RL 에이전트를 로드하여 다음 '행동'(매수/매도/유지)을 결정한 뒤 주문을 실행합니다.

-----

## 기술 스택

  * **프로그래밍 언어**: `Python`
  * **AI / 머신러닝**:
      * `TensorFlow (Keras)`: LSTM 모델 구축 및 학습
      * `Stable Baselines3`: 강화학습(PPO) 에이전트 구현
      * `Gymnasium`: 강화학습용 거래 환경 구축
      * `scikit-learn`: 데이터 정규화(MinMaxScaler)
  * **데이터 처리/분석**: `Pandas`, `NumPy`, `pandas-ta`, `pyarrow`
  * **데이터 수집**: `pykrx`, `FinanceDataReader`, `OpenDartReader`
  * **API 연동**: `Websockets`, `Requests` (키움증권 API)
  * **개발 환경**: `Jupyter Notebook`, `Google Colab`

-----

## 시작하기

### 1\. 사전 준비

  * **키움증권 API**: 모의투자 또는 실전투자 API 사용 신청 및 **토큰, 앱 키, 시크릿 키**를 발급받아야 합니다.
  * **Python 환경**: 64비트 Python 가상환경(`venv` 또는 `conda`) 사용을 권장합니다.

### 2\. 라이브러리 설치

```bash
# 기본 라이브러리
pip install tensorflow pandas numpy pandas_ta joblib pykrx FinanceDataReader OpenDartReader
# 강화학습 및 웹소켓 관련
pip install gymnasium "stable-baselines3[extra]" websockets==11.0.3 requests
```

### 3\. 실행 순서

이 프로젝트는 두 가지 모델링 방식(LSTM, RL)을 포함하므로, 원하는 방식에 따라 아래 순서를 따릅니다.

#### A. LSTM 모델 기반 자동매매

1.  **데이터 전처리 실행**: `process_and_save_data_with_scalers.py` 스크립트를 실행하여 학습에 필요한 데이터와 스케일러를 생성합니다.
2.  **LSTM 모델 학습**: `모델링.ipynb` 노트북을 열고 셀을 순차적으로 실행하여 각 종목에 대한 LSTM 모델을 학습시키고 저장합니다.
3.  **자동매매 클라이언트 실행**: `websocket_client(final).py` 파일 내의 API 키, 계좌번호 등 설정값을 수정한 후 실행합니다.

#### B. 강화학습 모델 기반 자동매매

1.  **데이터 수집 및 환경/에이전트 학습**:
      * `final_주식.ipynb` 또는 `colab_stock.ipynb` 노트북을 열고 셀을 순차적으로 실행합니다.
      * 노트북 내부에서 데이터 수집, 환경 구성, 에이전트 학습 및 모델 저장이 모두 이루어집니다. `mode='train'`으로 설정하여 실행합니다.
2.  **자동매매 클라이언트 실행**: `websocket_client(멀티).py` 파일 내의 설정값을 수정한 후 실행합니다. 이 클라이언트는 학습된 RL 모델을 불러와 매매를 수행합니다.

-----

## ⚙️ 주요 설정

  * **`websocket_client(final).py` (LSTM용)**: 파일 상단에서 `KIWOOM_TOKEN`, `KIWOOM_ACCOUNT_NO_STR`, `BUY_THRESHOLD`, `STOP_LOSS_PCT` 등 매매 전략과 관련된 주요 변수를 설정할 수 있습니다.
  * **`final_주식.ipynb` (RL용)**: `Config` 클래스 내에서 환경(`ENV_*`), 에이전트(`AGENT_*`), 위험 관리(`RISK_*`)와 관련된 다양한 파라미터를 상세하게 설정할 수 있습니다.

-----

**면책 조항**: 본 프로젝트는 학습 및 연구 목적으로 제작되었으며, 실제 투자에 대한 수익을 보장하지 않습니다. 모든 투자의 책임은 투자자 본인에게 있습니다.
