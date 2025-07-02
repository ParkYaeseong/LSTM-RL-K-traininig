# -*- coding: utf-8 -*-
# 파일명: process_and_save_data_with_scalers.py (multiprocessing 보호 추가)
# 실행 환경: 64비트 Python (myenv)
# 필요 라이브러리: pandas, numpy, pykrx, FinanceDataReader, pandas_ta, scikit-learn, joblib, pyarrow

import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import os
import time
import warnings
from datetime import datetime, timedelta
import joblib
import traceback
import gc
import multiprocessing # ★★★ multiprocessing 임포트 ★★★

warnings.filterwarnings('ignore')

# --- 함수 정의 등은 메인 블록 밖에 위치 가능 ---
# (만약 별도 함수가 있다면 여기에 정의)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★              메인 실행 로직 시작 (if __name__ == "__main__":)              ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
if __name__ == "__main__":

    # ★★★ multiprocessing 사용 시 Windows 환경 보호 ★★★
    multiprocessing.freeze_support()

    # --- 1. 설정값 정의 ---
    START_DATE = "20200101"
    END_DATE = "20250505"
    MARKETS_TO_PROCESS = ["KOSPI", "KOSDAQ"]

    # --- 경로 설정 ---
    # BASE_SAVE_PATH = '/content/drive/MyDrive/processed_stock_data_full_v1' # Colab 경로 예시
    BASE_SAVE_PATH = r'G:\내 드라이브\processed_stock_data_full_v2' # ★ 로컬 경로 확인 ★
    PROCESSED_DATA_PATH = os.path.join(BASE_SAVE_PATH, 'processed_parquet')
    SCALER_SAVE_PATH = os.path.join(BASE_SAVE_PATH, 'scalers')

    # --- 기술적 지표 및 피처 설정 ---
    SMA_LENGTHS = [5, 20, 60, 120]; RSI_LENGTH = 14; MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
    BBANDS_LENGTH = 20; BBANDS_STD = 2; ATR_LENGTH = 14; STOCH_K = 14; STOCH_D = 3

    COLUMNS_TO_NORMALIZE = [
        'open', 'high', 'low', 'close', 'volume', 'amount',
        *[f'SMA_{l}' for l in SMA_LENGTHS], f'RSI_{14}',
        f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        f'BBL_{BBANDS_LENGTH}_{float(BBANDS_STD)}', f'BBM_{BBANDS_LENGTH}_{float(BBANDS_STD)}', f'BBU_{BBANDS_LENGTH}_{float(BBANDS_STD)}', f'BBB_{BBANDS_LENGTH}_{float(BBANDS_STD)}', f'BBP_{BBANDS_LENGTH}_{float(BBANDS_STD)}',
        f'ATRr_{ATR_LENGTH}', 'OBV', f'STOCHk_{STOCH_K}_{STOCH_D}_3', f'STOCHd_{STOCH_K}_{STOCH_D}_3',
        'PBR', 'PER', 'USD_KRW'
    ]

    # --- 2. 경로 생성 ---
    print("--- 경로 생성 ---")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(SCALER_SAVE_PATH, exist_ok=True)
    print(f"처리된 데이터 저장 경로: {PROCESSED_DATA_PATH}")
    print(f"스케일러 저장 경로: {SCALER_SAVE_PATH}")

    # --- 3. 전역 데이터 수집 (거시경제, 재무) ---
    print("\n--- 전역 데이터 수집 (거시경제, 재무) ---")
    global_fetch_start = time.time()
    # ... (이전과 동일한 전역 데이터 수집 로직) ...
    print("  > USD/KRW 환율 데이터 수집..."); df_usdkrw = pd.DataFrame(columns=['USD_KRW'])
    try: df_usdkrw = fdr.DataReader('USD/KRW', START_DATE, END_DATE)[['Close']].rename(columns={'Close': 'USD_KRW'}); df_usdkrw.index = pd.to_datetime(df_usdkrw.index.date); print(f"    - 환율 데이터 수집 완료: {df_usdkrw.shape[0]} 행")
    except Exception as e: print(f"    - 오류: 환율 데이터 수집 실패 - {e}")
    print(f"  > 재무 데이터 수집 ({START_DATE}~{END_DATE})... (월말 기준)")
    all_tickers_list = []; df_fundamental_full = pd.DataFrame()
    for market in MARKETS_TO_PROCESS: all_tickers_list.extend(stock.get_market_ticker_list(date=END_DATE, market=market))
    monthly_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='M').strftime('%Y%m%d').tolist(); fundamental_dfs = []
    for date_str in monthly_dates:
        try:
            df_fund_part = stock.get_market_fundamental(date_str); df_fund_part['date'] = pd.to_datetime(date_str)
            fundamental_dfs.append(df_fund_part[['date', 'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']])
            if len(fundamental_dfs) % 12 == 0: print(f"    - {date_str} 까지 재무 데이터 수집 중...")
            time.sleep(0.5)
        except Exception as e: print(f"    - 경고: {date_str} 재무 데이터 수집 중 오류 - {e}"); continue
    if fundamental_dfs:
        df_fundamental_full = pd.concat(fundamental_dfs); df_fundamental_full.reset_index(inplace=True)
        df_fundamental_full.rename(columns={'티커': 'ticker'}, inplace=True); df_fundamental_full.set_index(['date', 'ticker'], inplace=True)
        df_fundamental_full = df_fundamental_full[~df_fundamental_full.index.duplicated(keep='last')]; print(f"    - 재무 데이터 (월말 기준) 처리 완료: {df_fundamental_full.shape[0]} 레코드")
    else: print("    - 재무 데이터 수집 실패 또는 데이터 없음.")
    global_fetch_end = time.time(); print(f"  > 전역 데이터 수집 총 소요 시간: {(global_fetch_end - global_fetch_start)/60:.2f} 분")


    # --- 4. 개별 종목 데이터 처리 및 저장 루프 ---
    print("\n--- 개별 종목 데이터 처리 및 저장 시작 ---")
    total_process_start_time = time.time()
    processed_count = 0
    failed_count = 0
    
    # 기존 all_tickers_unique 생성 로직 (KOSPI, KOSDAQ 모든 종목을 가져옴)
    #all_tickers_unique_original = sorted(list(set(all_tickers_list))) # 원본 리스트 유지 (선택 사항)
    #print(f"원래 처리 대상 총 {len(all_tickers_unique_original)}개 종목")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★      여기에 다음 한 줄을 추가하여 001450만 처리하도록 수정      ★★★
    # all_tickers_unique = ['001450']  # 특정 종목(001450)만 테스트하도록 설정
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    all_tickers_unique = sorted(list(set(all_tickers_list))) # 모든 KOSPI, KOSDAQ 종목
    print(f"수정된 처리 대상: {all_tickers_unique} (총 {len(all_tickers_unique)}개 종목 처리 시작...)")

    for i, ticker in enumerate(all_tickers_unique):
        print(f"\n[{i+1}/{len(all_tickers_unique)}] Ticker: {ticker} 처리 시작...")
        ticker_start_time = time.time()

        try:
            # 4.1 OHLCV 데이터 + 거래대금
            print("  DEBUG: 4.1 OHLCV 데이터 가져오기 시도...")
            df_ohlcv_raw = stock.get_market_ohlcv(START_DATE, END_DATE, ticker)
            if df_ohlcv_raw.empty:
                print(f"  >> OHLCV 데이터 없음 ({ticker}). 건너뜁니다."); failed_count += 1; continue
            
            print(f"  DEBUG: ({ticker}) df_ohlcv_raw.head():\n{df_ohlcv_raw.head().to_string()}")
            print(f"  DEBUG: ({ticker}) df_ohlcv_raw.columns: {df_ohlcv_raw.columns.tolist()}")

            df_stock = df_ohlcv_raw.copy() # 원본 DataFrame 복사
            df_stock.index.name = 'date'
            df_stock.index = pd.to_datetime(df_stock.index.date)
            
            # 표준 컬럼명으로 변경
            rename_map = {'시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume', '등락률': 'change'}
            df_stock.rename(columns=rename_map, inplace=True)

            # 'amount' (거래대금) 컬럼 생성 또는 사용
            if '거래대금' in df_ohlcv_raw.columns: # pykrx가 '거래대금' 컬럼을 직접 제공하는 경우
                print(f"  DEBUG: ({ticker}) '거래대금' 컬럼이 원본에 존재하여 사용합니다.")
                df_stock['amount'] = df_ohlcv_raw['거래대금'].astype(float)
            elif 'close' in df_stock.columns and 'volume' in df_stock.columns:
                print(f"  DEBUG: ({ticker}) '거래대금' 컬럼이 없어 'close' * 'volume'으로 'amount'를 계산합니다.")
                df_stock['amount'] = df_stock['close'].astype(float) * df_stock['volume'].astype(float)
            else:
                print(f"  DEBUG: WARNING - ({ticker}) '거래대금'을 가져오거나 계산할 수 없습니다. 'amount'를 0으로 설정합니다.")
                df_stock['amount'] = 0.0 # 대체값 (문제가 지속되면 이 경우를 자세히 조사해야 함)

            # amount 생성 후 상태 확인
            print(f"  DEBUG: ({ticker}) 최종 df_stock['amount'].head():\n{df_stock['amount'].head().to_string()}")
            print(f"  DEBUG: ({ticker}) 최종 df_stock['amount'].describe():\n{df_stock['amount'].describe().to_string()}")
            
            # 생성된 amount 컬럼의 NaN 값 0으로 채우기 (계산 과정에서 발생 가능성 고려)
            if df_stock['amount'].isnull().any():
                print(f"  DEBUG: ({ticker}) 'amount' 컬럼에 NaN 값 발견. 0으로 채웁니다.")
                df_stock['amount'].fillna(0, inplace=True)
            
            print(f"  DEBUG: ({ticker}) 최종 df_stock['amount'] 중 0이 아닌 값 개수: {(df_stock['amount'] != 0).sum()}")
            print(f"  DEBUG: ({ticker}) 최종 df_stock['amount'].isnull().sum() (fillna 후): {df_stock['amount'].isnull().sum()}")

            # 4.2 기술적 지표 추가
            print("  > 기술적 지표 계산...")
            custom_ta = ta.Strategy(
                name="TradingStrategyTA",
                ta=[ {"kind": "sma", "length": l} for l in SMA_LENGTHS ] + [
                    {"kind": "rsi", "length": RSI_LENGTH}, {"kind": "macd", "fast": MACD_FAST, "slow": MACD_SLOW, "signal": MACD_SIGNAL},
                    {"kind": "bbands", "length": BBANDS_LENGTH, "std": BBANDS_STD}, {"kind": "atr", "length": ATR_LENGTH},
                    {"kind": "obv"}, {"kind": "stoch", "k": STOCH_K, "d": STOCH_D}
                ]
            )
            # ★★★ 이 부분이 multiprocessing을 사용할 수 있음 ★★★
            df_stock.ta.strategy(custom_ta)

            # pandas_ta 컬럼 이름 표준화
            df_stock.rename(columns={
                f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
                f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
                f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
                f'BBL_{BBANDS_LENGTH}_{float(BBANDS_STD)}': f'BBL_{BBANDS_LENGTH}_{float(BBANDS_STD)}',
                f'BBM_{BBANDS_LENGTH}_{float(BBANDS_STD)}': f'BBM_{BBANDS_LENGTH}_{float(BBANDS_STD)}',
                f'BBU_{BBANDS_LENGTH}_{float(BBANDS_STD)}': f'BBU_{BBANDS_LENGTH}_{float(BBANDS_STD)}',
                f'BBB_{BBANDS_LENGTH}_{float(BBANDS_STD)}': f'BBB_{BBANDS_LENGTH}_{float(BBANDS_STD)}',
                f'BBP_{BBANDS_LENGTH}_{float(BBANDS_STD)}': f'BBP_{BBANDS_LENGTH}_{float(BBANDS_STD)}',
                f'ATRr_{ATR_LENGTH}': f'ATRr_{ATR_LENGTH}',
                f'STOCHk_{STOCH_K}_{STOCH_D}_3': f'STOCHk_{STOCH_K}_{STOCH_D}_3',
                f'STOCHd_{STOCH_K}_{STOCH_D}_3': f'STOCHd_{STOCH_K}_{STOCH_D}_3'
            }, inplace=True)

            # 4.3 전역 데이터 병합
            print("  > 전역 데이터 병합...")
            # ... (이전과 동일한 병합 로직) ...
            df_stock = pd.merge(df_stock, df_usdkrw, left_index=True, right_index=True, how='left'); df_stock['USD_KRW'].ffill(inplace=True); df_stock['USD_KRW'].bfill(inplace=True)
            if not df_fundamental_full.empty:
                if ticker in df_fundamental_full.index.get_level_values('ticker'):
                     df_fund_ticker = df_fundamental_full.xs(ticker, level='ticker'); df_stock = pd.merge_asof(df_stock.sort_index(), df_fund_ticker[['PBR', 'PER']].sort_index(), left_index=True, right_index=True, direction='backward')
                else: df_stock['PBR'] = np.nan; df_stock['PER'] = np.nan
            else: df_stock['PBR'] = np.nan; df_stock['PER'] = np.nan
            df_stock[['PBR', 'PER']].ffill(inplace=True); df_stock[['PBR', 'PER']].bfill(inplace=True)


            # 4.4 이벤트 특징 추가
            print("  > 이벤트 특징 추가...")
            df_stock['is_month_end'] = df_stock.index.is_month_end.astype(int)

            # 4.5 최종 데이터 정리 (NaN 처리)
            print("  > 최종 데이터 정리 (NaN 처리)...")
            initial_rows = len(df_stock)
            df_stock.dropna(inplace=True) # 모든 컬럼에 대해 NaN 있는 행 제거
            final_rows = len(df_stock)
            print(f"    - NaN 처리 후 데이터: {final_rows} 행 (처리 전: {initial_rows})")
            if final_rows < 60: print(f"  >> 최종 데이터 길이 부족 ({final_rows}). 건너뜁니다."); failed_count += 1; continue # continue 전에 gc.collect() 등 필요할 수 있음

            # ★★★ 디버깅 코드 추가 시작 ★★★
            print(f"  DEBUG: 스케일링 직전 df_stock 컬럼 목록: {df_stock.columns.tolist()}")
            if 'amount' in df_stock.columns:
                print(f"  DEBUG: 'amount' 컬럼의 상위 5개 값:\n{df_stock['amount'].head().to_string()}")
                print(f"  DEBUG: 'amount' 컬럼의 NaN 값 개수: {df_stock['amount'].isnull().sum()} (전체: {len(df_stock)})")
                print(f"  DEBUG: 'amount' 컬럼의 유니크한 값 개수 (NaN 제외): {len(df_stock['amount'].dropna().unique())}")
                if len(df_stock['amount'].dropna().unique()) <= 1:
                    print(f"  DEBUG: 'amount' 컬럼의 유니크한 값 (NaN 제외): {df_stock['amount'].dropna().unique()}")
            else:
                print(f"  DEBUG: !!! 'amount' 컬럼이 df_stock에 존재하지 않습니다 !!!")
            # ★★★ 디버깅 코드 추가 끝 ★★★
            
            # 4.6 데이터 정규화 및 스케일러 저장
            print("  > 데이터 정규화 및 스케일러 저장...")
            df_normalized = df_stock.copy()
            scalers_to_save = {}
            for col in COLUMNS_TO_NORMALIZE:
                if col in df_normalized.columns:
                    data_to_scale = df_normalized[[col]].values.astype(float)
                    unique_vals = df_normalized[col].dropna().unique()
                    if len(unique_vals) > 1:
                         scaler = MinMaxScaler()
                         scaled_data = scaler.fit_transform(data_to_scale)
                         df_normalized[col + '_norm'] = scaled_data
                         scalers_to_save[col] = scaler # 원본 컬럼 이름으로 스케일러 저장
                    else: df_normalized[col + '_norm'] = 0.0; print(f"    - 경고: '{col}' 값 동일. 0 정규화.")
                else: print(f"    - 경고: 컬럼 '{col}' 없음. '_norm' 0으로 채움."); df_normalized[col + '_norm'] = 0.0

            scaler_filename = os.path.join(SCALER_SAVE_PATH, f'{ticker}_scalers.joblib')
            try: joblib.dump(scalers_to_save, scaler_filename); print(f"    - 스케일러 저장 완료: {scaler_filename}")
            except Exception as e: print(f"    - 오류: 스케일러 저장 실패 ({ticker}): {e}"); traceback.print_exc()

            # 4.7 최종 데이터 파일 저장 (Parquet)
            print("  > 최종 데이터 Parquet 파일 저장...")
            final_save_path = os.path.join(PROCESSED_DATA_PATH, f"{ticker}.parquet")
            df_normalized.to_parquet(final_save_path)
            print(f"    - 저장 완료: {final_save_path}")
            processed_count += 1

        except Exception as e:
            print(f"  >> 오류: 티커 {ticker} 처리 중 예외 발생: {e}")
            traceback.print_exc()
            failed_count += 1
            continue
        finally:
            ticker_end_time = time.time()
            print(f"  Ticker: {ticker} 처리 시간: {ticker_end_time - ticker_start_time:.2f} 초")
            gc.collect()

    # --- 최종 결과 출력 ---
    total_process_end_time = time.time()
    print("\n--- 전체 개별 종목 데이터 처리 및 저장 완료 ---")
    print(f"총 소요 시간: {(total_process_end_time - total_process_start_time) / 60:.2f} 분")
    print(f"성공적으로 처리/저장된 종목 수: {processed_count}")
    print(f"실패 또는 건너뛴 종목 수: {failed_count}")