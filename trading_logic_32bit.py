# -*- coding: utf-8 -*-
# 파일명: trading_logic_32bit.py (수정 8 - 테스트 버전)
# 실행 환경: 32비트 Python (py32_kiwoom)

import sys
from pykiwoom.manager import KiwoomManager # ★★★ KiwoomManager만 임포트 (단순화) ★★★
import time
import zmq
import json
import numpy as np
import traceback
import pprint
import gc
import signal

class TradingBot:
    def __init__(self):
        """ 트레이딩 봇 초기화 (최소화) """
        print("트레이딩 봇 초기화...")
        self.ipc_connected = False
        self.model_server_socket = None
        self.zmq_context = None
        self.connected = False # API 연결 상태
        self.default_account = None
        self.km = None # KiwoomManager 인스턴스

        # KiwoomManager 인스턴스 생성 시도
        print("KiwoomManager 생성 및 실행 시도...")
        try:
            self.km = KiwoomManager() # 내부적으로 서브프로세스 실행 및 로그인 창 표시 기대
            print("KiwoomManager 객체 생성 완료. (서브프로세스 초기화 및 로그인 진행 중일 수 있음)")
            # ★★★ __init__에서는 더 이상 로그인 완료 대기나 후속 작업 안 함 ★★★
        except Exception as e:
            print(f"KiwoomManager 생성 중 오류: {e}")
            traceback.print_exc()
            self.km = None # 실패 시 None 처리

    # ★★★ _setup_ipc, get_account_info, register_real_time_data 등은 나중에 호출 ★★★
    def _setup_ipc(self):
        # ... (이전과 동일) ...
        print("ZeroMQ 클라이언트 설정...")
        self.zmq_context = zmq.Context()
        self.model_server_socket = self.zmq_context.socket(zmq.REQ)
        ipc_address = "tcp://127.0.0.1:5555"
        try:
            self.model_server_socket.connect(ipc_address)
            print(f"모델 서버에 연결 시도: {ipc_address}")
            self.model_server_socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.model_server_socket.setsockopt(zmq.LINGER, 0)
            ping_request = {'type': 'ping'}
            self.model_server_socket.send_string(json.dumps(ping_request))
            response_str = self.model_server_socket.recv_string()
            response = json.loads(response_str)
            if response.get('status') == 'success' and response.get('message') == 'pong':
                print("모델 서버 연결 확인 완료 (ping-pong 성공).")
                self.ipc_connected = True
            else: print(f"모델 서버 연결 확인 실패: {response}"); self.ipc_connected = False
        except Exception as e: print(f"IPC 설정 중 오류 발생: {e}"); self.ipc_connected = False

    def get_account_info(self):
        # ... (이전과 동일) ...
        if not self.connected or not self.km: return
        try:
             print("계좌 정보 요청...")
             self.km.put_method(("GetLoginInfo", "ACCNO"))
             account_list_raw = self.km.get_method()
             self.km.put_method(("GetLoginInfo", "USER_NAME"))
             user_name = self.km.get_method()
             account_list = account_list_raw.split(';')[:-1]
             print(f"사용자: {user_name}, 계좌: {account_list}")
             if account_list: self.default_account = account_list[0]; print(f"기본 계좌 설정: {self.default_account}")
        except Exception as e: print(f"계좌 정보 요청 중 오류 발생: {e}")

    def register_real_time_data(self, codes="005930"):
         if not self.connected or not self.km: return
         print("실시간 데이터 등록 시도...")
         # ... (이전과 동일한 등록 로직, km 사용) ...
         fids = "10;15;11;12;27;28"
         screen_no = "1000"
         try:
             # KiwoomManager를 통해 메소드 호출
             reg_cmd = ("SetRealReg", [screen_no, codes, fids, "0"])
             self.km.put_method(reg_cmd)
             result = self.km.get_method() # SetRealReg는 반환값이 의미 없을 수 있음
             print(f"실시간 데이터 등록 요청 결과: {result} (성공 여부 확인 필요)")
             # ★★★ KiwoomManager에서 실시간 데이터를 어떻게 받는지 별도 확인 필요 ★★★
         except Exception as e:
             print(f"실시간 데이터 등록 중 오류 발생: {e}")


    def request_prediction(self, ticker, sequence_data):
        # ... (이전과 동일) ...
        if not self.ipc_connected: return None
        if isinstance(sequence_data, np.ndarray): sequence_list = sequence_data.tolist()
        elif isinstance(sequence_data, list): sequence_list = sequence_data
        else: print("오류: sequence_data는 NumPy 배열 또는 리스트여야 합니다."); return None
        request = { 'type': 'predict', 'ticker': ticker, 'sequence': sequence_list }
        try:
            self.model_server_socket.send_string(json.dumps(request))
            response_str = self.model_server_socket.recv_string()
            response = json.loads(response_str)
            if response.get('status') == 'success': return response.get('prediction')
            else: print(f"모델 예측 실패: {response.get('message')}"); return None
        except zmq.Again: print("모델 서버 응답 시간 초과."); return None
        except Exception as e: print(f"모델 서버 통신/처리 오류: {e}"); return None

    def run_trading_loop(self):
        """ 메인 트레이딩 로직 루프 """
        if not self.connected or not self.km or not self.ipc_connected:
             print("API 또는 IPC 미연결 상태. 트레이딩 루프를 시작할 수 없습니다.")
             return
        print("\n--- 메인 트레이딩 루프 시작 (Ctrl+C로 종료) ---")
        # ... (이전 trading loop 로직 유지, 단 데이터 수신 방식은 확인 필요) ...
        while True:
            try:
                # === 실시간/체결 데이터 수신 (KiwoomManager 방식 확인 필요!) ===
                # print("실시간/체결 데이터 확인 중...") # 예시

                # === 임시 로직: 주기적 예측 요청 ===
                ticker_to_predict = '005930'
                dummy_sequence = np.random.rand(20, 27).tolist()
                prediction = self.request_prediction(ticker_to_predict, dummy_sequence)
                if prediction is not None:
                     print(f"[{ticker_to_predict}] 예측된 정규화 값: {prediction:.6f}")

                time.sleep(5)
            except KeyboardInterrupt: print("Ctrl+C 감지. 트레이딩 루프 종료."); break
            except Exception as e: print(f"트레이딩 루프 중 오류 발생: {e}"); traceback.print_exc(); time.sleep(5)

    def disconnect(self):
        """ 자원 정리 """
        print("봇 종료 및 자원 정리...")
        # ... (이전과 동일한 disconnect 로직) ...
        if hasattr(self, 'model_server_socket') and self.model_server_socket and not self.model_server_socket.closed: self.model_server_socket.close()
        if hasattr(self, 'zmq_context') and self.zmq_context and not self.zmq_context.closed: self.zmq_context.term()
        print("ZeroMQ 연결 종료됨.")
        if hasattr(self, 'km'): print("KiwoomManager 종료 시도..."); # self.km.terminate() # 종료 메소드 확인 필요
        print("정리 완료.")


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★              이 아래 __main__ 부분이 크게 변경됨              ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
if __name__ == "__main__":
    print("TradingBot 인스턴스 생성 시작...")
    bot = TradingBot() # __init__ 실행 -> KiwoomManager 생성 시도

    if bot.km: # KiwoomManager 객체가 성공적으로 생성되었다면
        print("\nKiwoomManager 인스턴스 생성됨.")
        print("★★★ 중요: 키움증권 로그인 창이 뜨면 로그인해주세요. (30초 대기) ★★★")
        time.sleep(30) # 사용자가 로그인할 시간을 줌

        # 로그인 완료 후, 연결 상태 확인 및 후속 작업 시도
        print("\n로그인 완료 가정 후, API 연결 상태 확인 및 후속 작업 시도...")
        try:
            # GetConnectState 호출 시도
            bot.km.put_method(("GetConnectState",))
            connect_state = bot.km.get_method() # 결과 대기
            print(f"GetConnectState() 결과: {connect_state}")

            if connect_state == 1:
                print("API 연결 성공 확인.")
                bot.connected = True
                bot.get_account_info()      # 계좌 정보 가져오기
                bot.register_real_time_data() # 실시간 데이터 등록
                bot._setup_ipc()            # IPC 설정 (모델 서버 연결)

                # 트레이딩 루프 시작 (연결이 모두 성공했을 때)
                if bot.ipc_connected:
                     bot.run_trading_loop()
                else:
                     print("모델 서버(IPC) 연결 실패로 트레이딩 루프를 시작하지 않습니다.")

            else:
                print("API 연결 실패 확인.")
                bot.connected = False

        except Exception as e:
            print(f"로그인 후 작업 또는 API 호출 중 오류 발생: {e}")
            traceback.print_exc()
        finally:
            # 프로그램 종료 전 정리 작업
            bot.disconnect()
    else:
        print("KiwoomManager 생성 실패로 프로그램을 종료합니다.")

    print("프로그램 최종 종료.")