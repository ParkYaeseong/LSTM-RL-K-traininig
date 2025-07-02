# -*- coding: utf-8 -*-
# 파일명: kiwoom_api_client_32bit.py (수정 9 - Kiwoom 직접 사용 + 수동 루프)
# 실행 환경: 32비트 Python (py32_kiwoom)

import sys
from PyQt5.QtWidgets import QApplication # PyQt 이벤트 처리를 위해 필요
from pykiwoom.kiwoom import * # Kiwoom 클래스 직접 사용
import time
import zmq                          # ZeroMQ (IPC 통신용)
import json                         # 데이터 직렬화/역직렬화 (JSON)
import numpy as np                  # 데이터 처리용 (선택적)
import traceback                    # 오류 상세 출력용
import gc                           # 가비지 컬렉션
import signal                       # 종료 시그널 처리용

class KiwoomAPIClient:
    def __init__(self):
        """ 클래스 초기화 (Kiwoom 직접 사용) """
        print("Kiwoom API 클라이언트 초기화 (Kiwoom 직접 사용)...")
        # QApplication 인스턴스 확인 및 생성 (중요)
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        self.kiwoom = Kiwoom() # ★★★ KiwoomManager 대신 Kiwoom 직접 생성 ★★★
        self.connected = False
        self.ipc_connected = False
        self.default_account = None
        self.is_running = True # 루프 제어 플래그

        self._set_event_handlers()  # 이벤트 핸들러 정의
        self._setup_ipc()           # ZeroMQ IPC 설정
        self._connect_api()         # API 접속 시도 (block=True)

        if self.connected:
            self.register_real_time_data() # 로그인 성공 시 실시간 등록

    def _set_event_handlers(self):
        """ 이벤트 핸들러 정의 (자동 연결 방식) """
        print("이벤트 핸들러 설정 (메소드 정의)...")
        # .connect() 호출 사용 안 함
        print("이벤트 핸들러 정의 완료 (자동 연결 대기).")

    def _setup_ipc(self):
        """ ZeroMQ 클라이언트 설정 """
        print("ZeroMQ 클라이언트 설정...")
        # ... (이전과 동일한 IPC 설정 코드) ...
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.ipc_address = "tcp://127.0.0.1:5555"
        try:
            self.socket.connect(self.ipc_address)
            print(f"모델 서버에 연결 시도: {self.ipc_address}")
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.setsockopt(zmq.LINGER, 0)
            ping_request = {'type': 'ping'}
            self.socket.send_string(json.dumps(ping_request))
            response_str = self.socket.recv_string()
            response = json.loads(response_str)
            if response.get('status') == 'success' and response.get('message') == 'pong':
                print("모델 서버 연결 확인 완료 (ping-pong 성공).")
                self.ipc_connected = True
            else: print(f"모델 서버 연결 확인 실패: {response}"); self.ipc_connected = False
        except Exception as e: print(f"IPC 설정 중 오류 발생: {e}"); self.ipc_connected = False

    def _connect_api(self):
        """ API 접속 시도 (블로킹 방식) """
        try:
            print("키움증권 Open API+ 접속 시도 (로그인 대기)...")
            self.kiwoom.CommConnect(block=True) # 로그인 완료까지 대기
            if self.kiwoom.GetConnectState() == 1:
                print("API 접속 및 로그인 성공.")
                self.connected = True
                self._get_login_info()
            else:
                print("API 접속 또는 로그인 실패.")
                self.connected = False
        except Exception as e:
            print(f"API 연결/로그인 중 오류 발생: {e}")
            self.connected = False

    def _get_login_info(self):
        """ 로그인 성공 후 계좌 정보 등 확인 """
        # ... (이전과 동일) ...
        if not self.connected: return
        try:
            account_list_raw = self.kiwoom.GetLoginInfo("ACCNO")
            account_list = account_list_raw.split(';')[:-1]
            user_id = self.kiwoom.GetLoginInfo("USER_ID")
            user_name = self.kiwoom.GetLoginInfo("USER_NAME")
            server_type = "모의투자" if self.kiwoom.GetLoginInfo("GetServerGubun") == "1" else "실서버"
            print("-" * 30)
            print(f"서버 구분: {server_type}")
            print(f"사용자 ID: {user_id}")
            print(f"사용자 이름: {user_name}")
            print(f"보유 계좌: {account_list}")
            if account_list: self.default_account = account_list[0]; print(f"기본 사용 계좌: {self.default_account}")
            else: self.default_account = None; print("사용 가능한 계좌가 없습니다.")
            print("-" * 30)
        except Exception as e: print(f"로그인 정보 조회 중 오류: {e}")

    def register_real_time_data(self, codes="005930"):
        """ 지정한 종목들의 실시간 데이터 수신 등록 """
        # ... (이전과 동일) ...
        if not self.connected: return
        fids = "10;15;11;12;27;28"
        screen_no = "1000"
        try:
            self.kiwoom.SetRealReg(screen_no, codes, fids, "0")
            print(f"실시간 데이터 등록 요청: 화면번호={screen_no}, 종목코드={codes}, FID={fids}")
        except Exception as e: print(f"실시간 데이터 등록 중 오류 발생: {e}")

    # --- 이벤트 핸들러 메소드 정의 ---
    # ... (_handler_real_data, _handler_tr_data, _handler_chejan_data, _handler_receive_msg - 이전과 동일, try/except 포함) ...
    def _handler_real_data(self, code, real_type, real_data):
        try:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            price = abs(int(self.kiwoom.GetCommRealData(code, 10)))
            volume = abs(int(self.kiwoom.GetCommRealData(code, 15)))
            change_rate = float(self.kiwoom.GetCommRealData(code, 12))
            ask_price = abs(int(self.kiwoom.GetCommRealData(code, 27)))
            bid_price = abs(int(self.kiwoom.GetCommRealData(code, 28)))
            print(f"[{current_time}] 실시간 [{code}] {real_type} | "
                  f"현재가: {price:,}, 등락률: {change_rate:+.2f}%, "
                  f"거래량: {volume:,}, 매도호가: {ask_price:,}, 매수호가: {bid_price:,}")
        except ValueError as ve: print(f"[{current_time}] 실시간 [{code}] 데이터 처리 중 값 오류. real_data: {real_data}, error: {ve}")
        except Exception as e: print(f"실시간 데이터(_handler_real_data) 처리 중 예외 발생: {e}"); traceback.print_exc()

    def _handler_tr_data(self, screen_no, rqname, trcode, recordname, prev_next, data_len, err_code, msg1, msg2):
        try: print(f"TR 데이터 수신: 화면={screen_no}, 요청명={rqname}, TR코드={trcode}, 연속={prev_next}, 메시지={msg1}")
        except Exception as e: print(f"TR 데이터(_handler_tr_data) 처리 중 예외 발생: {e}"); traceback.print_exc()

    def _handler_chejan_data(self, gubun, item_cnt, fid_list):
        try: print(f"체결 데이터 수신: 구분={gubun}, 항목수={item_cnt}, FID목록={fid_list}")
        except Exception as e: print(f"체결 데이터(_handler_chejan_data) 처리 중 예외 발생: {e}"); traceback.print_exc()

    def _handler_receive_msg(self, screen_no, rqname, trcode, msg):
        try: print(f"서버 메시지 수신: 화면={screen_no}, 요청명={rqname}, TR코드={trcode}, 메시지={msg}")
        except Exception as e: print(f"서버 메시지(_handler_receive_msg) 처리 중 예외 발생: {e}"); traceback.print_exc()

    def request_prediction(self, ticker, sequence_data):
        """ 모델 서버(64bit)에 예측 요청 """
        # ... (이전과 동일) ...
        if not self.ipc_connected or not self.socket or self.socket.closed: return None
        if isinstance(sequence_data, np.ndarray): sequence_list = sequence_data.tolist()
        elif isinstance(sequence_data, list): sequence_list = sequence_data
        else: print("오류: sequence_data는 NumPy 배열 또는 리스트여야 합니다."); return None
        request = { 'type': 'predict', 'ticker': ticker, 'sequence': sequence_list }
        try:
            self.socket.send_string(json.dumps(request))
            response_str = self.socket.recv_string()
            response = json.loads(response_str)
            if response.get('status') == 'success': return response.get('prediction')
            else: print(f"모델 예측 실패: {response.get('message')}"); return None
        except zmq.Again: print("모델 서버 응답 시간 초과."); return None
        except Exception as e: print(f"모델 서버 통신/처리 오류: {e}"); return None

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★         run 메소드 및 __main__ 블록 재수정 (v9 방식)         ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    def run(self):
        """ 메인 이벤트 처리 루프 (로그인 성공 후 호출됨) """
        print(">>> API 클라이언트 메인 루프 시작 (이벤트 처리 대기)...")

        # QApplication 인스턴스 가져오기 (init에서 생성/확인됨)
        app = QApplication.instance()
        if app is None:
             print(">>> 오류: QApplication 인스턴스가 없습니다! 이벤트 처리가 불가합니다.")
             self.is_running = False # 루프 진입 방지 또는 즉시 종료
             return # run 메소드 종료

        loop_count = 0
        while self.is_running:
            loop_count += 1
            # print(f"\n>>> 메인 루프 반복 #{loop_count}...") # 너무 자주 출력될 수 있으므로 주석 처리

            try:
                # --- 연결 상태 확인 (덜 자주 확인하거나 제거 고려) ---
                # 매 루프마다 GetConnectState 호출은 부하가 될 수 있고,
                # 로그인 직후에는 문제가 발생했었으므로 일단 주석 처리
                # if loop_count % 100 == 0: # 예: 10초마다 확인 (sleep 0.1초 기준)
                #    current_connect_state = self.kiwoom.GetConnectState()
                #    if not self.connected or not current_connect_state:
                #        print(">>> !! API 연결 끊김 감지됨 (run 루프 내부) !!")
                #        self.is_running = False
                #        break
                # ----------------------------------------------------

                # --- PyQt 이벤트 처리 ---
                app.processEvents()

                # --- 주기적 작업 (필요시 여기에 추가) ---

                # --- CPU 부하 감소 ---
                time.sleep(0.05) # 0.05 ~ 0.1초 정도가 적당

            except KeyboardInterrupt:
                print(">>> Ctrl+C 감지! 종료 절차 시작...")
                self.is_running = False
            except Exception as e:
                 print(f">>> 메인 루프 실행 중 예외 발생: {e}")
                 traceback.print_exc()
                 self.is_running = False # 오류 발생 시 루프 종료

        print(f">>> 메인 이벤트 루프 종료됨. is_running = {self.is_running}")

    def disconnect(self):
         """ 연결 종료 및 자원 정리 """
         print(">>> disconnect 메소드 호출됨.")
         self.is_running = False # run 메소드 루프 종료 보장
         # ... (이전 disconnect 코드와 동일: RealData 해제, ZeroMQ 정리) ...
         print("클라이언트 종료 절차 시작...")
         try:
              if hasattr(self, 'connected') and self.connected: # connected 속성 확인
                   # 실시간 등록 해제 시 화면번호 확인 필요
                   self.kiwoom.DisconnectRealData("1000")
                   print("실시간 데이터 수신 해제 완료.")
         except Exception as e: print(f"실시간 데이터 해제 중 오류: {e}")
         if hasattr(self, 'socket') and self.socket and not self.socket.closed: self.socket.close()
         if hasattr(self, 'context') and self.context and not self.context.closed: self.context.term()
         print("ZeroMQ 연결 종료됨.")
         self.connected = False
         self.ipc_connected = False


if __name__ == "__main__":
    # QApplication 인스턴스는 스크립트 전체에서 하나만 존재해야 함
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 자동 종료 방지는 수동 루프 사용 시 크게 의미 없음
    # app.setQuitOnLastWindowClosed(False)

    print("KiwoomAPIClient 인스턴스 생성...")
    client = KiwoomAPIClient() # __init__ 실행 (로그인 포함)

    # 시그널 핸들러 설정 (Ctrl+C 등 정상 종료 유도)
    def signal_handler(sig, frame):
        print(f'종료 시그널 {sig} 감지! disconnect 호출...')
        client.disconnect()
        # QApplication.quit() # run 메소드 루프를 종료시키기 위해 quit() 호출 불필요
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Terminate

    # 로그인 성공 여부 확인 후 메인 루프 실행
    if client.connected:
        print(">>> 로그인 성공 확인. client.run() 호출...")
        try:
            client.run() # ★★★ 직접 만든 이벤트 처리 루프 실행 ★★★
        finally:
            # run 메소드가 어떤 이유로든 종료되면 disconnect 호출
            print(">>> run() 메소드 종료 확인. 최종 disconnect 호출...")
            client.disconnect()
    else:
        print(">>> API 연결 실패로 메인 루프를 시작하지 않습니다.")
        client.disconnect() # 실패 시에도 혹시 모를 자원 정리

    print(">>> 프로그램 최종 종료.")
    sys.exit(0) # 정상 종료 코드