# -*- coding: utf-8 -*-
# 파일명: model_server_64bit.py (예시)
# 실행 환경: 64비트 Python (myenv)

import zmq
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json # 데이터 직렬화/역직렬화 위해

# --- 설정 ---
IPC_ADDRESS = "tcp://127.0.0.1:5555" # 통신 주소 (API 클라이언트와 동일하게)
MODEL_SAVE_PATH = r'G:\내 드라이브\lstm_models_per_stock_v1' # 모델 경로 (로컬 경로)

# --- 모델 로딩 함수 (필요시 개선) ---
# 간단히 딕셔너리에 로드 (메모리 주의)
loaded_models = {}
def load_model_for_ticker(ticker):
    if ticker in loaded_models:
        return loaded_models[ticker]
    else:
        model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}.keras")
        if os.path.exists(model_path):
            print(f"모델 로딩: {model_path}")
            loaded_models[ticker] = tf.keras.models.load_model(model_path)
            return loaded_models[ticker]
        else:
            print(f"경고: 모델 파일 없음: {model_path}")
            return None

# --- ZeroMQ 서버 설정 ---
context = zmq.Context()
socket = context.socket(zmq.REP) # 응답(Reply) 소켓
socket.bind(IPC_ADDRESS)
print(f"모델 서버 시작됨. {IPC_ADDRESS} 에서 요청 대기 중...")

# --- 요청 처리 루프 ---
while True:
    try:
        # 1. 클라이언트로부터 메시지(요청) 수신 (JSON 형식 가정)
        message_str = socket.recv_string()
        print(f"\n수신된 요청: {message_str}")
        request_data = json.loads(message_str) # JSON 문자열 -> 딕셔너리

        # 2. 요청 분석 및 처리
        response_data = {} # 응답으로 보낼 데이터
        request_type = request_data.get('type')

        if request_type == 'predict':
            ticker = request_data.get('ticker')
            sequence_data_list = request_data.get('sequence') # 리스트 형태로 전달받음

            if ticker and sequence_data_list:
                model = load_model_for_ticker(ticker)
                if model:
                    # 리스트를 NumPy 배열로 변환 (형태 확인 필요)
                    # sequence_data = np.array(sequence_data_list).reshape(1, TIME_STEPS, num_features) # 형태 맞춰주기
                    # 여기서 TIME_STEPS와 num_features는 모델 학습 시와 동일해야 함
                    # 실제로는 sequence_data_list의 구조를 보고 정확히 변환해야 함!
                    # 예시: shape(1, 20, 27) 로 변환 가정
                    try:
                        sequence_np = np.array(sequence_data_list).astype(np.float32)
                        if sequence_np.ndim == 2: # (TIME_STEPS, num_features) 형태라면
                             input_sequence = np.expand_dims(sequence_np, axis=0) # (1, TIME_STEPS, num_features)
                             # 입력 shape 확인
                             print(f"모델 입력 shape: {input_sequence.shape}")
                             # 예측 수행
                             prediction_norm = model.predict(input_sequence, verbose=0)
                             predicted_value = float(prediction_norm[0][0]) # 스칼라 값으로 추출
                             response_data = {'status': 'success', 'prediction': predicted_value}
                             print(f"예측 완료 ({ticker}): {predicted_value}")
                        else:
                             raise ValueError("수신된 시퀀스 데이터의 차원이 올바르지 않습니다.")

                    except Exception as e:
                         print(f"예측 처리 중 오류: {e}")
                         response_data = {'status': 'error', 'message': str(e)}

                else:
                    response_data = {'status': 'error', 'message': f"모델 파일을 찾을 수 없음: {ticker}"}
            else:
                 response_data = {'status': 'error', 'message': '요청 데이터 부족 (ticker 또는 sequence)'}

        elif request_type == 'ping': # 간단한 연결 확인용
             response_data = {'status': 'success', 'message': 'pong'}

        else:
            response_data = {'status': 'error', 'message': '알 수 없는 요청 타입'}

        # 3. 클라이언트에 응답 전송 (JSON 형식)
        response_str = json.dumps(response_data) # 딕셔너리 -> JSON 문자열
        socket.send_string(response_str)
        print(f"응답 전송: {response_str}")

    except zmq.ZMQError as e:
        print(f"ZeroMQ 오류 발생: {e}")
        time.sleep(1)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        # 잘못된 형식의 요청에는 에러 응답 전송 시도 (선택적)
        try:
             socket.send_string(json.dumps({'status': 'error', 'message': '잘못된 JSON 형식'}))
        except zmq.ZMQError:
             pass
    except KeyboardInterrupt:
        print("\n모델 서버 종료 중...")
        break
    except Exception as e:
        print(f"처리 중 예외 발생: {e}")
        # 예외 발생 시 에러 응답 전송 시도 (선택적)
        try:
             socket.send_string(json.dumps({'status': 'error', 'message': f'서버 내부 오류: {e}'}))
        except zmq.ZMQError:
             pass
        time.sleep(1) # 오류 후 잠시 대기

# --- 종료 처리 ---
socket.close()
context.term()
print("모델 서버 종료됨.")