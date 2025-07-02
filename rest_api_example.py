# -*- coding: utf-8 -*-
# 파일명: rest_api_example.py (토큰 발급 + 계좌 조회 통합)
# 실행 환경: 64비트 Python (myenv)

import requests
import json
import pprint
import traceback
import time # 혹시 필요할 수 있으므로 유지

# ★★★ 중요: 키움증권에서 발급받은 실제 값으로 변경하세요 ★★★
APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss"  
APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw" 
ACCOUNT_NO = 81026247
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# API URL 설정
IS_MOCK_TRADING = True # 모의투자 사용 여부 (True: 모의투자, False: 실전투자)

if IS_MOCK_TRADING:
    BASE_URL = "https://mockapi.kiwoom.com"
    print(">> 모의투자 환경으로 설정되었습니다.")
else:
    BASE_URL = "https://api.kiwoom.com"
    print(">> 실전투자 환경으로 설정되었습니다.")

TOKEN_URL = f"{BASE_URL}/oauth2/token"
API_URL_ACCOUNT = f"{BASE_URL}/api/dostk/acnt" # 계좌 관련 API 기본 URL

# TR ID
TR_ID_GET_CASH = "kt00001" # 예수금 상세 현황 요청


def get_access_token(app_key, app_secret):
    """ OAuth 2.0 접근 토큰 발급 요청 """
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "secretkey": app_secret
    }
    try:
        print(f"\n[1] 접근 토큰 발급 요청 시도 -> {TOKEN_URL}")
        res = requests.post(TOKEN_URL, headers=headers, data=json.dumps(body))
        res.raise_for_status() # 오류 발생 시 예외 발생
        token_data = res.json()

        if token_data.get('token'):
            print("  > 접근 토큰 발급 성공!")
            full_token = token_data['token']
            print(f"    - Access Token (Full): {full_token}") # 전체 토큰 출력
            return full_token
        else:
            print("  > 접근 토큰 발급 실패:")
            print(f"    - Return Code: {token_data.get('return_code')}")
            print(f"    - Return Msg: {token_data.get('return_msg')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  > HTTP 요청 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
             try: print(f"  > 오류 응답 내용: {e.response.json()}")
             except json.JSONDecodeError: print(f"  > 오류 응답 내용 (텍스트): {e.response.text}")
        return None
    except Exception as e:
        print(f"  > 토큰 발급 중 예외 발생: {e}")
        return None


def get_cash_balance(token, acc_no):
    """ 예수금 상세 현황 조회 (kt00001) """
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {token}",
        "api-id": TR_ID_GET_CASH
    }
    body = {
        "계좌번호": acc_no,
        "qry_tp": "1"
    }

    try:
        print(f"\n[2] 예수금 상세 현황({TR_ID_GET_CASH}) 요청 시도 -> {API_URL_ACCOUNT}")
        print(f"  > 요청 Body: {json.dumps(body)}")
        res = requests.post(API_URL_ACCOUNT, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        response_data = res.json()

        print(f"  > 응답 수신:")
        print(f"    - Return Code: {response_data.get('return_code')}")
        print(f"    - Return Msg: {response_data.get('return_msg')}")

        # ★★★ 전체 응답 내용 출력 부분 주석 해제 ★★★
        #print("\n  > (전체 응답 내용):")
        #pprint.pprint(response_data) # 전체 응답 구조와 키 확인용
        # ★★★★★★★★★★★★★★★★★★★★★★★★★

        if response_data.get('return_code') == 0:
            print("\n  > [ 예수금 상세 현황 ]")

            # ★★★ 실제 키 이름 사용 및 숫자 변환 ★★★
            try:
                # response_data에서 직접 키로 접근 (output 중첩 없음)
                # .get(key, '0')으로 키가 없을 경우 기본값 '0' 사용 후 int 변환
                deposit_str = response_data.get('d2_entra', '0') # D+2 예수금 추정 키
                available_cash_str = response_data.get('ord_alow_amt', '0') # 주문가능금액 확실한 키

                # 문자열을 정수(int)로 변환, 변환 실패 시 0으로 처리
                deposit = int(deposit_str) if deposit_str.isdigit() else 0
                available_cash = int(available_cash_str) if available_cash_str.isdigit() else 0

                # 보기 좋게 콤마(,) 추가하여 출력
                print(f"    - 예수금 (d+2) ['d2_entra']: {deposit:,}")
                print(f"    - 주문가능금액 ['ord_alow_amt']: {available_cash:,}")

            except Exception as e:
                 print(f"    - 오류: 응답 데이터 파싱 또는 숫자 변환 중 오류 발생 - {e}")
                 traceback.print_exc()
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

            return response_data
        else:
            print("  > API 호출 실패.")
            pprint.pprint(response_data) # 실패 시 전체 내용 출력
            return None
        
    except requests.exceptions.RequestException as e:
        # ... (오류 처리 동일) ...
        print(f"  > HTTP 요청 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
             try: print(f"  > 오류 응답 내용: {e.response.json()}")
             except json.JSONDecodeError: print(f"  > 오류 응답 내용 (텍스트): {e.response.text}")
        return None
    except Exception as e:
        # ... (오류 처리 동일) ...
        print(f"  > API 호출 중 예외 발생: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if APP_KEY == "YOUR_APP_KEY" or APP_SECRET == "YOUR_APP_SECRET":
        print("!!! 중요: 코드 상단의 APP_KEY 와 APP_SECRET 변수에 실제 발급받은 키를 입력하세요 !!!")
    elif ACCOUNT_NO == "YOUR_ACCOUNT_NUMBER_HERE":
         print("!!! 중요: 코드 상단의 ACCOUNT_NO 변수에 실제 계좌번호를 입력하세요 !!!")
    else:
        access_token = get_access_token(APP_KEY, APP_SECRET)
        if access_token:
            print("\n----------------------------------------")
            print(f"계좌번호 [{ACCOUNT_NO}] 정보 조회를 시작합니다...")
            account_data = get_cash_balance(access_token, ACCOUNT_NO) # APP_KEY, SECRET은 함수 내부에서 안씀

            if account_data and account_data.get('return_code') == 0:
                print("\n계좌 정보 조회 성공!")
            else:
                print("\n계좌 정보 조회에 실패했습니다.")
        else:
            print("\n토큰 발급에 실패하여 계좌 정보를 조회할 수 없습니다.")
