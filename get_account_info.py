# -*- coding: utf-8 -*-
# 파일명: get_account_info.py (예시)
# 실행 환경: 64비트 Python (myenv)

import requests
import json
import pprint

# ★★★ 설정값 ★★★
# 이전 단계에서 발급받은 실제 Access Token 전체를 복사하여 붙여넣으세요.
ACCESS_TOKEN = "DGeBa_V3fx..." # 예: "DGeBa_V3fxK..."
# 조회할 계좌번호 ('-' 제외하고 입력)
ACCOUNT_NO = 81026247 
# 앱키/시크릿키 (API 가이드에 따라 헤더에 추가 필요 시 사용)
APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss"  # 앱키 (API 신청 후 발급)
APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw" # 시크릿키 (API 신청 후 발급)
# ★★★★★★★★★★★

# API 엔드포인트 URL (계좌 관련)
# 모의투자 시:
API_URL = "https://mockapi.kiwoom.com/api/dostk/acnt"
# 실전투자 시:
# API_URL = "https://api.kiwoom.com/api/dostk/acnt"

# TR ID (예수금 상세 현황 요청)
TR_ID = "kt00001"

def get_cash_balance(token, acc_no, app_key, app_secret):
    """ 예수금 상세 현황 조회 (kt00001) """
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {token}", # 발급받은 접근 토큰 사용
        "api-id": TR_ID
        # API 가이드에 따라 appkey, appsecret 헤더가 필요할 수 있음
        # "appkey": app_key,
        # "appsecret": app_secret
    }
    # body: kt00001 TR의 정확한 입력값을 API 가이드에서 확인 필요!
    # 여기서는 계좌번호만 필요하다고 가정함
    body = {
        "acnt_no": acc_no # 키 이름이 다를 수 있음 (예: "계좌번호")
        # 비밀번호 등이 필요할 수 있음: "password": "YOUR_ACC_PASSWORD"
    }

    try:
        print(f"{TR_ID} 요청 시도 -> {API_URL}")
        res = requests.post(API_URL, headers=headers, data=json.dumps(body))
        res.raise_for_status() # HTTP 오류 발생 시 예외

        response_data = res.json() # 응답 JSON 파싱

        print(f"\n--- {TR_ID} 응답 결과 ---")
        print(f"Return Code: {response_data.get('return_code')}")
        print(f"Return Msg: {response_data.get('return_msg')}")

        if response_data.get('return_code') == 0:
            print("\n[ 예수금 상세 현황 ]")
            # 응답 Body의 정확한 키 이름은 API 가이드 확인 필요
            # 일반적인 키 이름 예시:
            deposit = response_data.get('output', {}).get('dpsa_amt') # 예수금 (d+2)
            available_cash = response_data.get('output', {}).get('ord_psbl_amt') # 주문가능금액
            print(f"  - 예수금 (d+2): {deposit}")
            print(f"  - 주문가능금액: {available_cash}")
            print("\n(전체 응답 내용):")
            pprint.pprint(response_data) # 전체 응답 출력
            return response_data # 성공 시 전체 데이터 반환
        else:
            print("API 호출 실패.")
            pprint.pprint(response_data)
            return None

    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
             try: print(f"오류 응답 내용: {e.response.json()}")
             except json.JSONDecodeError: print(f"오류 응답 내용 (텍스트): {e.response.text}")
        return None
    except Exception as e:
        print(f"API 호출 중 예외 발생: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if ACCESS_TOKEN == "YOUR_ACCESS_TOKEN_HERE" or ACCOUNT_NO == "YOUR_ACCOUNT_NUMBER_HERE":
        print("!!! 중요: 코드 상단의 ACCESS_TOKEN 과 ACCOUNT_NO 변수에 실제 값을 입력하세요 !!!")
    else:
        print(f"계좌번호 [{ACCOUNT_NO}]의 예수금 상세 현황 조회를 시작합니다...")
        account_data = get_cash_balance(ACCESS_TOKEN, ACCOUNT_NO, APP_KEY, APP_SECRET)

        if account_data and account_data.get('return_code') == 0:
            print("\n계좌 정보 조회 성공!")
        else:
            print("\n계좌 정보 조회에 실패했습니다.")