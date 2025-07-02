# -*- coding: utf-8 -*-
# 파일명: get_token.py (예시)
# 실행 환경: 64비트 Python (myenv)

import requests
import json

# ★★★ 중요: 키움증권에서 발급받은 실제 값으로 변경하세요 ★★★
APP_KEY = "LE1PsyDirouZ8B1OxnNGrUImw-_eXshzhDwwCcweKss"  # 앱키 (API 신청 후 발급)
APP_SECRET = "e1VMIr_CKOiBfiNgG_kKjuT7C3GQUQHnmm993AgKYNw" # 시크릿키 (API 신청 후 발급)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# 접근 토큰 발급 URL
# 모의투자 시: 
TOKEN_URL =  "https://mockapi.kiwoom.com/oauth2/token"
# 실전투자 시:
#TOKEN_URL = "https://api.kiwoom.com/oauth2/token"

def get_access_token(app_key, app_secret):
    """ OAuth 2.0 접근 토큰 발급 요청 """
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "secretkey": app_secret
    }
    try:
        print(f"접근 토큰 발급 요청 시도 -> {TOKEN_URL}")
        res = requests.post(TOKEN_URL, headers=headers, data=json.dumps(body))
        res.raise_for_status() # 오류 발생 시 예외 발생

        token_data = res.json() # 응답 JSON 파싱

        # ★★★ 수정된 부분 ★★★
        if token_data.get('token'): # 'access_token' 대신 'token' 키 확인
            print("접근 토큰 발급 성공!")
            # 토큰 정보 출력
            print(f"  - Access Token: {token_data['token']}") # 일부만 출력
            print(f"  - Expires In: {token_data.get('expires_in')} 초") # expires_in 키도 확인 (응답에 있다면)
            print(f"  - Expires Datetime: {token_data.get('expires_dt')}") # 만료 일시
            print(f"  - Token Type: {token_data.get('token_type')}")
            return token_data['token'] # 'token' 키에서 값을 반환
        # ★★★★★★★★★★★★★
        else:
            # 실패 처리 (예: return_code가 0이 아니거나 token 키가 없는 경우)
            print("접근 토큰 발급 실패:")
            print(f"  - Return Code: {token_data.get('return_code')}")
            print(f"  - Return Msg: {token_data.get('return_msg')}")
            print(f"  - Full Response: {token_data}")
            return None

    except requests.exceptions.RequestException as e:
        # ... (기존 오류 처리 동일) ...
        print(f"HTTP 요청 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
             try: print(f"오류 응답 내용: {e.response.json()}")
             except json.JSONDecodeError: print(f"오류 응답 내용 (텍스트): {e.response.text}")
        return None
    except Exception as e:
        print(f"토큰 발급 중 예외 발생: {e}")
        return None

if __name__ == "__main__":
    # APP_KEY, APP_SECRET이 설정되었는지 확인
    if APP_KEY == "YOUR_APP_KEY" or APP_SECRET == "YOUR_APP_SECRET":
        print("!!! 중요: 코드 상단의 APP_KEY 와 APP_SECRET 변수에 실제 발급받은 키를 입력하세요 !!!")
    else:
        access_token = get_access_token(APP_KEY, APP_SECRET)

        if access_token:
            print("\n토큰 발급 성공! 이 토큰을 사용하여 다른 API를 호출할 수 있습니다.")
            # 이 토큰은 유효 시간(expires_in) 동안만 사용할 수 있습니다.
            # 보통 만료 전에 재발급 받거나, 만료 시 다시 발급받아야 합니다.
        else:
            print("\n토큰 발급에 실패했습니다. 앱키/시크릿키 및 API 신청 상태를 확인하세요.")