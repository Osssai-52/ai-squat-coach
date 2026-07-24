# BodyBuddy — 체형 분석 기반 맞춤 운동 추천 + 실시간 자세 교정

**앱 흐름**: 신체 정보 입력(선택) → 체형 분석(정면·측면 촬영, 부위 누락 시 안내) →
맞춤 운동 추천(2~3개) → 운동법 확인 → **스쿼트는 실시간 자세 교정 지원** → 기록 탭에서 개선 추이 확인

> **범위 합의 (변경 금지 — 팀 전원 동의 후에만 수정)**
> - 체형 분석: 규칙 기반 (거북목/어깨 말림/상체 기울기/어깨·골반 수평/다리 정렬)
> - 실시간 교정 종목: **스쿼트 1개** — 자체 학습 ML 분류기 (프로젝트의 데이터사이언스 핵심)
> - 스쿼트 촬영: **45° 반측면** 기준 (영상·사진·데모 전부 동일 각도)
> - 분류 클래스 5개: **정상 / 깊이 부족 / 허리 굽음 / 발뒤꿈치 들림 / 무릎 모임** — [docs/class-definitions.md](docs/class-definitions.md)

## 파이프라인

카메라 영상 → MediaPipe 관절 좌표 추출 → 관절 각도 계산(피처) → 자세 분류 → 피드백 출력

## 폴더 구조

| 폴더 | 내용 | 담당 |
|---|---|---|
| `frontend/` | 브라우저 앱 (카메라, 스켈레톤 시각화, 피드백 UI) | A |
| `ml/` | 피처 추출, 모델 학습·평가 (Python) | B |
| `data/` | 촬영 영상·라벨 (git에 올리지 않음, 공유 드라이브 사용) | 전원 |
| `docs/` | 촬영 규칙, 클래스 정의, 발표 자료 | 전원 |

## 실행 방법

### 프론트엔드 (Node 불필요)

```bash
cd frontend
python -m http.server 8000
```

브라우저에서 http://localhost:8000 접속 → 카메라 허용 → 스켈레톤 + 관절 각도 실시간 표시.

### ML 환경 (Python 3.11)

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 데이터 → 모델 → 배포 (2~3일차 루틴)

```bash
cd ml
# 스쿼트 동작 분류
python extract_features.py ../data ../data/features.csv   # 영상+이미지 → 학습 CSV
python train.py ../data/features.csv                      # 학습 + 발표 차트(figures/) + 모델 덤프
copy models\model_rules.json ..\frontend\                 # 프론트에 모델 배포

# 체형 분류 (거북목/어깨 말림)
python extract_posture.py ../data/posture ../data/posture_features.csv
python train_posture.py ../data/posture_features.csv
copy models\posture_model.json ..\frontend\
```

- 모델 JSON이 frontend/에 있으면 자동으로 ML 판정, 없으면 규칙 기반 폴백
- 스쿼트 판정 엔진은 운동 화면 "⚙" 패널, 체형 판정 엔진은 브라우저 콘솔에서 확인

## Google 로그인 설정 (1회, 팀 구글 계정 필요)

코드는 완성돼 있고 **클라이언트 ID만 넣으면 활성화**됩니다. 미설정 상태에서는 개발용 "Continue without account" 버튼이 대신 표시됩니다.

1. [console.cloud.google.com](https://console.cloud.google.com) → 새 프로젝트 생성
2. API 및 서비스 → OAuth 동의 화면 → External → 앱 이름/이메일 입력 → **테스트 사용자에 팀원 전원 + 데모용 계정 추가** (누락 시 로그인 차단)
3. 사용자 인증 정보 → OAuth 클라이언트 ID 만들기 → 유형 "웹 애플리케이션" → 승인된 JavaScript 원본에 `http://localhost:8000` 추가
4. 발급된 `xxxx.apps.googleusercontent.com`을 [frontend/auth.js](frontend/auth.js)의 `GOOGLE_CLIENT_ID`에 붙여넣기

⚠ 데모 주의: 인터넷 연결 필요. 데모 기기 주소가 localhost:8000이 아니면 그 주소도 원본에 추가할 것. 로그인이 말썽이면 auth.js의 ID를 잠시 비우면 우회 버튼으로 데모 가능.

## 기술 스택

- **포즈 추정**: MediaPipe Pose Landmarker (브라우저: tasks-vision JS / 학습용: Python)
- **피처**: 무릎 각도, 고관절 각도, 허리 기울기 (좌표 원본 대신 각도 → 키·카메라 거리에 무관)
- **분류 모델**: 규칙 기반 베이스라인 → Random Forest/XGBoost (Python 학습 후 JSON 규칙 덤프로 JS 추론)

## 5일 일정

- 1일차: MediaPipe 연동, 실시간 관절 추출 + 각도 계산 ✅(스캐폴딩 완료), 데이터 촬영 시작
- 2일차: 라벨링 + 학습 CSV 생성, 규칙 기반 베이스라인, 모델 학습·평가
- 3일차: 모델 실시간 연결, 렙 카운팅, 음성 피드백, 리포트 — **이후 기능 동결**
- 4일차: 실전 환경 테스트, 백업 영상 녹화, 발표 자료
- 5일차: 통합 테스트, 데모 리허설
