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

## 실행 방법

### 프론트엔드 (Node 불필요)

```bash
cd frontend
python serve.py
```

(`python -m http.server`도 되지만 브라우저가 JS를 캐시해서 수정이 반영 안 될 수 있음 — serve.py는 캐시를 끔)

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

## 기술 스택

- **포즈 추정**: MediaPipe Pose Landmarker (브라우저: tasks-vision JS / 학습용: Python)
- **피처**: 무릎 각도, 고관절 각도, 허리 기울기 (좌표 원본 대신 각도 → 키·카메라 거리에 무관)
- **분류 모델**: 규칙 기반 베이스라인 → Random Forest/XGBoost (Python 학습 후 JSON 규칙 덤프로 JS 추론)
