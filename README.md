# AI 실시간 운동 자세 교정 서비스 (스쿼트)

> **범위 합의 (변경 금지 — 팀 전원 동의 후에만 수정)**
> - 종목: **스쿼트 1개**
> - 촬영: **45° 반측면** 기준 (영상·사진·데모 전부 동일 각도 — 수집 데이터가 45°라서 통일)
> - v1 분류 클래스 4개: **정상 / 깊이 부족 / 허리 굽음 / 발뒤꿈치 들림**
> - 무릎 모임(knee)은 확장 과제 — [docs/class-definitions.md](docs/class-definitions.md) 참고

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
python extract_features.py ../data ../data/features.csv   # 영상+이미지 → 학습 CSV
python train.py ../data/features.csv                      # 학습 + 발표 차트(figures/) + 모델 덤프
copy models\model_rules.json ..\frontend\                 # 프론트에 모델 배포
```

- `frontend/model_rules.json`이 있으면 앱이 자동으로 ML 판정으로 전환 (없으면 규칙 기반 v0)
- 현재 판정 엔진은 운동 화면의 "디버그 ⚙" 패널에서 확인 가능

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
