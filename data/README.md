# data/

영상·이미지·라벨 데이터 폴더. **git에 올라가지 않음** (.gitignore 처리).

## 폴더 구조

```
data/
├── videos/                  ← 자체 촬영 영상 (파일명 규칙: 이름_클래스_회차.mp4)
│   ├── minsu_good_01.mp4
│   └── jiyeon_depth_03.mp4
├── images/                  ← 인터넷 수집 이미지 (폴더명 = 라벨)
│   ├── good/
│   ├── depth/
│   ├── back/
│   └── knee/
└── features.csv             ← 아래 명령으로 생성되는 학습용 피처
```

## 학습용 CSV 생성

```bash
cd ml
python extract_features.py ../data ../data/features.csv
```

- 영상: 렙을 자동 분리해 **최저점 근처 프레임만** 추출 (라벨 오염 방지)
- 이미지: 스쿼트 최저점 자세 사진만 넣을 것 (서 있는 사진 금지 — 전량 학습에 들어감)
- 포즈 미검출 이미지는 자동 제외되고 로그에 표시됨

## 수집 기준 요약

- 측면 뷰(±20°), 전신(발목까지), 1인만 나온 사진
- 출처는 무료 라이선스(Pexels/Unsplash/Pixabay/Kaggle) 우선, 출처 목록 기록
- 촬영 규칙은 [docs/recording-rules.md](../docs/recording-rules.md) 참고
