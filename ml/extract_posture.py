"""체형 사진 → 학습용 CSV 변환.

사용법:
    python extract_posture.py ../data/posture ../data/posture_features.csv

입력 구조 (폴더명 = 라벨, 측면 서 있는 사진):
    data/posture/normal/    ← 바른 자세
    data/posture/neck/      ← 거북목 (과장해서 촬영)
    data/posture/shoulder/  ← 어깨 말림 (과장해서 촬영)

파일명 규칙은 스쿼트 사진과 동일:
    팀원 촬영 → 이름_번호.jpg (밑줄 앞 이름 = 그룹)
    웹 수집   → 밑줄 없는 고유 이름.jpg (파일 단위 그룹)
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from features import compute_posture_features

VALID_LABELS = {"normal", "neck", "shoulder"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    posture_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    mp_pose = mp.solutions.pose
    rows = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        for label_dir in sorted(posture_dir.iterdir()):
            if not label_dir.is_dir() or label_dir.name not in VALID_LABELS:
                continue
            label = label_dir.name
            count = 0
            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  읽기 실패: {img_path.name}")
                    continue
                result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not result.pose_landmarks:
                    print(f"  포즈 미검출 (제외): {label}/{img_path.name}")
                    continue
                f = compute_posture_features(result.pose_landmarks.landmark)
                if any(v is None for v in f.values()):
                    continue
                rows.append({
                    "source": f"posture/{label}/{img_path.name}",
                    "label": label,
                    "group": img_path.stem.split("_")[0],
                    **{k: round(v, 4) for k, v in f.items()},
                })
                count += 1
            print(f"  {label}: {count}장 추출")

    if not rows:
        print(f"추출된 데이터 없음. 입력 경로 확인: {posture_dir}")
        sys.exit(1)

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "source", "label", "group", "forward_head", "round_shoulder", "trunk_lean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n완료: {len(rows)}행 → {out_csv}")


if __name__ == "__main__":
    main()
