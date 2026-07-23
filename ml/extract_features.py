"""촬영 사진(최저점 프레임 1장씩) → 학습용 CSV 변환 (2일차 오전 작업의 핵심 도구).

사용법:
    python extract_features.py ../data/photos ../data/features.csv

각 사진은 이미 한 렙의 최저점 순간 1장이라고 가정 (프레임 추출 불필요).
파일명 규칙에서 라벨을 자동 추출한다: 이름_클래스_회차.jpg
예: minsu_good_01.jpg, jiyeon_depth_03.jpg
클래스: good / knee / back / depth / heel
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from features import compute_features

VALID_LABELS = {"good", "knee", "back", "depth", "heel"}


def label_from_filename(path: Path) -> str | None:
    parts = path.stem.split("_")
    for p in parts:
        if p in VALID_LABELS:
            return p
    return None


def process_photo(path: Path, pose, writer, label: str) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  건너뜀 (읽기 실패): {path.name}")
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        print(f"  건너뜀 (사람 인식 실패): {path.name}")
        return False
    f = compute_features(result.pose_landmarks.landmark)
    if f["knee_angle"] is None or f["knee_valgus_ratio"] is None or f["heel_lift_ratio"] is None:
        print(f"  건너뜀 (일부 관절 미검출): {path.name}")
        return False
    writer.writerow({
        "photo": path.name,
        "label": label,
        **{k: round(v, 2) for k, v in f.items()},
    })
    return True


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    photo_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    photos = sorted(
        p for p in photo_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not photos:
        print(f"사진 없음: {photo_dir}")
        sys.exit(1)

    mp_pose = mp.solutions.pose
    kept = 0
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose, \
         open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["photo", "label", "knee_angle", "hip_angle", "trunk_lean",
                            "knee_valgus_ratio", "heel_lift_ratio"]
        )
        writer.writeheader()
        for p in photos:
            label = label_from_filename(p)
            if label is None:
                print(f"  건너뜀 (파일명에 라벨 없음): {p.name}")
                continue
            if process_photo(p, pose, writer, label):
                kept += 1

    print(f"\n완료: {kept}/{len(photos)}장 추출 → {out_csv}")


if __name__ == "__main__":
    main()
