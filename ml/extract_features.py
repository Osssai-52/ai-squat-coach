"""촬영 영상 → 학습용 CSV 변환 (2일차 오전 작업의 핵심 도구).

사용법:
    python extract_features.py ../data/videos ../data/features.csv

파일명 규칙에서 라벨을 자동 추출한다: 이름_클래스_회차.mp4
예: minsu_good_01.mp4, jiyeon_depth_03.mp4
클래스: good / knee / back / depth
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from features import compute_features

VALID_LABELS = {"good", "knee", "back", "depth"}


def label_from_filename(path: Path) -> str | None:
    parts = path.stem.split("_")
    for p in parts:
        if p in VALID_LABELS:
            return p
    return None


def process_video(path: Path, pose, writer, label: str):
    cap = cv2.VideoCapture(str(path))
    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            continue
        f = compute_features(result.pose_landmarks.landmark)
        if f["knee_angle"] is None:
            continue
        writer.writerow({
            "video": path.name,
            "frame": frame_idx,
            "label": label,
            **{k: round(v, 2) for k, v in f.items()},
        })
        kept += 1
    cap.release()
    print(f"  {path.name}: {kept}/{frame_idx} 프레임 추출 (label={label})")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    video_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}
    )
    if not videos:
        print(f"영상 없음: {video_dir}")
        sys.exit(1)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose, \
         open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["video", "frame", "label", "knee_angle", "hip_angle", "trunk_lean"]
        )
        writer.writeheader()
        for v in videos:
            label = label_from_filename(v)
            if label is None:
                print(f"  건너뜀 (파일명에 라벨 없음): {v.name}")
                continue
            process_video(v, pose, writer, label)

    print(f"\n완료 → {out_csv}")


if __name__ == "__main__":
    main()
