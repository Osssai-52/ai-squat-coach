"""촬영 영상 + 수집 이미지 → 학습용 CSV 변환.

사용법:
    python extract_features.py ../data ../data/features.csv

입력 구조:
    data/videos/이름_클래스_회차.mp4   ← 파일명에서 라벨 자동 추출 (good/knee/back/depth)
    data/images/good/*.jpg            ← 폴더명이 라벨
    data/images/depth/*.jpg           ← (jpg/jpeg/png/webp)
    ...

영상은 렙을 자동 분리해서 **최저점 근처 프레임만** 추출한다.
(서 있는/이동 중 프레임을 오류 클래스로 라벨링하는 오염 방지 —
 판정도 최저점에서만 하므로 학습·추론 조건이 일치)

CSV의 group 컬럼(영상=사람 이름, 이미지=파일명)은 train.py에서
같은 사람/출처가 train과 test에 섞이지 않게 분할하는 데 쓰인다.
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from features import compute_features

VALID_LABELS = {"good", "knee", "back", "depth", "heel"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 렙 분리 / 최저점 필터 (frontend CONFIG와 논리 일치 유지)
REP_ENTER_KNEE = 150   # 무릎 각도가 이 아래로 내려가면 렙 시작
REP_EXIT_KNEE = 155    # 다시 이 위로 올라오면 렙 종료
MIN_REP_DEPTH = 140    # 렙 최저 무릎 각도가 이보다 크면 렙으로 안 침 (잡음 제거)
BOTTOM_TOLERANCE = 15  # 최저점 ± 이 각도 이내 프레임만 학습에 사용


def label_from_filename(path: Path) -> str | None:
    for p in path.stem.split("_"):
        if p in VALID_LABELS:
            return p
    return None


def person_from_filename(path: Path) -> str:
    """이름_클래스_회차.mp4 → '이름'. 같은 사람이 train/test에 안 섞이게 하는 그룹 키."""
    return path.stem.split("_")[0]


def extract_video_frames(path: Path, pose):
    """영상의 전 프레임 피처를 뽑은 뒤 렙별 최저점 근처 프레임만 반환."""
    cap = cv2.VideoCapture(str(path))
    frames = []  # (frame_idx, features)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            continue
        f = compute_features(result.pose_landmarks.landmark)
        if any(v is None for v in f.values()):
            continue
        frames.append((idx, f))
    cap.release()

    # 렙 분리 → 최저점 ± BOTTOM_TOLERANCE 프레임만 채택
    kept = []
    rep = []       # 현재 렙에 속한 (idx, features)
    in_rep = False
    reps = 0

    def flush_rep(rep_frames):
        nonlocal reps
        if not rep_frames:
            return
        rep_min = min(f["knee_angle"] for _, f in rep_frames)
        if rep_min > MIN_REP_DEPTH:
            return  # 충분히 앉지 않은 구간은 렙이 아님 (카메라 앞 이동 등 잡음)
        reps += 1
        kept.extend((i, f) for i, f in rep_frames
                    if f["knee_angle"] <= rep_min + BOTTOM_TOLERANCE)

    for i, f in frames:
        k = f["knee_angle"]
        if not in_rep and k < REP_ENTER_KNEE:
            in_rep = True
            rep = [(i, f)]
        elif in_rep:
            if k >= REP_EXIT_KNEE:
                flush_rep(rep)
                in_rep = False
                rep = []
            else:
                rep.append((i, f))
    flush_rep(rep)  # 영상이 렙 도중에 끝난 경우

    return kept, len(frames), reps


def row(source, frame, label, group, f):
    return {
        "source": source, "frame": frame, "label": label, "group": group,
        **{k: round(v, 2) for k, v in f.items()},
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    data_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    video_dir = data_dir / "videos"
    image_dir = data_dir / "images"

    mp_pose = mp.solutions.pose
    rows = []

    # --- 영상 (동적 모드) ---
    videos = sorted(p for p in video_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS) if video_dir.is_dir() else []
    if videos:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            for v in videos:
                label = label_from_filename(v)
                if label is None:
                    print(f"  건너뜀 (파일명에 라벨 없음): {v.name}")
                    continue
                kept, total, reps = extract_video_frames(v, pose)
                rows.extend(row(v.name, i, label, person_from_filename(v), f)
                            for i, f in kept)
                print(f"  {v.name}: {reps}렙 감지, 최저점 프레임 {len(kept)}/{total} 추출 (label={label})")

    # --- 이미지 (정지 모드, 폴더명 = 라벨) ---
    image_count = 0
    if image_dir.is_dir():
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            for label_dir in sorted(image_dir.iterdir()):
                if not label_dir.is_dir() or label_dir.name not in VALID_LABELS:
                    continue
                label = label_dir.name
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
                    f = compute_features(result.pose_landmarks.landmark)
                    if any(v is None for v in f.values()):
                        continue
                    src = f"images/{label}/{img_path.name}"
                    # 그룹 = 파일명의 밑줄 앞부분: "minsu_01.jpg" → minsu (사람 단위),
                    # 밑줄이 없으면 파일명 전체 → 파일 단위 (웹 수집 사진)
                    rows.append(row(src, 0, label, img_path.stem.split("_")[0], f))
                    image_count += 1
        print(f"  이미지 {image_count}장 추출")

    if not rows:
        print(f"추출된 데이터 없음. 입력 경로 확인: {video_dir}, {image_dir}")
        sys.exit(1)

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "source", "frame", "label", "group",
            "knee_angle", "hip_angle", "trunk_lean", "foot_angle", "knee_ankle_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n완료: {len(rows)}행 → {out_csv}")


if __name__ == "__main__":
    main()
