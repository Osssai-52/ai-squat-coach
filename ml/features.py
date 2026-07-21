"""관절 각도 피처 계산.

주의: frontend/app.js의 angleAt/trunkLean과 정의가 반드시 동일해야 함.
학습(Python)과 추론(JS)의 피처가 어긋나면 모델이 무용지물이 된다.
"""
import math

# MediaPipe Pose 랜드마크 인덱스
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def angle_at(a, b, c):
    """세 점 a-b-c에서 b를 꼭짓점으로 하는 각도(도). 점은 (x, y) 튜플."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def trunk_lean(shoulder, hip):
    """어깨-엉덩이 선과 수직선 사이 각도(도). 0 = 꼿꼿이 선 상태."""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]  # 이미지 좌표: 아래로 갈수록 y 증가
    return abs(math.degrees(math.atan2(dx, -dy)))


def pick_side(lms):
    """측면 촬영이므로 가시성 높은 쪽 관절 사용. lms는 MediaPipe landmark 리스트."""
    left_vis = lms[L_HIP].visibility + lms[L_KNEE].visibility
    right_vis = lms[R_HIP].visibility + lms[R_KNEE].visibility
    if left_vis >= right_vis:
        idx = (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE)
    else:
        idx = (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE)
    return [(lms[i].x, lms[i].y) for i in idx]


def compute_features(lms):
    """랜드마크 → 피처 dict. 프레임 단위로 호출."""
    shoulder, hip, knee, ankle = pick_side(lms)
    return {
        "knee_angle": angle_at(hip, knee, ankle),
        "hip_angle": angle_at(shoulder, hip, knee),
        "trunk_lean": trunk_lean(shoulder, hip),
    }
