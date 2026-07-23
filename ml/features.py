"""관절 각도 피처 계산.

주의: frontend/app.js의 angleAt/trunkLean/footAngle과 정의가 반드시 동일해야 함.
학습(Python)과 추론(JS)의 피처가 어긋나면 모델이 무용지물이 된다.
"""
import math

# MediaPipe Pose 랜드마크 인덱스
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32  # foot_index(발끝)


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


def foot_angle(heel, toe):
    """뒤꿈치→발끝 선이 수평과 이루는 각도(도).
    바닥에 발이 붙어 있으면 ~0, 뒤꿈치가 들리면(까치발) 커진다."""
    dx = abs(toe[0] - heel[0])
    dy = toe[1] - heel[1]  # 뒤꿈치가 들리면 heel.y < toe.y → dy > 0
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def knee_ankle_ratio(lms):
    """양 무릎 간격 ÷ 양 발목 간격. 무릎이 안쪽으로 모이면(valgus) 1보다 작아진다.
    45° 촬영 기준 피처 — 완전 측면에서는 간격이 0에 가까워져 무의미."""
    knee_w = math.hypot(lms[L_KNEE].x - lms[R_KNEE].x, lms[L_KNEE].y - lms[R_KNEE].y)
    ankle_w = math.hypot(lms[L_ANKLE].x - lms[R_ANKLE].x, lms[L_ANKLE].y - lms[R_ANKLE].y)
    if ankle_w < 1e-6:
        return None
    return knee_w / ankle_w


def pick_side(lms):
    """측면 촬영이므로 가시성 높은 쪽 관절 사용. lms는 MediaPipe landmark 리스트."""
    left_vis = lms[L_HIP].visibility + lms[L_KNEE].visibility
    right_vis = lms[R_HIP].visibility + lms[R_KNEE].visibility
    if left_vis >= right_vis:
        idx = (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT)
    else:
        idx = (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT)
    return [(lms[i].x, lms[i].y) for i in idx]


def compute_features(lms):
    """랜드마크 → 피처 dict. 프레임 단위로 호출."""
    shoulder, hip, knee, ankle, heel, toe = pick_side(lms)
    return {
        "knee_angle": angle_at(hip, knee, ankle),
        "hip_angle": angle_at(shoulder, hip, knee),
        "trunk_lean": trunk_lean(shoulder, hip),
        "foot_angle": foot_angle(heel, toe),
        "knee_ankle_ratio": knee_ankle_ratio(lms),
    }
