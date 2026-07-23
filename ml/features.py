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
L_HEEL, R_HEEL = 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32


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


def knee_valgus_ratio(lms):
    """무릎 사이 거리 / 발목 사이 거리 비율. 무릎이 발목보다 안쪽으로 모이면 1보다 작아짐.
    좌우 랜드마크가 둘 다 필요해서 pick_side()와 무관하게 전체 lms에서 직접 계산 (45도 촬영 전제)."""
    knee_dist = abs(lms[L_KNEE].x - lms[R_KNEE].x)
    ankle_dist = abs(lms[L_ANKLE].x - lms[R_ANKLE].x)
    if ankle_dist == 0:
        return None
    return knee_dist / ankle_dist


def heel_lift_ratio(heel, toe):
    """발끝-발뒤꿈치 거리로 정규화한 뒤꿈치 들림 정도. 값이 클수록 뒤꿈치가 지면에서 뜬 것.
    heel, toe는 (x, y) 튜플."""
    foot_len = math.hypot(toe[0] - heel[0], toe[1] - heel[1])
    if foot_len == 0:
        return None
    return (toe[1] - heel[1]) / foot_len  # 이미지 좌표: 아래로 갈수록 y 증가


def pick_side(lms):
    """45도 대각선 촬영이므로 가시성 높은 쪽 관절로 시상면 각도 계산. lms는 MediaPipe landmark 리스트."""
    left_vis = lms[L_HIP].visibility + lms[L_KNEE].visibility
    right_vis = lms[R_HIP].visibility + lms[R_KNEE].visibility
    if left_vis >= right_vis:
        idx = (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT_INDEX)
    else:
        idx = (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT_INDEX)
    return [(lms[i].x, lms[i].y) for i in idx]


def compute_features(lms):
    """랜드마크 → 피처 dict. 프레임(또는 정지 이미지) 단위로 호출."""
    shoulder, hip, knee, ankle, heel, toe = pick_side(lms)
    return {
        "knee_angle": angle_at(hip, knee, ankle),
        "hip_angle": angle_at(shoulder, hip, knee),
        "trunk_lean": trunk_lean(shoulder, hip),
        "knee_valgus_ratio": knee_valgus_ratio(lms),
        "heel_lift_ratio": heel_lift_ratio(heel, toe),
    }
