// 공용 포즈 유틸: MediaPipe 로딩(싱글턴) + 각도 계산
// 주의: 각도 정의는 ml/features.py와 반드시 동일하게 유지할 것
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

export { PoseLandmarker, DrawingUtils };

// MediaPipe Pose 랜드마크 인덱스
export const LM = {
  NOSE: 0,
  L_EAR: 7, R_EAR: 8,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
  L_HEEL: 29, R_HEEL: 30,
  L_FOOT: 31, R_FOOT: 32,
};

let landmarker = null;
let loading = null;

export function getLandmarker() {
  if (landmarker) return Promise.resolve(landmarker);
  if (loading) return loading;
  loading = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    const makeOptions = (delegate) => ({
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    try {
      landmarker = await PoseLandmarker.createFromOptions(fileset, makeOptions("GPU"));
    } catch (e) {
      console.warn("GPU delegate 실패, CPU 전환:", e);
      landmarker = await PoseLandmarker.createFromOptions(fileset, makeOptions("CPU"));
    }
    return landmarker;
  })();
  return loading;
}

// ---------- 각도 유틸 ----------
export function angleAt(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const n1 = Math.hypot(v1.x, v1.y);
  const n2 = Math.hypot(v2.x, v2.y);
  if (n1 === 0 || n2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, (v1.x * v2.x + v1.y * v2.y) / (n1 * n2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

export function trunkLean(shoulder, hip) {
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y; // 화면 좌표: 아래로 갈수록 y 증가
  return Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI);
}

export function footAngle(heel, toe) {
  const dx = Math.abs(toe.x - heel.x);
  const dy = toe.y - heel.y; // 뒤꿈치가 들리면 heel.y < toe.y → dy > 0
  if (dx === 0 && dy === 0) return null;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

export function kneeAnkleRatio(lms) {
  const kneeW = Math.hypot(lms[LM.L_KNEE].x - lms[LM.R_KNEE].x, lms[LM.L_KNEE].y - lms[LM.R_KNEE].y);
  const ankleW = Math.hypot(lms[LM.L_ANKLE].x - lms[LM.R_ANKLE].x, lms[LM.L_ANKLE].y - lms[LM.R_ANKLE].y);
  if (ankleW < 1e-6) return null;
  return kneeW / ankleW;
}

// 카메라에 가까운(가시성 높은) 쪽 관절 선택
export function pickSide(lms) {
  const leftVis = (lms[LM.L_HIP].visibility ?? 0) + (lms[LM.L_KNEE].visibility ?? 0);
  const rightVis = (lms[LM.R_HIP].visibility ?? 0) + (lms[LM.R_KNEE].visibility ?? 0);
  return leftVis >= rightVis
    ? { ear: lms[LM.L_EAR], shoulder: lms[LM.L_SHOULDER], hip: lms[LM.L_HIP], knee: lms[LM.L_KNEE],
        ankle: lms[LM.L_ANKLE], heel: lms[LM.L_HEEL], toe: lms[LM.L_FOOT] }
    : { ear: lms[LM.R_EAR], shoulder: lms[LM.R_SHOULDER], hip: lms[LM.R_HIP], knee: lms[LM.R_KNEE],
        ankle: lms[LM.R_ANKLE], heel: lms[LM.R_HEEL], toe: lms[LM.R_FOOT] };
}

// 스쿼트 판정용 피처 5개 (ml/features.py compute_features와 동일 정의)
export function computeSquatFeatures(lms) {
  const s = pickSide(lms);
  return {
    kneeAngle: angleAt(s.hip, s.knee, s.ankle),
    hipAngle: angleAt(s.shoulder, s.hip, s.knee),
    trunkLean: trunkLean(s.shoulder, s.hip),
    footAngle: footAngle(s.heel, s.toe),
    kneeAnkleRatio: kneeAnkleRatio(lms),
  };
}
