// 체형 분석: 촬영 프레임 검증 → 지표 계산 → 판정(ML 우선, 규칙 폴백) → 운동 추천
import { LM, trunkLean, kneeAnkleRatio } from "./pose.js";
import { loadPostureModel, predictPosture } from "./inference.js";

// 체형 ML 모델 (ml/train_posture.py가 덤프한 posture_model.json — 없으면 규칙 기반)
let postureML = null;
export async function initPostureML() {
  postureML = await loadPostureModel("posture_model.json");
  return !!postureML;
}
export function isPostureML() { return !!postureML; }

// ---------- 촬영 프레임 검증 (부위 누락 안내) ----------
const FRAME_MARGIN = 0.02;

function landmarkVisible(p) {
  const inFrame =
    p.x > FRAME_MARGIN && p.x < 1 - FRAME_MARGIN &&
    p.y > FRAME_MARGIN && p.y < 1 - FRAME_MARGIN;
  const vis = p.visibility;
  return inFrame && (vis == null || vis > 0.3);
}

const PART_CHECKS = [
  { ids: [LM.NOSE, LM.L_EAR, LM.R_EAR], any: true, msg: "머리까지 화면에 나오게 해주세요" },
  { ids: [LM.L_SHOULDER, LM.R_SHOULDER], any: false, msg: "어깨가 가려지지 않게 서 주세요" },
  { ids: [LM.L_HIP, LM.R_HIP], any: false, msg: "몸 전체가 화면에 들어오게 해주세요" },
  { ids: [LM.L_KNEE, LM.R_KNEE], any: false, msg: "무릎까지 나오도록 뒤로 물러나 주세요" },
  { ids: [LM.L_ANKLE, LM.R_ANKLE, LM.L_FOOT, LM.R_FOOT], any: false, msg: "발까지 전신이 나오도록 뒤로 물러나 주세요" },
];

// 반환: { ok: true } 또는 { ok: false, msg: "안내 문구" }
export function validateFrame(lms) {
  if (!lms) return { ok: false, msg: "사람이 인식되지 않아요. 밝은 곳에서 전신이 보이게 서 주세요" };
  for (const check of PART_CHECKS) {
    const results = check.ids.map((i) => landmarkVisible(lms[i]));
    const pass = check.any ? results.some(Boolean) : results.every(Boolean);
    if (!pass) return { ok: false, msg: check.msg };
  }
  return { ok: true };
}

// ---------- 체형 지표 계산 ----------
// 임계값은 초기 추정치 — 팀원 실측(정상 자세/과장 자세)으로 튜닝 필요
const T = {
  FORWARD_HEAD: 0.12,   // 귀-어깨 수평 오프셋 / 몸통 길이
  ROUND_SHOULDER: 0.10, // 어깨-엉덩이 수평 오프셋 / 몸통 길이
  TRUNK_LEAN: 8,        // 상체 기울기(도)
  TILT: 0.045,          // 좌우 높이차 / 어깨 너비
  KNEE_IN: 0.75,        // 무릎/발목 간격비 하한 (X다리 경향)
  KNEE_OUT: 1.45,       // 무릎/발목 간격비 상한 (O다리 경향)
};

function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// side: 측면 사진 랜드마크 / front: 정면 사진 랜드마크
// 반환: 항목 리스트 [{key, name, value, unit, status: "ok"|"warn", desc}]
export function analyzePosture({ front, side }) {
  const items = [];

  if (side) {
    const sh = mid(side[LM.L_SHOULDER], side[LM.R_SHOULDER]);
    const hip = mid(side[LM.L_HIP], side[LM.R_HIP]);
    const ear = mid(side[LM.L_EAR], side[LM.R_EAR]);
    const torso = dist(sh, hip);

    if (torso > 1e-6) {
      const fh = Math.abs(ear.x - sh.x) / torso;
      const rs = Math.abs(sh.x - hip.x) / torso;
      const lean = trunkLean(sh, hip);

      // ML 모델이 있으면 분류 결과로 판정, 없으면 임계값 규칙
      let neckWarn = fh > T.FORWARD_HEAD;
      let shoulderWarn = rs > T.ROUND_SHOULDER;
      if (postureML) {
        const label = predictPosture(postureML, { forwardHead: fh, roundShoulder: rs, trunkLean: lean });
        if (label) {
          neckWarn = label === "neck";
          shoulderWarn = label === "shoulder";
        }
      }

      items.push({
        key: "forwardHead", name: "목 정렬", value: (fh * 100).toFixed(0), unit: "%",
        status: neckWarn ? "warn" : "ok",
        desc: neckWarn
          ? "머리가 어깨보다 앞으로 나와 있어요 (거북목 경향)"
          : "귀와 어깨가 잘 정렬되어 있어요",
      });
      items.push({
        key: "roundShoulder", name: "어깨 정렬", value: (rs * 100).toFixed(0), unit: "%",
        status: shoulderWarn ? "warn" : "ok",
        desc: shoulderWarn
          ? "어깨가 앞으로 말려 있어요 (라운드 숄더 경향)"
          : "어깨가 골반 위에 잘 놓여 있어요",
      });
    }

    const lean = trunkLean(sh, hip);
    items.push({
      key: "trunk", name: "상체 기울기", value: lean.toFixed(0), unit: "°",
      status: lean > T.TRUNK_LEAN ? "warn" : "ok",
      desc: lean > T.TRUNK_LEAN
        ? "상체가 앞뒤로 기울어 있어요 (척추 정렬 주의)"
        : "상체가 곧게 서 있어요",
    });
  }

  if (front) {
    const shoulderW = dist(front[LM.L_SHOULDER], front[LM.R_SHOULDER]);
    if (shoulderW > 1e-6) {
      const st = Math.abs(front[LM.L_SHOULDER].y - front[LM.R_SHOULDER].y) / shoulderW;
      items.push({
        key: "shoulderTilt", name: "어깨 수평", value: (st * 100).toFixed(1), unit: "%",
        status: st > T.TILT ? "warn" : "ok",
        desc: st > T.TILT ? "좌우 어깨 높이에 차이가 있어요" : "어깨 높이가 수평에 가까워요",
      });
    }
    const hipW = dist(front[LM.L_HIP], front[LM.R_HIP]);
    if (hipW > 1e-6) {
      const pt = Math.abs(front[LM.L_HIP].y - front[LM.R_HIP].y) / hipW;
      items.push({
        key: "pelvisTilt", name: "골반 수평", value: (pt * 100).toFixed(1), unit: "%",
        status: pt > T.TILT ? "warn" : "ok",
        desc: pt > T.TILT ? "좌우 골반 높이에 차이가 있어요" : "골반이 수평에 가까워요",
      });
    }
    const ratio = kneeAnkleRatio(front);
    if (ratio != null) {
      const warn = ratio < T.KNEE_IN || ratio > T.KNEE_OUT;
      items.push({
        key: "legAlign", name: "다리 정렬", value: ratio.toFixed(2), unit: "",
        status: warn ? "warn" : "ok",
        desc: ratio < T.KNEE_IN ? "무릎이 안쪽으로 모이는 경향이 있어요"
          : ratio > T.KNEE_OUT ? "무릎이 바깥으로 벌어지는 경향이 있어요"
          : "무릎과 발목 정렬이 좋아요",
      });
    }
  }

  return items;
}

// ---------- 운동 라이브러리 ----------
export const EXERCISES = {
  squat: {
    name: "스쿼트", tag: "하체 · 전신 근력", live: true,
    summary: "하체와 코어를 한 번에 강화하는 기본 운동. 실시간 자세 분석을 지원해요.",
    steps: [
      "발을 어깨너비로 벌리고 발끝을 살짝 바깥으로 둡니다",
      "가슴을 펴고 시선은 정면을 유지합니다",
      "엉덩이를 뒤로 빼며 허벅지가 수평이 될 때까지 앉습니다",
      "무릎이 발끝 방향과 같은 방향을 향하게 유지합니다",
      "발뒤꿈치로 바닥을 밀며 일어섭니다",
    ],
    dose: "10회 × 3세트, 세트 사이 60초 휴식",
  },
  plank: {
    name: "플랭크", tag: "코어 안정화",
    summary: "척추를 곧게 유지하는 힘을 기르는 정적 코어 운동.",
    steps: [
      "팔꿈치를 어깨 아래에 두고 엎드립니다",
      "머리부터 발끝까지 일직선을 만듭니다",
      "배에 힘을 주고 허리가 꺼지지 않게 버팁니다",
      "호흡을 멈추지 않습니다",
    ],
    dose: "30초 × 3세트부터 시작, 점차 늘리기",
  },
  bandPullApart: {
    name: "밴드 풀어파트", tag: "등 상부 · 어깨",
    summary: "말린 어깨를 뒤로 되돌리는 등 상부 강화 운동.",
    steps: [
      "밴드를 어깨너비로 잡고 팔을 앞으로 뻗습니다",
      "팔꿈치를 편 채 밴드를 가슴 높이에서 양옆으로 당깁니다",
      "어깨뼈를 뒤로 모은다는 느낌으로 2초 유지",
      "천천히 돌아옵니다",
    ],
    dose: "15회 × 3세트",
  },
  wallAngel: {
    name: "벽 천사", tag: "어깨 가동성 · 자세",
    summary: "벽에 등을 대고 팔을 올렸다 내리며 어깨·등 정렬을 회복하는 운동.",
    steps: [
      "벽에 뒤통수·등·엉덩이를 붙이고 섭니다",
      "팔꿈치를 90도로 굽혀 벽에 붙입니다",
      "벽에서 떨어지지 않게 팔을 천천히 위로 올립니다",
      "천천히 내리며 반복합니다",
    ],
    dose: "10회 × 3세트",
  },
  hipBridge: {
    name: "힙 브릿지", tag: "둔근 · 골반 안정화",
    summary: "골반을 안정시키고 둔근을 깨우는 기초 운동.",
    steps: [
      "누워서 무릎을 세우고 발을 골반너비로 둡니다",
      "발뒤꿈치로 바닥을 누르며 엉덩이를 들어 올립니다",
      "어깨-골반-무릎이 일직선이 되면 2초 유지",
      "천천히 내립니다",
    ],
    dose: "12회 × 3세트",
  },
  clamshell: {
    name: "클램쉘", tag: "둔근 · 무릎 정렬",
    summary: "무릎이 안쪽으로 모이는 습관을 잡아주는 엉덩이 옆 근육 운동.",
    steps: [
      "옆으로 누워 무릎을 45도로 굽힙니다",
      "발뒤꿈치를 붙인 채 위쪽 무릎을 천장으로 엽니다",
      "골반이 뒤로 돌아가지 않게 고정합니다",
      "천천히 닫습니다",
    ],
    dose: "한쪽 15회 × 3세트",
  },
  sidePlank: {
    name: "사이드 플랭크", tag: "옆구리 · 좌우 균형",
    summary: "좌우 비대칭을 잡아주는 옆면 코어 운동.",
    steps: [
      "옆으로 누워 팔꿈치를 어깨 아래에 둡니다",
      "골반을 들어 머리-골반-발이 일직선이 되게 합니다",
      "약한 쪽을 한 세트 더 합니다",
    ],
    dose: "한쪽 20초 × 3세트",
  },
};

// 체형 분석 결과 → 추천 운동 2~3개
// 스쿼트는 실시간 교정 지원 운동이라 항상 포함 (자리 보장), 나머지는 주의 항목 기준 최대 2개
export function buildRecommendations(items) {
  const warn = new Set(items.filter((i) => i.status === "warn").map((i) => i.key));
  const others = [];
  const add = (key, reason) => {
    if (others.length < 2 && !others.some((p) => p.key === key)) others.push({ key, reason });
  };

  if (warn.has("legAlign")) add("clamshell", "무릎 정렬 습관을 잡아줘요");
  if (warn.has("forwardHead") || warn.has("roundShoulder")) {
    add("bandPullApart", "말린 어깨를 뒤로 되돌리는 데 효과적이에요");
    add("wallAngel", "어깨·목 정렬 회복에 도움이 돼요");
  }
  if (warn.has("trunk")) add("plank", "척추를 곧게 유지하는 코어 힘을 길러줘요");
  if (warn.has("shoulderTilt") || warn.has("pelvisTilt")) {
    add("sidePlank", "좌우 균형을 잡는 데 도움이 돼요");
    add("hipBridge", "골반 안정화에 좋아요");
  }
  if (others.length === 0) add("plank", "코어의 기본 운동이에요");

  const squat = {
    key: "squat",
    reason: warn.has("legAlign")
      ? "다리 정렬을 실시간으로 확인하며 할 수 있어요"
      : "하체 근력의 기본 운동이에요",
  };
  return [squat, ...others];
}
