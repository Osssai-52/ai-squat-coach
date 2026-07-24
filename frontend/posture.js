// Posture analysis: frame validation → metrics → assessment (ML first, rule fallback) → recommendations
import { LM, trunkLean, kneeAnkleRatio } from "./pose.js";
import { loadPostureModel, predictPosture } from "./inference.js";

// Posture ML model (posture_model.json from ml/train_posture.py — falls back to rules if absent)
let postureML = null;
export async function initPostureML() {
  postureML = await loadPostureModel("posture_model.json");
  return !!postureML;
}
export function isPostureML() { return !!postureML; }

// ---------- Capture frame validation (missing-body-part guidance) ----------
const FRAME_MARGIN = 0.02;

function landmarkVisible(p) {
  const inFrame =
    p.x > FRAME_MARGIN && p.x < 1 - FRAME_MARGIN &&
    p.y > FRAME_MARGIN && p.y < 1 - FRAME_MARGIN;
  const vis = p.visibility;
  return inFrame && (vis == null || vis > 0.3);
}

const PART_CHECKS = [
  { ids: [LM.NOSE, LM.L_EAR, LM.R_EAR], any: true, msg: "Make sure your head is in the frame" },
  { ids: [LM.L_SHOULDER, LM.R_SHOULDER], any: false, msg: "Keep your shoulders visible" },
  { ids: [LM.L_HIP, LM.R_HIP], any: false, msg: "Fit your whole body in the frame" },
  { ids: [LM.L_KNEE, LM.R_KNEE], any: false, msg: "Step back so your knees are visible" },
  { ids: [LM.L_ANKLE, LM.R_ANKLE, LM.L_FOOT, LM.R_FOOT], any: false, msg: "Step back until your feet are in the frame" },
];

// Returns { ok: true } or { ok: false, msg: "guidance" }
export function validateFrame(lms) {
  if (!lms) return { ok: false, msg: "No one detected. Make sure your full body is visible in good lighting" };
  for (const check of PART_CHECKS) {
    const results = check.ids.map((i) => landmarkVisible(lms[i]));
    const pass = check.any ? results.some(Boolean) : results.every(Boolean);
    if (!pass) return { ok: false, msg: check.msg };
  }
  return { ok: true };
}

// ---------- Posture metrics ----------
// Thresholds are initial estimates — tune with team measurements (normal vs exaggerated poses)
const T = {
  FORWARD_HEAD: 0.12,   // ear-shoulder horizontal offset / torso length
  ROUND_SHOULDER: 0.10, // shoulder-hip horizontal offset / torso length
  TRUNK_LEAN: 8,        // torso tilt (degrees)
  TILT: 0.045,          // left-right height difference / width
  KNEE_IN: 0.75,        // knee/ankle gap ratio lower bound (knees caving in)
  KNEE_OUT: 1.45,       // knee/ankle gap ratio upper bound (knees bowing out)
};

function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// side: side-view landmarks / front: front-view landmarks
// Returns items: [{key, name, value, unit, status: "ok"|"warn", desc}]
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

      // Use ML classification when a model is present, threshold rules otherwise
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
        key: "forwardHead", name: "Neck Alignment", value: (fh * 100).toFixed(0), unit: "%",
        status: neckWarn ? "warn" : "ok",
        desc: neckWarn
          ? "Your head sits forward of your shoulders (forward head posture)"
          : "Ears and shoulders are well aligned",
      });
      items.push({
        key: "roundShoulder", name: "Shoulder Alignment", value: (rs * 100).toFixed(0), unit: "%",
        status: shoulderWarn ? "warn" : "ok",
        desc: shoulderWarn
          ? "Your shoulders roll forward (rounded shoulders)"
          : "Shoulders sit nicely over your hips",
      });
    }

    const lean = trunkLean(sh, hip);
    items.push({
      key: "trunk", name: "Torso Tilt", value: lean.toFixed(0), unit: "°",
      status: lean > T.TRUNK_LEAN ? "warn" : "ok",
      desc: lean > T.TRUNK_LEAN
        ? "Your torso leans noticeably — watch your spine alignment"
        : "You're standing nice and tall",
    });
  }

  if (front) {
    const shoulderW = dist(front[LM.L_SHOULDER], front[LM.R_SHOULDER]);
    if (shoulderW > 1e-6) {
      const st = Math.abs(front[LM.L_SHOULDER].y - front[LM.R_SHOULDER].y) / shoulderW;
      items.push({
        key: "shoulderTilt", name: "Shoulder Level", value: (st * 100).toFixed(1), unit: "%",
        status: st > T.TILT ? "warn" : "ok",
        desc: st > T.TILT ? "One shoulder sits higher than the other" : "Shoulders are nearly level",
      });
    }
    const hipW = dist(front[LM.L_HIP], front[LM.R_HIP]);
    if (hipW > 1e-6) {
      const pt = Math.abs(front[LM.L_HIP].y - front[LM.R_HIP].y) / hipW;
      items.push({
        key: "pelvisTilt", name: "Pelvis Level", value: (pt * 100).toFixed(1), unit: "%",
        status: pt > T.TILT ? "warn" : "ok",
        desc: pt > T.TILT ? "One hip sits higher than the other" : "Hips are nearly level",
      });
    }
    const ratio = kneeAnkleRatio(front);
    if (ratio != null) {
      const warn = ratio < T.KNEE_IN || ratio > T.KNEE_OUT;
      items.push({
        key: "legAlign", name: "Leg Alignment", value: ratio.toFixed(2), unit: "",
        status: warn ? "warn" : "ok",
        desc: ratio < T.KNEE_IN ? "Your knees tend to cave inward"
          : ratio > T.KNEE_OUT ? "Your knees tend to bow outward"
          : "Knees and ankles line up well",
      });
    }
  }

  return items;
}

// ---------- Exercise library ----------
export const EXERCISES = {
  squat: {
    name: "Squat", tag: "Legs · Full body", live: true,
    summary: "The all-in-one move for legs and core — with real-time form analysis.",
    steps: [
      "Stand with feet shoulder-width apart, toes slightly out",
      "Keep your chest up and eyes forward",
      "Push your hips back and sink until your thighs are parallel to the floor",
      "Keep your knees tracking in line with your toes",
      "Drive through your heels to stand back up",
    ],
    dose: "10 reps × 3 sets, rest 60s between sets",
  },
  plank: {
    name: "Plank", tag: "Core stability",
    summary: "A static core hold that builds the strength to keep your spine tall.",
    steps: [
      "Place your elbows under your shoulders and lie face down",
      "Form a straight line from head to heels",
      "Brace your abs and don't let your hips sag",
      "Keep breathing throughout",
    ],
    dose: "Start with 30s × 3 sets, build up gradually",
  },
  bandPullApart: {
    name: "Band Pull-Apart", tag: "Upper back · Shoulders",
    summary: "Strengthens the upper back to pull rounded shoulders back where they belong.",
    steps: [
      "Hold a band at shoulder width, arms extended forward",
      "Keeping elbows straight, pull the band apart at chest height",
      "Squeeze your shoulder blades together and hold for 2 seconds",
      "Return slowly",
    ],
    dose: "15 reps × 3 sets",
  },
  wallAngel: {
    name: "Wall Angel", tag: "Shoulder mobility · Posture",
    summary: "Slide your arms along a wall to restore shoulder and upper-back alignment.",
    steps: [
      "Stand with your head, back, and hips against a wall",
      "Bend your elbows to 90° and press them to the wall",
      "Slide your arms up slowly without losing wall contact",
      "Lower slowly and repeat",
    ],
    dose: "10 reps × 3 sets",
  },
  hipBridge: {
    name: "Hip Bridge", tag: "Glutes · Pelvic stability",
    summary: "Wakes up the glutes and stabilizes the pelvis.",
    steps: [
      "Lie on your back, knees bent, feet hip-width apart",
      "Press through your heels and lift your hips",
      "Hold 2 seconds when shoulders, hips, and knees form a line",
      "Lower slowly",
    ],
    dose: "12 reps × 3 sets",
  },
  clamshell: {
    name: "Clamshell", tag: "Glutes · Knee alignment",
    summary: "Targets the outer glutes that keep your knees from caving inward.",
    steps: [
      "Lie on your side with knees bent at 45°",
      "Keeping heels together, open your top knee toward the ceiling",
      "Don't let your pelvis roll backward",
      "Close slowly",
    ],
    dose: "15 reps per side × 3 sets",
  },
  sidePlank: {
    name: "Side Plank", tag: "Obliques · Balance",
    summary: "A side core hold that evens out left-right imbalances.",
    steps: [
      "Lie on your side with your elbow under your shoulder",
      "Lift your hips until head, hips, and feet form a line",
      "Do one extra set on your weaker side",
    ],
    dose: "20s per side × 3 sets",
  },
};

// Scan results → 2-3 recommended exercises
// Squat always included (it has live form check); up to 2 others based on flagged items
export function buildRecommendations(items) {
  const warn = new Set(items.filter((i) => i.status === "warn").map((i) => i.key));
  const others = [];
  const add = (key, reason) => {
    if (others.length < 2 && !others.some((p) => p.key === key)) others.push({ key, reason });
  };

  if (warn.has("legAlign")) add("clamshell", "Helps correct knees caving inward");
  if (warn.has("forwardHead") || warn.has("roundShoulder")) {
    add("bandPullApart", "Great for pulling rounded shoulders back");
    add("wallAngel", "Restores shoulder and neck alignment");
  }
  if (warn.has("trunk")) add("plank", "Builds the core strength to stand tall");
  if (warn.has("shoulderTilt") || warn.has("pelvisTilt")) {
    add("sidePlank", "Helps balance left-right asymmetry");
    add("hipBridge", "Good for pelvic stability");
  }
  if (others.length === 0) add("plank", "A core essential for everyone");

  const squat = {
    key: "squat",
    reason: warn.has("legAlign")
      ? "Check your knee alignment in real time"
      : "The foundation of lower-body strength",
  };
  return [squat, ...others];
}
