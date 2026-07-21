// AI 스쿼트 코치 — 1일차 뼈대
// 카메라 → MediaPipe 포즈 추출 → 각도 계산 → 규칙 기반 판정 v0 → 화면 피드백
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---------- 설정값 (팀 데이터로 튜닝할 것) ----------
const CONFIG = {
  // 규칙 기반 v0 임계값
  DEPTH_KNEE_ANGLE: 100,   // 최저점에서 무릎 각도가 이보다 크면 "깊이 부족"
  TRUNK_LEAN_MAX: 50,      // 최저점에서 허리 기울기(수직 기준 도)가 이보다 크면 "허리 굽음"
  // 동작 구간 판정
  STANDING_KNEE_ANGLE: 160, // 이 이상이면 서 있는 상태
  BOTTOM_ENTER_DELTA: 5,    // 무릎 각도 변화가 이 이하로 안정되면 최저점 후보
  // 피드백 스무딩: 같은 판정이 연속 N프레임 이상일 때만 표시
  SMOOTHING_FRAMES: 5,
  // 인식 안정화
  MIN_VISIBILITY: 0.6,     // 핵심 관절 평균 가시성이 이 미만이면 "인식 불안정" 처리
  LANDMARK_ALPHA: 0.4,     // 랜드마크 EMA 스무딩 계수 (작을수록 부드럽고 반응 느림)
};

// MediaPipe 랜드마크 인덱스 (33개 중 사용하는 것)
const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
};

// ---------- DOM ----------
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);
const ui = {
  knee: $("knee-angle"), hip: $("hip-angle"), trunk: $("trunk-lean"),
  phase: $("squat-phase"), reps: $("rep-count"),
  feedback: $("feedback"), status: $("status"),
};

// ---------- 각도 유틸 (ml/features.py와 반드시 동일한 정의 유지) ----------
// 세 점 A-B-C에서 B를 꼭짓점으로 하는 각도(도)
function angleAt(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const n1 = Math.hypot(v1.x, v1.y);
  const n2 = Math.hypot(v2.x, v2.y);
  if (n1 === 0 || n2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, dot / (n1 * n2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

// 어깨-엉덩이 선이 수직선과 이루는 각도(도). 0 = 꼿꼿이 선 상태
function trunkLean(shoulder, hip) {
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y; // 화면 좌표: 아래로 갈수록 y 증가
  return Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI);
}

// 측면 촬영이므로 카메라에 가까운(가시성 높은) 쪽 관절만 사용
function pickSide(lms) {
  const leftVis = (lms[LM.L_HIP].visibility ?? 0) + (lms[LM.L_KNEE].visibility ?? 0);
  const rightVis = (lms[LM.R_HIP].visibility ?? 0) + (lms[LM.R_KNEE].visibility ?? 0);
  return leftVis >= rightVis
    ? { shoulder: lms[LM.L_SHOULDER], hip: lms[LM.L_HIP], knee: lms[LM.L_KNEE], ankle: lms[LM.L_ANKLE] }
    : { shoulder: lms[LM.R_SHOULDER], hip: lms[LM.R_HIP], knee: lms[LM.R_KNEE], ankle: lms[LM.R_ANKLE] };
}

// ---------- 인식 안정화 ----------
// 핵심 관절(어깨·엉덩이·무릎)의 평균 가시성으로 "진짜 사람이 잡혔는지" 판정
const CORE_LMS = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE];
let lastAvgVisibility = null; // 디버깅용
function poseReliable(lms) {
  // 일부 MediaPipe JS 빌드는 visibility를 안 주거나 전부 0으로 준다.
  // 그 경우 게이트를 끄지 않으면 모든 프레임이 차단됨.
  const maxVis = Math.max(...lms.map((p) => p.visibility ?? 0));
  if (maxVis < 0.01) {
    lastAvgVisibility = null; // visibility 미지원 → 게이트 비활성화
    return true;
  }
  const avg = CORE_LMS.reduce((s, i) => s + (lms[i].visibility ?? 0), 0) / CORE_LMS.length;
  lastAvgVisibility = avg;
  return avg >= CONFIG.MIN_VISIBILITY;
}

// 랜드마크 EMA 스무딩: 프레임 간 떨림 제거
let smoothedLms = null;
function smoothLandmarks(lms) {
  if (!smoothedLms || smoothedLms.length !== lms.length) {
    smoothedLms = lms.map((p) => ({ ...p }));
    return smoothedLms;
  }
  const a = CONFIG.LANDMARK_ALPHA;
  for (let i = 0; i < lms.length; i++) {
    const s = smoothedLms[i], p = lms[i];
    s.x += a * (p.x - s.x);
    s.y += a * (p.y - s.y);
    s.z += a * (p.z - s.z);
    s.visibility = p.visibility;
  }
  return smoothedLms;
}

function computeFeatures(lms) {
  const s = pickSide(lms);
  return {
    kneeAngle: angleAt(s.hip, s.knee, s.ankle),
    hipAngle: angleAt(s.shoulder, s.hip, s.knee),
    trunkLean: trunkLean(s.shoulder, s.hip),
  };
}

// ---------- 동작 구간 분리 + 렙 카운팅 ----------
// standing → descending → bottom → ascending → standing (1렙)
const squat = {
  phase: "standing",
  prevKnee: null,
  minKneeThisRep: 999,
  bottomFeatures: null, // 최저점 프레임의 피처 (여기서만 자세 판정)
  reps: 0,
};

function updatePhase(f) {
  const k = f.kneeAngle;
  if (k == null) return null;
  const delta = squat.prevKnee == null ? 0 : k - squat.prevKnee;
  squat.prevKnee = k;
  let bottomEvent = null;

  switch (squat.phase) {
    case "standing":
      if (k < CONFIG.STANDING_KNEE_ANGLE - 10) {
        squat.phase = "descending";
        squat.minKneeThisRep = k;
        squat.bottomFeatures = null;
      }
      break;
    case "descending":
      if (k < squat.minKneeThisRep) {
        squat.minKneeThisRep = k;
        squat.bottomFeatures = { ...f };
      }
      // 각도가 다시 커지기 시작하면 최저점을 지난 것
      if (delta > CONFIG.BOTTOM_ENTER_DELTA) {
        squat.phase = "ascending";
        bottomEvent = squat.bottomFeatures; // 최저점 확정 → 이 프레임으로 판정
      }
      break;
    case "ascending":
      if (k >= CONFIG.STANDING_KNEE_ANGLE) {
        squat.phase = "standing";
        squat.reps += 1;
      }
      // 올라가다 다시 내려가는 경우(불완전 렙) 처리
      if (delta < -CONFIG.BOTTOM_ENTER_DELTA) {
        squat.phase = "descending";
      }
      break;
  }
  return bottomEvent;
}

// ---------- 규칙 기반 판정 v0 (2일차에 ML 모델로 교체) ----------
function classifyRuleBased(f) {
  if (f.kneeAngle > CONFIG.DEPTH_KNEE_ANGLE) return { label: "depth", msg: "깊이 부족! 더 내려가세요" };
  if (f.trunkLean > CONFIG.TRUNK_LEAN_MAX) return { label: "back", msg: "허리를 세우세요" };
  return { label: "good", msg: "좋은 자세!" };
}

// ---------- 피드백 표시 (스무딩) ----------
let lastLabel = null;
let sameCount = 0;
let feedbackTimer = null;

function showFeedback(result) {
  // 렙 단위 판정이라 최저점마다 1회 호출됨 → 스무딩은 프레임 판정으로 바꿀 때 사용
  ui.feedback.textContent = result.msg;
  ui.feedback.classList.toggle("ok", result.label === "good");
  ui.feedback.classList.remove("hidden");
  speak(result.msg);
  clearTimeout(feedbackTimer);
  feedbackTimer = setTimeout(() => ui.feedback.classList.add("hidden"), 2000);
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "ko-KR";
  speechSynthesis.speak(u);
}

// ---------- 메인 루프 ----------
let landmarker = null;
let drawer = null;
let lastVideoTime = -1;

async function init() {
  try {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    const makeOptions = (delegate) => ({
      baseOptions: {
        // lite → full: 떨림이 훨씬 적음. 데모 기기에서 프레임이 안 나오면 lite로 되돌릴 것
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
      // 낮으면 배경 사물을 사람으로 오인해 유령 스켈레톤이 생김
      minPoseDetectionConfidence: 0.6,
      minPosePresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    try {
      landmarker = await PoseLandmarker.createFromOptions(fileset, makeOptions("GPU"));
    } catch (gpuErr) {
      console.warn("GPU delegate 실패, CPU로 전환:", gpuErr);
      landmarker = await PoseLandmarker.createFromOptions(fileset, makeOptions("CPU"));
    }
    ui.status.textContent = "카메라 연결 중…";

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise((r) => (video.onloadedmetadata = r));
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    drawer = new DrawingUtils(ctx);
    ui.status.textContent = "동작 인식 중";
    requestAnimationFrame(loop);
  } catch (err) {
    ui.status.textContent = `오류: ${err.message}`;
    console.error(err);
  }
}

function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks.length > 0 && poseReliable(result.landmarks[0])) {
      ui.status.textContent = "동작 인식 중";
      const lms = smoothLandmarks(result.landmarks[0]);
      drawer.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, { color: "#4fc3f7", lineWidth: 3 });
      drawer.drawLandmarks(lms, { color: "#ffca28", radius: 4 });

      const f = computeFeatures(lms);
      ui.knee.textContent = f.kneeAngle != null ? `${f.kneeAngle.toFixed(0)}°` : "–";
      ui.hip.textContent = f.hipAngle != null ? `${f.hipAngle.toFixed(0)}°` : "–";
      ui.trunk.textContent = `${f.trunkLean.toFixed(0)}°`;

      const bottomFeatures = updatePhase(f);
      const phaseKo = { standing: "서 있음", descending: "내려가는 중", ascending: "올라오는 중" };
      ui.phase.textContent = phaseKo[squat.phase] ?? squat.phase;
      ui.reps.textContent = squat.reps;

      if (bottomFeatures) {
        showFeedback(classifyRuleBased(bottomFeatures));
      }
    } else {
      // 사람이 확실히 안 잡히면 스켈레톤을 그리지 않고 스무딩 상태 초기화
      smoothedLms = null;
      squat.prevKnee = null;
      const vis = lastAvgVisibility != null ? ` (가시성 ${lastAvgVisibility.toFixed(2)})` : "";
      ui.status.textContent = result.landmarks.length === 0
        ? "사람이 감지되지 않음 — 조명과 거리를 확인하세요"
        : `인식 불안정 — 전신(측면)이 화면에 들어오게 서 주세요${vis}`;
    }
  }
  requestAnimationFrame(loop);
}

init();
