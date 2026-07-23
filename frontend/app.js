// AI 스쿼트 코치
// 화면 흐름: 시작 → 운동(카메라 → MediaPipe → 각도 → 최저점 판정 → 피드백) → 리포트
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { loadPostureModel, predictPosture } from "./inference.js";

// ---------- 설정값 (팀 데이터로 튜닝할 것) ----------
const CONFIG = {
  // 규칙 기반 v0 임계값
  DEPTH_KNEE_ANGLE: 100,   // 최저점에서 무릎 각도가 이보다 크면 "깊이 부족"
  TRUNK_LEAN_MAX: 50,      // 최저점에서 허리 기울기(수직 기준 도)가 이보다 크면 "허리 굽음"
  // 동작 구간 판정
  STANDING_KNEE_ANGLE: 160, // 이 이상이면 서 있는 상태
  BOTTOM_ENTER_DELTA: 5,    // 무릎 각도가 이만큼 다시 커지면 최저점을 지난 것
  // 인식 안정화
  MIN_VISIBILITY: 0,       // 가시성 게이트 비활성화 (v1 기준). 유령 스켈레톤이 다시 생기면 0.3~0.5로 올려서 테스트
  LANDMARK_ALPHA: 0.4,     // 랜드마크 EMA 스무딩 계수 (작을수록 부드럽고 반응 느림)
};

// 클래스 메타 (ml/train.py의 클래스 코드와 동일하게 유지)
const CLASSES = {
  good:  { name: "정상",      msg: "좋은 자세!",          color: "var(--c-good)" },
  depth: { name: "깊이 부족", msg: "더 깊이 앉으세요!",    color: "var(--c-depth)" },
  back:  { name: "허리 굽음", msg: "허리를 세우세요!",     color: "var(--c-back)" },
  knee:  { name: "무릎 모임", msg: "무릎을 벌려 주세요!",  color: "var(--c-knee)" },
};

// MediaPipe 랜드마크 인덱스 (33개 중 사용하는 것)
const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const ui = {
  screens: { start: $("screen-start"), workout: $("screen-workout"), report: $("screen-report") },
  startBtn: $("btn-start"), startStatus: $("start-status"),
  knee: $("knee-angle"), hip: $("hip-angle"), trunk: $("trunk-lean"),
  phase: $("squat-phase"), reps: $("rep-count"), goalDisplay: $("rep-goal-display"),
  repDots: $("rep-dots"),
  feedback: $("feedback"), status: $("status"),
  debugPanel: $("debug-panel"), engine: $("engine"),
  reportGood: $("report-good"), reportTotal: $("report-total"), reportTopError: $("report-top-error"),
  distBar: $("dist-bar"), distLegend: $("dist-legend"), repList: $("rep-list"),
};

function showScreen(name) {
  for (const [k, el] of Object.entries(ui.screens)) el.classList.toggle("hidden", k !== name);
}

// ---------- 세션 상태 ----------
const session = {
  goalReps: 10,
  results: [],        // 렙별 판정 결과 [{label, kneeAngle}, ...]
  pendingResult: null, // 최저점 판정 후 렙 완료(기립)까지 보관
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

function resetSquat() {
  squat.phase = "standing";
  squat.prevKnee = null;
  squat.minKneeThisRep = 999;
  squat.bottomFeatures = null;
  squat.reps = 0;
}

function updatePhase(f) {
  const k = f.kneeAngle;
  if (k == null) return { bottomFeatures: null, repCompleted: false };
  const delta = squat.prevKnee == null ? 0 : k - squat.prevKnee;
  squat.prevKnee = k;
  let bottomFeatures = null;
  let repCompleted = false;

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
        bottomFeatures = squat.bottomFeatures; // 최저점 확정 → 이 프레임으로 판정
      }
      break;
    case "ascending":
      if (k >= CONFIG.STANDING_KNEE_ANGLE) {
        squat.phase = "standing";
        squat.reps += 1;
        repCompleted = true;
      }
      // 올라가다 다시 내려가는 경우(불완전 렙) 처리
      if (delta < -CONFIG.BOTTOM_ENTER_DELTA) {
        squat.phase = "descending";
      }
      break;
  }
  return { bottomFeatures, repCompleted };
}

// ---------- 자세 판정 ----------
// ML 모델(model_rules.json)이 있으면 Random Forest, 없으면 규칙 기반 v0
let postureModel = null;

function classifyRuleBased(f) {
  if (f.kneeAngle > CONFIG.DEPTH_KNEE_ANGLE) return "depth";
  if (f.trunkLean > CONFIG.TRUNK_LEAN_MAX) return "back";
  return "good";
}

function classifyPosture(f) {
  if (postureModel) {
    const label = predictPosture(postureModel, f);
    if (label && CLASSES[label]) return label;
  }
  return classifyRuleBased(f);
}

// ---------- 피드백 ----------
let feedbackTimer = null;
function showFeedback(label) {
  const c = CLASSES[label];
  ui.feedback.textContent = c.msg;
  ui.feedback.classList.toggle("ok", label === "good");
  ui.feedback.classList.remove("hidden");
  speak(c.msg);
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

// ---------- 렙 도트 ----------
function initRepDots() {
  ui.repDots.innerHTML = "";
  for (let i = 0; i < session.goalReps; i++) ui.repDots.appendChild(document.createElement("i"));
}
function updateRepDot(index, label) {
  const dot = ui.repDots.children[index];
  if (dot) dot.className = label === "good" ? "good" : "err";
}

// ---------- 메인 루프 ----------
let landmarker = null;
let drawer = null;
let lastVideoTime = -1;
let running = false;
let stream = null;

async function loadModel() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  // 인식 설정은 v1(첫 버전) 기준: lite 모델 + 기본 신뢰도. 이 기기에서 full 모델/신뢰도 0.6은 스켈레톤이 안 떴음
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
  } catch (gpuErr) {
    console.warn("GPU delegate 실패, CPU로 전환:", gpuErr);
    landmarker = await PoseLandmarker.createFromOptions(fileset, makeOptions("CPU"));
  }
}

async function startWorkout() {
  ui.startBtn.disabled = true;
  try {
    if (!landmarker) {
      ui.startStatus.textContent = "모델 로딩 중…";
      await loadModel();
    }
    ui.startStatus.textContent = "카메라 연결 중…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise((r) => (video.onloadedmetadata = r));
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    drawer = new DrawingUtils(ctx);

    // 세션 초기화
    session.results = [];
    session.pendingResult = null;
    resetSquat();
    smoothedLms = null;
    ui.reps.textContent = "0";
    ui.goalDisplay.textContent = ` / ${session.goalReps}`;
    ui.phase.textContent = "준비";
    initRepDots();

    ui.startStatus.textContent = "";
    showScreen("workout");
    ui.status.textContent = "동작 인식 중";
    running = true;
    lastVideoTime = -1;
    requestAnimationFrame(loop);
  } catch (err) {
    ui.startStatus.textContent = `오류: ${err.message}`;
    console.error(err);
  } finally {
    ui.startBtn.disabled = false;
  }
}

function endWorkout() {
  running = false;
  speechSynthesis?.cancel();
  clearTimeout(feedbackTimer);
  ui.feedback.classList.add("hidden");
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  renderReport();
  showScreen("report");
}

function loop() {
  if (!running) return;
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

      const { bottomFeatures, repCompleted } = updatePhase(f);
      const phaseKo = { standing: "서 있음", descending: "내려가는 중", ascending: "올라오는 중" };
      ui.phase.textContent = phaseKo[squat.phase] ?? squat.phase;
      ui.reps.textContent = squat.reps;

      if (bottomFeatures) {
        const label = classifyPosture(bottomFeatures);
        session.pendingResult = { label, kneeAngle: bottomFeatures.kneeAngle };
        showFeedback(label);
      }
      if (repCompleted) {
        const r = session.pendingResult ?? { label: "good", kneeAngle: null };
        session.pendingResult = null;
        updateRepDot(session.results.length, r.label);
        session.results.push(r);
        if (session.results.length >= session.goalReps) {
          speak("운동 완료!");
          endWorkout();
          return;
        }
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

// ---------- 리포트 ----------
function renderReport() {
  const results = session.results;
  const total = results.length;
  const counts = {};
  for (const r of results) counts[r.label] = (counts[r.label] ?? 0) + 1;
  const goodCount = counts.good ?? 0;

  ui.reportGood.textContent = goodCount;
  ui.reportTotal.textContent = total;

  const errors = Object.entries(counts).filter(([l]) => l !== "good").sort((a, b) => b[1] - a[1]);
  ui.reportTopError.innerHTML = errors.length
    ? `${CLASSES[errors[0][0]].name} <small>${errors[0][1]}회</small>`
    : "없음 🎉";

  // 분포 바 (등장한 클래스만, good 먼저)
  ui.distBar.innerHTML = "";
  ui.distLegend.innerHTML = "";
  const order = ["good", "depth", "back", "knee"];
  const shown = order.filter((l) => (counts[l] ?? 0) > 0);
  ui.distBar.setAttribute("aria-label",
    shown.map((l) => `${CLASSES[l].name} ${counts[l]}회`).join(", "));
  shown.forEach((l, i) => {
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.style.flex = counts[l];
    seg.style.background = CLASSES[l].color;
    const first = i === 0, last = i === shown.length - 1;
    seg.style.borderRadius = first && last ? "4px" : first ? "4px 0 0 4px" : last ? "0 4px 4px 0" : "0";
    ui.distBar.appendChild(seg);
  });
  // 범례는 4클래스 항상 표시 (0회 포함 — 색·이름 대응을 고정)
  for (const l of order) {
    const span = document.createElement("span");
    span.innerHTML = `<i style="background:${CLASSES[l].color}"></i>${CLASSES[l].name} ${counts[l] ?? 0}`;
    ui.distLegend.appendChild(span);
  }

  // 렙별 리스트
  ui.repList.innerHTML = "";
  results.forEach((r, i) => {
    const li = document.createElement("li");
    const angle = r.kneeAngle != null ? `무릎 ${r.kneeAngle.toFixed(0)}°` : "";
    const lab = r.label === "good" ? CLASSES.good.name : `${CLASSES[r.label].name} — ${CLASSES[r.label].msg}`;
    li.innerHTML = `<span class="n">${i + 1}회</span><i style="background:${CLASSES[r.label].color}"></i>` +
      `<span class="lab">${lab}</span><span class="ang">${angle}</span>`;
    ui.repList.appendChild(li);
  });
}

// ---------- 이벤트 바인딩 ----------
document.querySelectorAll(".chip[data-goal]").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip[data-goal]").forEach((c) => c.classList.remove("on"));
    chip.classList.add("on");
    session.goalReps = Number(chip.dataset.goal);
  });
});
ui.startBtn.addEventListener("click", startWorkout);
$("btn-end").addEventListener("click", endWorkout);
$("btn-again").addEventListener("click", () => showScreen("start"));
$("btn-debug").addEventListener("click", () => ui.debugPanel.classList.toggle("hidden"));

// 시작 화면에서 미리 모델 로딩 (시작 버튼 누를 때 대기 시간 감소)
showScreen("start");
loadModel().catch((e) => console.warn("모델 사전 로딩 실패(시작 시 재시도):", e));
loadPostureModel().then((m) => {
  postureModel = m;
  const engine = m ? `ML (트리 ${m.trees.length}개)` : "규칙 기반 v0";
  ui.engine.textContent = engine;
  console.info(`자세 판정 엔진: ${engine}`);
});
