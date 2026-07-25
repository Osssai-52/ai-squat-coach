// BodyBuddy: 체형 분석 → 맞춤 운동 추천 → 스쿼트 실시간 자세 교정
import { getLandmarker, computeSquatFeatures, PoseLandmarker, DrawingUtils, LM } from "./pose.js";
import { validateFrame, analyzePosture, buildRecommendations, EXERCISES, initPostureML, isPostureML } from "./posture.js";
import { loadPostureModel, predictPosture } from "./inference.js";

// ---------- 스쿼트 판정 설정 (팀 실측으로 튜닝) ----------
// 임계값은 45° 실측 분포로 보정 (ml/train.py 상수와 동일 유지)
const CONFIG = {
  DEPTH_KNEE_ANGLE: 118,
  TRUNK_LEAN_MAX: 45,
  HEEL_FOOT_ANGLE: 34,
  KNEE_RATIO_MIN: 0.65,
  // 렙 감지: "엉덩이-무릎 세로 간격 ÷ 몸통 길이" 비율 기준 (스쿼트 깊이 progress)
  // 같은 프레임 안의 관절 간 상대 간격이라 카메라와의 거리 변화에 영향받지 않음.
  DEPTH_ENTER: 0.20,
  DEPTH_EXIT: 0.12,
  REP_MIN_PROGRESS: 0.45,
  BOTTOM_REBOUND: 0.04,
  STAND_STILL: 0.10,
  LANDMARK_ALPHA: 0.4,
  FOOT_ANGLE_REF: 26,
  BASELINE_ALPHA: 0.1,
};

const CLASSES = {
  good:  { name: "Good form",     banner: "Great form!",           color: "var(--c-good)",
           advice: "" },
  depth: { name: "Shallow depth", banner: "Go a little deeper!",   color: "var(--c-depth)",
           advice: "Lower your hips until your thighs are parallel to the floor" },
  back:  { name: "Rounded back",  banner: "Chest up!",             color: "var(--c-back)",
           advice: "Keep your chest open and look forward" },
  heel:  { name: "Heel lift",     banner: "Keep your heels down!", color: "var(--c-heel)",
           advice: "Keep your weight through mid-foot and heels" },
  knee:  { name: "Knees caving",  banner: "Push your knees out!",  color: "var(--c-knee)",
           advice: "Keep your knees tracking in line with your toes" },
};

// ---------- 저장소 ----------
const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem("fitform:" + key)) ?? fallback; }
    catch { return fallback; }
  },
  set(key, value) { localStorage.setItem("fitform:" + key, JSON.stringify(value)); },
};

// ---------- DOM / 화면 전환 ----------
const $ = (id) => document.getElementById(id);
const VIEWS = ["v-onboard", "v-shell", "v-capture", "v-result", "v-exlist", "v-exercise", "v-squat-setup", "v-workout", "v-report"];
// 브랜드 바는 온보딩·홈에서만 (기능 화면은 화면 제목 + 뒤로가기 중심)
const BRAND_VIEWS = new Set(["v-onboard", "v-shell"]);

function showView(id) {
  for (const v of VIEWS) $(v).classList.toggle("hidden", v !== id);
  $("brand-bar").classList.toggle("hidden", !BRAND_VIEWS.has(id));
}
function showTab(name) {
  $("tab-home").classList.toggle("hidden", name !== "home");
  $("tab-log").classList.toggle("hidden", name !== "log");
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  if (name === "home") renderHome();
  if (name === "log") renderLog();
}
function toast(el, msg, ms = 2200) {
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), ms);
}

// ---------- 시작 스플래시 ----------
function runSplash() {
  const splash = $("splash");
  const img = $("splash-logo");
  requestAnimationFrame(() => splash.classList.add("in"));
  setTimeout(() => {
    const targetEl = document.querySelector("#brand-bar img");
    if (targetEl && !$("brand-bar").classList.contains("hidden")) {
      const t = targetEl.getBoundingClientRect();
      const c = img.getBoundingClientRect();
      const dx = t.left + t.width / 2 - (c.left + c.width / 2);
      const dy = t.top + t.height / 2 - (c.top + c.height / 2);
      img.style.transform = `translate(${dx}px, ${dy}px) scale(${t.width / c.width})`;
    }
    splash.classList.add("fly");
    setTimeout(() => splash.remove(), 600);
  }, 950);
}

// ---------- 온보딩 ----------
function bmiInfo(h, w) {
  if (!h || !w) return null;
  const bmi = w / ((h / 100) ** 2);
  const cat = bmi < 18.5 ? "Underweight" : bmi < 23 ? "Normal" : bmi < 25 ? "Overweight" : "Obese";
  return { bmi: bmi.toFixed(1), cat };
}
function updateBmiLine() {
  const h = parseFloat($("in-height").value);
  const w = parseFloat($("in-weight").value);
  const info = bmiInfo(h, w);
  $("bmi-line").classList.toggle("hidden", !info);
  if (info) $("bmi-line").innerHTML = `BMI <b>${info.bmi}</b> · ${info.cat}`;
}
$("in-height").addEventListener("input", updateBmiLine);
$("in-weight").addEventListener("input", updateBmiLine);

function finishOnboard(save) {
  if (save) {
    const h = parseFloat($("in-height").value) || null;
    const w = parseFloat($("in-weight").value) || null;
    store.set("profile", { height: h, weight: w });
  }
  store.set("onboarded", true);
  showView("v-shell");
  showTab("home");
}
$("btn-onboard-start").addEventListener("click", () => finishOnboard(true));
$("btn-onboard-skip").addEventListener("click", () => finishOnboard(false));

// ---------- 홈 탭 ----------
function currentRecommendations() {
  const posture = store.get("posture", null);
  return posture ? buildRecommendations(posture.items) : [];
}

function renderHome() {
  const profile = store.get("profile", {});
  const info = bmiInfo(profile.height, profile.weight);
  $("home-sub").textContent = info
    ? `BMI ${info.bmi} (${info.cat}) · workouts tailored to your body`
    : "Scan your posture and get workouts made for you";

  const posture = store.get("posture", null);
  $("tab-home").classList.toggle("home-empty", !posture);
  $("card-scan").classList.toggle("hidden", !!posture);
  $("today-card").classList.toggle("hidden", !posture);
  $("reco-section").classList.toggle("hidden", !posture);
  if (!posture) return;

  // 오늘의 자세 요약: 무엇이 문제인지 문장으로
  const warns = posture.items.filter((i) => i.status === "warn");
  $("today-head").textContent = warns.length === 0
    ? "Looking great!"
    : `${warns.length} area${warns.length > 1 ? "s" : ""} need${warns.length > 1 ? "" : "s"} attention`;
  $("today-desc").textContent = warns.length === 0
    ? "All measured areas are in a good range. Keep it up!"
    : `${warns.map((w) => w.name).join(" and ")} need${warns.length > 1 ? "" : "s"} the most work.`;

  const chips = $("summary-chips");
  chips.innerHTML = "";
  for (const item of posture.items) {
    const pill = document.createElement("span");
    pill.className = `pill ${item.status}`;
    pill.textContent = `${item.name}: ${item.status === "ok" ? "Good" : "Needs attention"}`;
    chips.appendChild(pill);
  }

  const list = $("reco-list");
  list.innerHTML = "";
  for (const pick of currentRecommendations()) {
    const ex = EXERCISES[pick.key];
    const card = document.createElement("button");
    card.className = "reco-card";
    card.innerHTML =
      `<div class="t"><b>${ex.name}${ex.live ? ' <span class="badge live">Live form check</span>' : ""}</b>` +
      `<p>${pick.reason}</p></div><span class="arrow">›</span>`;
    card.addEventListener("click", () => openExercise(pick.key, "home"));
    list.appendChild(card);
  }
}

// ---------- 운동 라이브러리 (바로 운동하기 경로) ----------
function renderExerciseList() {
  const recoKeys = new Set(currentRecommendations().map((p) => p.key));
  const list = $("exlist");
  list.innerHTML = "";
  for (const [key, ex] of Object.entries(EXERCISES)) {
    const badges =
      (recoKeys.has(key) ? ' <span class="badge reco">Recommended</span>' : "") +
      (ex.live ? ' <span class="badge live">Live form check</span>' : "");
    const card = document.createElement("button");
    card.className = "reco-card";
    card.innerHTML =
      `<div class="t"><b>${ex.name}${badges}</b>` +
      `<p>${ex.tag} · ${ex.time}</p></div><span class="arrow">›</span>`;
    card.addEventListener("click", () => openExercise(key, "list"));
    list.appendChild(card);
  }
}

// ---------- 체형 분석 촬영 ----------
const capture = { step: "front", lms: { front: null, side: null }, stream: null };

async function startCapture() {
  capture.step = "front";
  capture.lms = { front: null, side: null };
  setCaptureStep();
  showView("v-capture");
  try {
    await getLandmarker();
    capture.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 720, height: 1280, facingMode: "user" }, audio: false,
    });
    $("cap-video").srcObject = capture.stream;
  } catch (err) {
    toast($("cap-toast"), `Couldn't open the camera: ${err.message}`, 3000);
  }
}
function stopCapture() {
  capture.stream?.getTracks().forEach((t) => t.stop());
  capture.stream = null;
}
function setCaptureStep() {
  const front = capture.step === "front";
  $("capture-title").textContent = `Posture Scan ${front ? "1" : "2"}/2`;
  $("guide-front").classList.toggle("hidden", !front);
  $("guide-side").classList.toggle("hidden", front);
  $("cap-hint").textContent = front
    ? "Stand facing the camera, matching the guide"
    : "Now turn to your side";
  $("cap-desc").textContent = front
    ? "Your whole body, head to toe, should be in frame"
    : "Full side view, with your whole body visible";
}

$("btn-shoot").addEventListener("click", async () => {
  const video = $("cap-video");
  if (!video.videoWidth) return;
  $("btn-shoot").disabled = true;
  try {
    const landmarker = await getLandmarker();
    const result = landmarker.detectForVideo(video, performance.now());
    const lms = result.landmarks[0] ?? null;
    const check = validateFrame(lms);
    if (!check.ok) {
      toast($("cap-toast"), check.msg);
      return;
    }
    capture.lms[capture.step] = lms;
    if (capture.step === "front") {
      capture.step = "side";
      setCaptureStep();
      toast($("cap-toast"), "Great! Now let's get your side view");
    } else {
      stopCapture();
      const items = analyzePosture(capture.lms);
      store.set("posture", { date: new Date().toISOString(), items });
      renderResult(items);
      showView("v-result");
    }
  } finally {
    $("btn-shoot").disabled = false;
  }
});

function renderResult(items) {
  const warns = items.filter((i) => i.status === "warn");
  const engine = isPostureML() ? " · ML analysis" : "";
  $("result-hero").innerHTML = warns.length === 0
    ? `<div class="big-num good">All Clear</div><p>All ${items.length} measurements look good${engine}</p>`
    : `<div class="big-num warn">${warns.length} to watch</div><p>${warns.map((w) => w.name).join(" · ")} could use some attention${engine}</p>`;
  const list = $("result-list");
  list.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="dot ${item.status}"></span><span class="m-name">${item.name}</span>` +
      `<span class="m-desc">${item.desc}</span><span class="m-val">${item.value}${item.unit}</span>`;
    list.appendChild(li);
  }
}

$("card-scan").addEventListener("click", startCapture);
$("btn-reanalyze").addEventListener("click", startCapture);
$("card-quick").addEventListener("click", () => { renderExerciseList(); showView("v-exlist"); });
$("btn-to-reco").addEventListener("click", () => { showView("v-shell"); showTab("home"); });

// ---------- 운동 상세 ----------
let currentExercise = null;
let exerciseOrigin = "home"; // "home"(추천에서) | "list"(라이브러리에서) — 뒤로가기 목적지
function openExercise(key, origin = "home") {
  currentExercise = key;
  exerciseOrigin = origin;
  const ex = EXERCISES[key];
  $("ex-title").textContent = ex.name;
  $("ex-tag").textContent = `${ex.tag} · ${ex.time}`;
  $("ex-summary").textContent = ex.summary;

  // 추천된 운동이면 이유를 함께 표시
  const pick = currentRecommendations().find((p) => p.key === key);
  $("ex-reason").classList.toggle("hidden", !pick);
  if (pick) $("ex-reason").textContent = `Recommended for you — ${pick.reason.toLowerCase()}`;

  const steps = $("ex-steps");
  steps.innerHTML = "";
  for (const s of ex.steps) {
    const li = document.createElement("li");
    li.textContent = s;
    steps.appendChild(li);
  }
  $("ex-dose").textContent = ex.dose;
  $("live-block").classList.toggle("hidden", !ex.live);
  showView("v-exercise");
}
$("btn-live").addEventListener("click", () => showView("v-squat-setup"));

// ---------- 스쿼트 세션 ----------
const session = { goalReps: 10, results: [], pendingResult: null };
let postureModel = null;
let landmarker = null;
let drawer = null;
let lastVideoTime = -1;
let running = false;
let workoutStream = null;
let smoothedLms = null;
let feedbackTimer = null;

const squat = {
  phase: "standing",
  reps: 0,
  baseGap: null,        // 서 있을 때 (무릎Y - 엉덩이Y)/몸통길이 기준 비율 (EMA)
  maxDropThisRep: 0,
  bottomFeatures: null, // 최저점 프레임 피처 (여기서만 자세 판정)
  footBaseline: null,
  lastDrop: 0,
};
function resetSquat() {
  Object.assign(squat, {
    phase: "standing", reps: 0, baseGap: null,
    maxDropThisRep: 0, bottomFeatures: null, footBaseline: null, lastDrop: 0,
  });
}

// 서 있는 동안 발 각도 기준선 수집 (EMA)
function updateFootBaseline(f) {
  if (f.footAngle == null) return;
  if (squat.footBaseline == null) squat.footBaseline = f.footAngle;
  else squat.footBaseline += CONFIG.BASELINE_ALPHA * (f.footAngle - squat.footBaseline);
}

// 판정 직전 발 각도를 학습 사진 분포 기준으로 환산
function calibrateFeatures(f) {
  if (f.footAngle == null || squat.footBaseline == null) return f;
  return { ...f, footAngle: f.footAngle - squat.footBaseline + CONFIG.FOOT_ANGLE_REF };
}

function smoothLandmarks(lms) {
  if (!smoothedLms || smoothedLms.length !== lms.length) {
    smoothedLms = lms.map((p) => ({ ...p }));
    return smoothedLms;
  }
  const a = CONFIG.LANDMARK_ALPHA;
  for (let i = 0; i < lms.length; i++) {
    const s = smoothedLms[i], p = lms[i];
    s.x += a * (p.x - s.x); s.y += a * (p.y - s.y); s.z += a * (p.z - s.z);
    s.visibility = p.visibility;
  }
  return smoothedLms;
}

// 스쿼트 깊이 progress: 엉덩이가 무릎 높이에 얼마나 가까워졌나 (0=서 있음, 1=무릎 높이)
// 관절 간 상대 간격을 같은 프레임의 몸통 길이로 나눠서, 카메라와의 거리·위치 변화에 불변
function computeDrop(lms) {
  const hipY = (lms[LM.L_HIP].y + lms[LM.R_HIP].y) / 2;
  const kneeY = (lms[LM.L_KNEE].y + lms[LM.R_KNEE].y) / 2;
  const shoulder = { x: (lms[LM.L_SHOULDER].x + lms[LM.R_SHOULDER].x) / 2,
                     y: (lms[LM.L_SHOULDER].y + lms[LM.R_SHOULDER].y) / 2 };
  const hip = { x: (lms[LM.L_HIP].x + lms[LM.R_HIP].x) / 2, y: hipY };
  const torso = Math.max(Math.hypot(shoulder.x - hip.x, shoulder.y - hip.y), 1e-3);

  const gap = (kneeY - hipY) / torso;

  if (squat.baseGap == null) {
    squat.baseGap = gap;
    return 0;
  }
  const progress = 1 - gap / squat.baseGap;
  if (squat.phase === "standing" && Math.abs(progress) < CONFIG.STAND_STILL) {
    squat.baseGap += 0.15 * (gap - squat.baseGap);
  }
  return progress;
}

function updatePhase(f, lms) {
  const drop = computeDrop(lms);
  squat.lastDrop = drop;
  let bottomFeatures = null, repCompleted = false;
  const deepEnough = () => squat.maxDropThisRep >= CONFIG.REP_MIN_PROGRESS;

  switch (squat.phase) {
    case "standing":
      updateFootBaseline(f);
      if (drop > CONFIG.DEPTH_ENTER) {
        squat.phase = "descending";
        squat.maxDropThisRep = drop;
        squat.bottomFeatures = { ...f };
      }
      break;
    case "descending":
      if (drop > squat.maxDropThisRep) {
        squat.maxDropThisRep = drop;
        squat.bottomFeatures = { ...f };
      }
      if (drop < squat.maxDropThisRep - CONFIG.BOTTOM_REBOUND) {
        squat.phase = "ascending";
        if (deepEnough()) bottomFeatures = squat.bottomFeatures;
      }
      break;
    case "ascending":
      if (drop > squat.maxDropThisRep) squat.phase = "descending";
      else if (drop < CONFIG.DEPTH_EXIT) {
        squat.phase = "standing";
        if (deepEnough()) { squat.reps += 1; repCompleted = true; }
        squat.maxDropThisRep = 0;
        squat.bottomFeatures = null;
      }
      break;
  }
  return { bottomFeatures, repCompleted };
}

function classifyRuleBased(f) {
  if (f.kneeAngle > CONFIG.DEPTH_KNEE_ANGLE) return "depth";
  if (f.trunkLean > CONFIG.TRUNK_LEAN_MAX) return "back";
  if (f.footAngle != null && f.footAngle > CONFIG.HEEL_FOOT_ANGLE) return "heel";
  if (f.kneeAnkleRatio != null && f.kneeAnkleRatio < CONFIG.KNEE_RATIO_MIN) return "knee";
  return "good";
}
function classifyPosture(f) {
  if (postureModel) {
    const label = predictPosture(postureModel, f);
    if (label && CLASSES[label]) return label;
  }
  return classifyRuleBased(f);
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  speechSynthesis.speak(u);
}
function showFeedback(label) {
  const c = CLASSES[label];
  $("feedback").textContent = c.banner;
  $("feedback").classList.toggle("ok", label === "good");
  $("feedback").classList.remove("hidden");
  speak(c.banner);
  clearTimeout(feedbackTimer);
  feedbackTimer = setTimeout(() => $("feedback").classList.add("hidden"), 2000);
}

function initRepDots() {
  const dots = $("rep-dots");
  dots.innerHTML = "";
  for (let i = 0; i < session.goalReps; i++) dots.appendChild(document.createElement("i"));
}
function updateRepDot(index, label) {
  const dot = $("rep-dots").children[index];
  if (dot) dot.className = label === "good" ? "good" : "err";
}

document.querySelectorAll(".chip[data-goal]").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip[data-goal]").forEach((c) => c.classList.remove("on"));
    chip.classList.add("on");
    session.goalReps = Number(chip.dataset.goal);
  });
});

$("btn-start-squat").addEventListener("click", async () => {
  const btn = $("btn-start-squat");
  btn.disabled = true;
  try {
    $("squat-status").textContent = "Preparing the model…";
    landmarker = await getLandmarker();
    $("squat-status").textContent = "Connecting to camera…";
    workoutStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }, audio: false,
    });
    const video = $("video");
    video.srcObject = workoutStream;
    await new Promise((r) => (video.onloadedmetadata = r));
    const canvas = $("overlay");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    drawer = new DrawingUtils(canvas.getContext("2d"));

    session.results = [];
    session.pendingResult = null;
    resetSquat();
    smoothedLms = null;
    $("rep-count").textContent = "0";
    $("rep-goal-display").textContent = ` / ${session.goalReps}`;
    $("squat-phase").textContent = "Ready";
    initRepDots();
    $("squat-status").textContent = "";
    showView("v-workout");
    $("status").textContent = "Tracking your form";
    running = true;
    lastVideoTime = -1;
    requestAnimationFrame(loop);
  } catch (err) {
    $("squat-status").textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    btn.disabled = false;
  }
});

function loop() {
  if (!running) return;
  const video = $("video");
  const canvas = $("overlay");
  const ctx = canvas.getContext("2d");
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks.length > 0) {
      $("status").textContent = "Tracking your form";
      const lms = smoothLandmarks(result.landmarks[0]);
      drawer.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, { color: "#6fb6ff", lineWidth: 3 });
      drawer.drawLandmarks(lms, { color: "#ffd54f", radius: 4 });

      const f = computeSquatFeatures(lms);
      $("knee-angle").textContent = f.kneeAngle != null ? `${f.kneeAngle.toFixed(0)}°` : "–";
      $("hip-angle").textContent = f.hipAngle != null ? `${f.hipAngle.toFixed(0)}°` : "–";
      $("trunk-lean").textContent = `${f.trunkLean.toFixed(0)}°`;
      $("foot-angle").textContent = f.footAngle == null ? "–"
        : squat.footBaseline == null ? `${f.footAngle.toFixed(0)}°`
        : `${f.footAngle.toFixed(0)}° (cal ${calibrateFeatures(f).footAngle.toFixed(0)}°)`;
      $("knee-ratio").textContent = f.kneeAnkleRatio != null ? f.kneeAnkleRatio.toFixed(2) : "–";

      const { bottomFeatures, repCompleted } = updatePhase(f, lms);
      $("hip-drop").textContent = `${(squat.lastDrop * 100).toFixed(0)}%`;
      const phaseLabel = { standing: "Standing", descending: "Going down", ascending: "Coming up" };
      $("squat-phase").textContent = phaseLabel[squat.phase] ?? squat.phase;
      $("rep-count").textContent = squat.reps;

      if (bottomFeatures) {
        const calibrated = calibrateFeatures(bottomFeatures);
        const label = classifyPosture(calibrated);
        session.pendingResult = { label, kneeAngle: bottomFeatures.kneeAngle };
        showFeedback(label);
      }
      if (repCompleted) {
        const r = session.pendingResult ?? { label: "good", kneeAngle: null };
        session.pendingResult = null;
        updateRepDot(session.results.length, r.label);
        session.results.push(r);
        if (session.results.length >= session.goalReps) {
          speak("Workout complete!");
          endWorkout();
          return;
        }
      }
    } else {
      smoothedLms = null;
      squat.baseGap = null;
      $("status").textContent = "Stand at 45° with your whole body in frame";
    }
  }
  requestAnimationFrame(loop);
}

function endWorkout() {
  running = false;
  speechSynthesis?.cancel();
  clearTimeout(feedbackTimer);
  $("feedback").classList.add("hidden");
  workoutStream?.getTracks().forEach((t) => t.stop());
  workoutStream = null;

  if (session.results.length > 0) {
    const counts = {};
    for (const r of session.results) counts[r.label] = (counts[r.label] ?? 0) + 1;
    const sessions = store.get("sessions", []);
    sessions.push({
      date: new Date().toISOString(),
      total: session.results.length,
      good: counts.good ?? 0,
      counts,
    });
    store.set("sessions", sessions);
  }
  renderReport();
  showView("v-report");
}
$("btn-end").addEventListener("click", endWorkout);
$("btn-debug").addEventListener("click", () => $("debug-panel").classList.toggle("hidden"));

// ---------- 리포트 ----------
function diagnose(counts, total) {
  const bad = (k) => counts[k] ?? 0;
  if (total > 0 && bad("good") === total) return "Every rep looked great. Ready to raise your target next time?";
  if (bad("depth") > 0 && bad("heel") > 0)
    return "Shallow depth together with heel lift often points to limited ankle mobility. Calf stretches can help.";
  if (bad("depth") > 0 && bad("back") > 0)
    return "Shallow depth with a rounded back usually means tight hips or a weak core. Try planks and hip bridges.";
  if (bad("knee") > 0)
    return "Knees caving inward usually means weak outer glutes. Clamshells are a great fix.";
  const worst = Object.entries(counts).filter(([k]) => k !== "good").sort((a, b) => b[1] - a[1])[0];
  return worst ? CLASSES[worst[0]].advice : "";
}

function renderReport() {
  const results = session.results;
  const total = results.length;
  const counts = {};
  for (const r of results) counts[r.label] = (counts[r.label] ?? 0) + 1;
  const good = counts.good ?? 0;
  const errors = Object.entries(counts).filter(([l]) => l !== "good").sort((a, b) => b[1] - a[1]);

  // 결과 요약: 결론형 피드백을 최상단에
  const summary = $("report-summary");
  let html = `<div class="rs-head">${good} of ${total} rep${total !== 1 ? "s" : ""} had good form</div>`;
  if (errors.length > 0) {
    html += `<div class="rs-issue">Main issue: <b>${CLASSES[errors[0][0]].name}</b> · occurred ${errors[0][1]} time${errors[0][1] > 1 ? "s" : ""}</div>`;
  }
  const tip = diagnose(counts, total);
  if (tip) html += `<div class="rs-tip">Try next time: ${tip}</div>`;
  summary.innerHTML = html;

  // 자세 분포
  const bar = $("dist-bar");
  const legend = $("dist-legend");
  bar.innerHTML = "";
  legend.innerHTML = "";
  const order = ["good", "depth", "back", "heel", "knee"];
  const shown = order.filter((l) => (counts[l] ?? 0) > 0);
  bar.setAttribute("aria-label", shown.map((l) => `${CLASSES[l].name}: ${counts[l]}`).join(", "));
  shown.forEach((l, i) => {
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.style.flex = counts[l];
    seg.style.background = CLASSES[l].color;
    const first = i === 0, last = i === shown.length - 1;
    seg.style.borderRadius = first && last ? "6px" : first ? "6px 0 0 6px" : last ? "0 6px 6px 0" : "0";
    bar.appendChild(seg);
  });
  for (const l of order) {
    const span = document.createElement("span");
    span.innerHTML = `<i style="background:${CLASSES[l].color}"></i>${CLASSES[l].name} ${counts[l] ?? 0}`;
    legend.appendChild(span);
  }

  // 렙별 카드: Issue와 Tip이 먼저, 각도는 보조 정보
  const list = $("rep-list");
  list.innerHTML = "";
  results.forEach((r, i) => {
    const c = CLASSES[r.label];
    const li = document.createElement("li");
    li.className = "rep-card";
    const isGood = r.label === "good";
    li.innerHTML =
      `<div class="rep-card-head"><b>Rep ${i + 1}</b><i style="background:${c.color}"></i>` +
      `<span class="rep-issue${isGood ? " good" : ""}">${isGood ? "Good form" : c.name}</span></div>` +
      (isGood ? "" : `<p class="rep-tip">Tip: ${c.advice}.</p>`) +
      (r.kneeAngle != null ? `<span class="rep-meta">Knee angle ${r.kneeAngle.toFixed(0)}°</span>` : "");
    list.appendChild(li);
  });
}
$("btn-report-done").addEventListener("click", () => { showView("v-shell"); showTab("log"); });

// ---------- 기록 탭 ----------
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function fmtDate(iso) {
  const d = new Date(iso);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
const pct = (s) => (s.total ? Math.round((s.good / s.total) * 100) : 0);

function renderLog() {
  const sessions = store.get("sessions", []);

  // 진행 요약: 좋아지고 있는지 + 계속 고쳐야 할 것
  const progressCard = $("progress-card");
  progressCard.classList.toggle("hidden", sessions.length === 0);
  if (sessions.length > 0) {
    if (sessions.length >= 2) {
      const first = pct(sessions[0]);
      const last = pct(sessions[sessions.length - 1]);
      $("progress-main").innerHTML =
        `Good-form reps: ${first}% → <span class="${last >= first ? "up" : ""}">${last}%</span>`;
    } else {
      $("progress-main").textContent = `First session: ${pct(sessions[0])}% good form`;
    }
    // 전체 세션에서 가장 잦은 오류
    const totals = {};
    for (const s of sessions) {
      for (const [k, v] of Object.entries(s.counts ?? {})) {
        if (k !== "good") totals[k] = (totals[k] ?? 0) + v;
      }
    }
    const worst = Object.entries(totals).sort((a, b) => b[1] - a[1])[0];
    $("progress-sub").textContent = worst
      ? `Needs work: ${CLASSES[worst[0]].name} — ${CLASSES[worst[0]].advice.toLowerCase()}.`
      : sessions.length >= 2 ? "No recurring issues. Keep it up!" : "Complete another session to see your trend.";
  }

  const posture = store.get("posture", null);
  const pList = $("log-posture-list");
  pList.innerHTML = "";
  $("log-posture-date").textContent = posture ? fmtDate(posture.date) : "";
  if (posture) {
    for (const item of posture.items) {
      const li = document.createElement("li");
      li.innerHTML =
        `<span class="dot ${item.status}"></span><span class="m-name">${item.name}</span>` +
        `<span class="m-desc">${item.desc}</span><span class="m-val">${item.value}${item.unit}</span>`;
      pList.appendChild(li);
    }
  } else {
    pList.innerHTML = '<li class="empty">No scans yet</li>';
  }

  const trend = $("trend-chart");
  trend.innerHTML = "";
  if (sessions.length === 0) {
    trend.innerHTML = '<p class="empty">Complete a squat session to see this</p>';
  } else {
    for (const s of sessions.slice(-7)) {
      const p = pct(s);
      const col = document.createElement("div");
      col.className = "bar-col";
      col.innerHTML =
        `<div class="bar-track"><div class="bar-fill" style="height:${p}%"></div></div>` +
        `<span class="bar-label">${p}%</span>`;
      trend.appendChild(col);
    }
  }

  const sList = $("session-list");
  sList.innerHTML = "";
  if (sessions.length === 0) {
    sList.innerHTML = '<li class="empty">No workouts yet</li>';
  } else {
    for (const s of [...sessions].reverse().slice(0, 10)) {
      const li = document.createElement("li");
      li.innerHTML = `<span>Squat · <b>${s.good} of ${s.total}</b> good reps</span><span class="s-date">${fmtDate(s.date)}</span>`;
      sList.appendChild(li);
    }
  }

  const profile = store.get("profile", {});
  const info = bmiInfo(profile.height, profile.weight);
  $("profile-line").textContent = info
    ? `Height ${profile.height} cm · Weight ${profile.weight} kg · BMI ${info.bmi} (${info.cat})`
    : "Not set";
}
$("btn-edit-profile").addEventListener("click", () => {
  const profile = store.get("profile", {});
  if (profile.height) $("in-height").value = profile.height;
  if (profile.weight) $("in-weight").value = profile.weight;
  updateBmiLine();
  showView("v-onboard");
});

// ---------- 탭/뒤로가기 ----------
document.querySelectorAll(".tab-btn").forEach((b) =>
  b.addEventListener("click", () => showTab(b.dataset.tab)));

const BACK_TARGET = {
  "v-capture": () => { stopCapture(); showView("v-shell"); showTab("home"); },
  "v-result": () => { showView("v-shell"); showTab("home"); },
  "v-exlist": () => { showView("v-shell"); showTab("home"); },
  "v-exercise": () => {
    if (exerciseOrigin === "list") { renderExerciseList(); showView("v-exlist"); }
    else { showView("v-shell"); showTab("home"); }
  },
  "v-squat-setup": () => showView("v-exercise"),
};
document.querySelectorAll("[data-back]").forEach((b) =>
  b.addEventListener("click", () => {
    const view = b.closest(".view").id;
    BACK_TARGET[view]?.();
  }));

// ---------- 부트스트랩 ----------
if (store.get("onboarded", false)) {
  showView("v-shell");
  showTab("home");
} else {
  showView("v-onboard");
}
runSplash();
getLandmarker().catch((e) => console.warn("pose model preload failed (will retry on use):", e));
loadPostureModel().then((m) => {
  postureModel = m;
  $("engine").textContent = m ? `ML (${m.trees.length} trees)` : "Rules v0";
});
initPostureML().then((ok) => console.info(`Posture engine: ${ok ? "ML" : "rules"}`));
