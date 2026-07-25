// 핏폼: 체형 분석 → 맞춤 운동 추천 → 스쿼트 실시간 자세 교정
import { getLandmarker, computeSquatFeatures, PoseLandmarker, DrawingUtils, LM } from "./pose.js";
import { validateFrame, analyzePosture, buildRecommendations, EXERCISES, initPostureML, isPostureML } from "./posture.js";
import { loadPostureModel, predictPosture } from "./inference.js";
import { initGoogleSignIn, googleSignOut } from "./auth.js";

// ---------- 스쿼트 판정 설정 (팀 실측으로 튜닝) ----------
// 임계값은 45° 실측 분포로 보정 (ml/train.py 상수와 동일 유지)
const CONFIG = {
  DEPTH_KNEE_ANGLE: 118,
  TRUNK_LEAN_MAX: 45,
  HEEL_FOOT_ANGLE: 34,
  KNEE_RATIO_MIN: 0.65,
  // 렙 감지: 엉덩이 하강 비율(서 있을 때 대비, 다리 길이로 정규화) 기준
  // 무릎 각도는 45° 라이브에서 오판이 많아(몸 돌리기·걷기에도 급변) 렙 감지엔 쓰지 않음
  DROP_ENTER: 0.10,        // 이만큼 내려가면 "내려가는 중" 시작
  DROP_EXIT: 0.06,         // 이 아래로 올라오면 기립(렙 종료)
  REP_MIN_DROP: 0.20,      // 최저점 하강이 이보다 얕으면 렙 미인정 (풀스쿼트 ~0.4, 하프 ~0.25)
  BOTTOM_REBOUND: 0.02,    // 최저점에서 이만큼 반등하면 "올라오는 중"
  STAND_STILL_DROP: 0.05,  // 이 이하일 때만 기준선(엉덩이 높이·다리 길이) 갱신
  LANDMARK_ALPHA: 0.4,
  // 발 각도 세션 캘리브레이션: 서 있을 때 기준선을 잡고, 판정 시
  // "학습 사진의 평발 평균(26°)" 기준으로 환산 — 카메라 세팅 차이에 의한 heel 오탐 방지
  FOOT_ANGLE_REF: 26,
  BASELINE_ALPHA: 0.1,
};

const CLASSES = {
  good:  { name: "Good form",     banner: "Great form!",           color: "var(--c-good)",
           advice: "" },
  depth: { name: "Shallow depth", banner: "Go a little deeper!",   color: "var(--c-depth)",
           advice: "Sink down until your thighs are parallel to the floor" },
  back:  { name: "Rounded back",  banner: "Chest up!",             color: "var(--c-back)",
           advice: "Look forward and open your chest to keep your back straight" },
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

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const VIEWS = ["v-login", "v-onboard", "v-shell", "v-capture", "v-result", "v-exlist", "v-exercise", "v-squat-setup", "v-workout", "v-report"];
function showView(id) {
  for (const v of VIEWS) $(v).classList.toggle("hidden", v !== id);
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
function renderHome() {
  const user = store.get("user", null);
  const firstName = user?.name ? user.name.split(" ")[0] : "";
  $("home-greeting").textContent = firstName ? `Let's get moving, ${firstName}` : "Let's get moving";

  const profile = store.get("profile", {});
  const info = bmiInfo(profile.height, profile.weight);
  $("home-sub").textContent = info
    ? `BMI ${info.bmi} (${info.cat}) · workouts tailored to your body`
    : "Scan your posture and get workouts made for you";

  const posture = store.get("posture", null);
  $("tab-home").classList.toggle("home-empty", !posture); // 분석 전엔 진입 카드 2개가 화면을 채움
  $("card-scan").classList.toggle("hidden", !!posture);   // 분석 후엔 요약+추천이 스캔 카드를 대체
  $("posture-summary").classList.toggle("hidden", !posture);
  $("reco-section").classList.toggle("hidden", !posture);
  if (!posture) return;

  const chips = $("summary-chips");
  chips.innerHTML = "";
  for (const item of posture.items) {
    const pill = document.createElement("span");
    pill.className = `pill ${item.status}`;
    pill.textContent = `${item.name}: ${item.status === "ok" ? "OK" : "Watch"}`;
    chips.appendChild(pill);
  }

  const list = $("reco-list");
  list.innerHTML = "";
  for (const pick of buildRecommendations(posture.items)) {
    const ex = EXERCISES[pick.key];
    const card = document.createElement("button");
    card.className = "reco-card";
    card.innerHTML =
      `<div class="t"><b>${ex.name}${ex.live ? ' <span class="live-badge">LIVE</span>' : ""}</b>` +
      `<p>${pick.reason}</p></div><span class="arrow">›</span>`;
    card.addEventListener("click", () => openExercise(pick.key, "home"));
    list.appendChild(card);
  }
}

// ---------- 운동 라이브러리 (바로 운동하기 경로) ----------
function renderExerciseList() {
  const list = $("exlist");
  list.innerHTML = "";
  for (const [key, ex] of Object.entries(EXERCISES)) {
    const card = document.createElement("button");
    card.className = "reco-card";
    card.innerHTML =
      `<div class="t"><b>${ex.name}${ex.live ? ' <span class="live-badge">LIVE</span>' : ""}</b>` +
      `<p>${ex.tag}</p></div><span class="arrow">›</span>`;
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
let exerciseOrigin = "home"; // "home"(추천 목록에서) | "list"(운동 라이브러리에서) — 뒤로가기 목적지
function openExercise(key, origin = "home") {
  currentExercise = key;
  exerciseOrigin = origin;
  const ex = EXERCISES[key];
  $("ex-title").textContent = ex.name;
  $("ex-tag").textContent = ex.tag;
  $("ex-summary").textContent = ex.summary;
  const steps = $("ex-steps");
  steps.innerHTML = "";
  for (const s of ex.steps) {
    const li = document.createElement("li");
    li.textContent = s;
    steps.appendChild(li);
  }
  $("ex-dose").textContent = ex.dose;
  $("btn-live").classList.toggle("hidden", !ex.live);
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
  baseHipY: null,      // 서 있을 때 엉덩이 높이 (EMA)
  legLen: null,        // 서 있을 때 엉덩이-발목 거리 (EMA, 정규화용)
  maxDropThisRep: 0,   // 이번 렙 최대 하강 비율
  bottomFeatures: null, // 최저점 프레임 피처 (여기서만 자세 판정)
  footBaseline: null,
  lastDrop: 0,          // 디버그 표시용
};
function resetSquat() {
  Object.assign(squat, {
    phase: "standing", reps: 0, baseHipY: null, legLen: null,
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

// lms(스무딩된 랜드마크)에서 엉덩이 하강 비율 계산 + 서 있을 때 기준선 갱신
function computeDrop(lms) {
  const hipY = (lms[LM.L_HIP].y + lms[LM.R_HIP].y) / 2;
  const ankleY = (lms[LM.L_ANKLE].y + lms[LM.R_ANKLE].y) / 2;
  const legLen = Math.max(Math.abs(ankleY - hipY), 1e-3);

  if (squat.baseHipY == null || squat.legLen == null) {
    squat.baseHipY = hipY;
    squat.legLen = legLen;
    return 0;
  }
  const drop = (hipY - squat.baseHipY) / squat.legLen;
  // 거의 서 있는 상태에서만 기준선 갱신 (앉는 중·이동 중 오염 방지)
  if (squat.phase === "standing" && Math.abs(drop) < CONFIG.STAND_STILL_DROP) {
    squat.baseHipY += 0.15 * (hipY - squat.baseHipY);
    squat.legLen += 0.15 * (legLen - squat.legLen);
  }
  return drop;
}

function updatePhase(f, lms) {
  const drop = computeDrop(lms);
  squat.lastDrop = drop;
  let bottomFeatures = null, repCompleted = false;
  // 충분히 앉았는지 (얕은 까딱임·잡동작은 렙/판정 대상이 아님)
  const deepEnough = () => squat.maxDropThisRep >= CONFIG.REP_MIN_DROP;

  switch (squat.phase) {
    case "standing":
      updateFootBaseline(f); // 서 있는 동안 발 각도 기준선 갱신
      if (drop > CONFIG.DROP_ENTER) {
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
        if (deepEnough()) bottomFeatures = squat.bottomFeatures; // 얕으면 판정 생략
      }
      break;
    case "ascending":
      if (drop > squat.maxDropThisRep) squat.phase = "descending"; // 다시 내려감 (불완전 렙)
      else if (drop < CONFIG.DROP_EXIT) {
        squat.phase = "standing";
        if (deepEnough()) { squat.reps += 1; repCompleted = true; } // 얕으면 렙 미인정
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
      squat.baseHipY = null; // 사람을 놓치면 기준선 재수집
      squat.legLen = null;
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

  // 세션 저장
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

  $("report-good").textContent = counts.good ?? 0;
  $("report-total").textContent = total;

  const errors = Object.entries(counts).filter(([l]) => l !== "good").sort((a, b) => b[1] - a[1]);
  $("report-top-error").innerHTML = errors.length
    ? `${CLASSES[errors[0][0]].name} <small>×${errors[0][1]}</small>` : "None 👍";

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

  const diag = diagnose(counts, total);
  $("diagnosis").classList.toggle("hidden", !diag);
  $("diagnosis-text").textContent = diag;

  const list = $("rep-list");
  list.innerHTML = "";
  results.forEach((r, i) => {
    const li = document.createElement("li");
    const angle = r.kneeAngle != null ? `Knee ${r.kneeAngle.toFixed(0)}°` : "";
    const lab = r.label === "good" ? "Good form" : `${CLASSES[r.label].name} — ${CLASSES[r.label].advice}`;
    li.innerHTML = `<span class="n">#${i + 1}</span><i style="background:${CLASSES[r.label].color}"></i>` +
      `<span class="lab">${lab}</span><span class="ang">${angle}</span>`;
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

function renderLog() {
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

  const sessions = store.get("sessions", []);
  const trend = $("trend-chart");
  trend.innerHTML = "";
  if (sessions.length === 0) {
    trend.innerHTML = '<p class="empty">Complete a squat session to see this</p>';
  } else {
    for (const s of sessions.slice(-7)) {
      const pct = s.total ? Math.round((s.good / s.total) * 100) : 0;
      const col = document.createElement("div");
      col.className = "bar-col";
      col.innerHTML =
        `<div class="bar-track"><div class="bar-fill" style="height:${pct}%"></div></div>` +
        `<span class="bar-label">${pct}%</span>`;
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

  // 계정 정보 표시
  const user = store.get("user", null);
  $("account-row").classList.toggle("hidden", !user);
  if (user) {
    $("account-name").textContent = user.name || "Google account";
    $("account-email").textContent = user.email || "";
    const avatar = $("account-avatar");
    if (user.picture) { avatar.src = user.picture; avatar.style.display = ""; }
    else avatar.style.display = "none";
  }
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
    if (exerciseOrigin === "list") showView("v-exlist");
    else { showView("v-shell"); showTab("home"); }
  },
  "v-squat-setup": () => showView("v-exercise"),
};
document.querySelectorAll("[data-back]").forEach((b) =>
  b.addEventListener("click", () => {
    const view = b.closest(".view").id;
    BACK_TARGET[view]?.();
  }));

// ---------- 로그인 ----------
function enterApp() {
  if (store.get("onboarded", false)) {
    showView("v-shell");
    showTab("home");
  } else {
    showView("v-onboard");
  }
}

function onSignedIn(user) {
  store.set("user", user);
  enterApp();
}

function showLogin() {
  showView("v-login");
  const configured = initGoogleSignIn($("gsi-btn"), onSignedIn);
  if (!configured) {
    // 클라이언트 ID 미설정: 안내 + 개발용 우회 버튼 표시
    $("login-note").classList.remove("hidden");
    $("btn-dev-skip").classList.remove("hidden");
  }
}

$("btn-dev-skip").addEventListener("click", () =>
  onSignedIn({ name: "Guest", email: "", picture: "" }));

$("btn-signout").addEventListener("click", () => {
  googleSignOut();
  localStorage.removeItem("fitform:user");
  showLogin();
});

// ---------- 부트스트랩 ----------
// 영어 UI 전환 마이그레이션: 이전 버전이 저장한 한국어 분석 결과 제거 (1회)
if (!store.get("migratedEn", false)) {
  localStorage.removeItem("fitform:posture");
  store.set("migratedEn", true);
}
if (store.get("user", null)) {
  enterApp();
} else {
  showLogin();
}
getLandmarker().catch((e) => console.warn("포즈 모델 사전 로딩 실패(사용 시 재시도):", e));
loadPostureModel().then((m) => {
  postureModel = m;
  $("engine").textContent = m ? `ML (${m.trees.length} trees)` : "Rules v0";
});
initPostureML().then((ok) => console.info(`Posture engine: ${ok ? "ML" : "rules"}`));
