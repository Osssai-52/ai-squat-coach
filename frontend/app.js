// 핏폼: 체형 분석 → 맞춤 운동 추천 → 스쿼트 실시간 자세 교정
import { getLandmarker, computeSquatFeatures, PoseLandmarker, DrawingUtils } from "./pose.js";
import { validateFrame, analyzePosture, buildRecommendations, EXERCISES } from "./posture.js";
import { loadPostureModel, predictPosture } from "./inference.js";

// ---------- 스쿼트 판정 설정 (팀 실측으로 튜닝) ----------
const CONFIG = {
  DEPTH_KNEE_ANGLE: 100,
  TRUNK_LEAN_MAX: 50,
  HEEL_FOOT_ANGLE: 25,
  KNEE_RATIO_MIN: 0.7,
  STANDING_KNEE_ANGLE: 160,
  BOTTOM_ENTER_DELTA: 5,
  LANDMARK_ALPHA: 0.4,
};

const CLASSES = {
  good:  { name: "정상",          banner: "좋은 자세예요!",        color: "var(--c-good)",
           advice: "" },
  depth: { name: "깊이 부족",     banner: "조금 더 앉아볼까요?",   color: "var(--c-depth)",
           advice: "허벅지가 수평이 될 때까지 내려가 보세요" },
  back:  { name: "허리 굽음",     banner: "가슴을 펴 주세요!",     color: "var(--c-back)",
           advice: "시선을 정면에 두고 가슴을 열면 허리가 펴져요" },
  heel:  { name: "발뒤꿈치 들림", banner: "뒤꿈치를 붙여 주세요!", color: "var(--c-heel)",
           advice: "무게중심을 발 중앙~뒤꿈치에 두세요" },
  knee:  { name: "무릎 모임",     banner: "무릎을 벌려 주세요!",   color: "var(--c-knee)",
           advice: "무릎이 발끝과 같은 방향을 향하게 하세요" },
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
const VIEWS = ["v-onboard", "v-shell", "v-capture", "v-result", "v-exercise", "v-squat-setup", "v-workout", "v-report"];
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
  const cat = bmi < 18.5 ? "저체중" : bmi < 23 ? "정상" : bmi < 25 ? "과체중" : "비만";
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
  const profile = store.get("profile", {});
  const info = bmiInfo(profile.height, profile.weight);
  $("home-sub").textContent = info
    ? `BMI ${info.bmi} (${info.cat}) · 체형에 맞는 운동을 추천해요`
    : "체형을 분석하고 맞춤 운동을 받아보세요";

  const posture = store.get("posture", null);
  $("posture-cta").classList.toggle("hidden", !!posture);
  $("posture-summary").classList.toggle("hidden", !posture);
  $("reco-section").classList.toggle("hidden", !posture);
  if (!posture) return;

  const chips = $("summary-chips");
  chips.innerHTML = "";
  for (const item of posture.items) {
    const pill = document.createElement("span");
    pill.className = `pill ${item.status}`;
    pill.textContent = `${item.name} ${item.status === "ok" ? "양호" : "주의"}`;
    chips.appendChild(pill);
  }

  const list = $("reco-list");
  list.innerHTML = "";
  for (const pick of buildRecommendations(posture.items)) {
    const ex = EXERCISES[pick.key];
    const card = document.createElement("button");
    card.className = "reco-card";
    card.innerHTML =
      `<div class="t"><b>${ex.name}${ex.live ? ' <span class="live-badge">실시간 분석</span>' : ""}</b>` +
      `<p>${pick.reason}</p></div><span class="arrow">›</span>`;
    card.addEventListener("click", () => openExercise(pick.key));
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
    toast($("cap-toast"), `카메라를 열 수 없어요: ${err.message}`, 3000);
  }
}
function stopCapture() {
  capture.stream?.getTracks().forEach((t) => t.stop());
  capture.stream = null;
}
function setCaptureStep() {
  const front = capture.step === "front";
  $("capture-title").textContent = `체형 분석 ${front ? "1" : "2"}/2`;
  $("guide-front").classList.toggle("hidden", !front);
  $("guide-side").classList.toggle("hidden", front);
  $("cap-hint").textContent = front
    ? "가이드 선에 맞춰 정면으로 서 주세요"
    : "이번엔 옆으로 돌아서 주세요";
  $("cap-desc").textContent = front
    ? "발끝부터 머리까지 전신이 보여야 해요"
    : "완전한 옆모습으로, 전신이 보이게 서 주세요";
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
      toast($("cap-toast"), "좋아요! 이제 옆모습을 찍을게요");
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
  $("result-hero").innerHTML = warns.length === 0
    ? `<div class="big-num good">모두 양호</div><p>측정한 ${items.length}개 항목이 정상 범위예요</p>`
    : `<div class="big-num warn">주의 ${warns.length}개</div><p>${warns.map((w) => w.name).join(" · ")} 항목을 관리해 보세요</p>`;
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

$("btn-analyze").addEventListener("click", startCapture);
$("btn-reanalyze").addEventListener("click", startCapture);
$("btn-to-reco").addEventListener("click", () => { showView("v-shell"); showTab("home"); });

// ---------- 운동 상세 ----------
let currentExercise = null;
function openExercise(key) {
  currentExercise = key;
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

const squat = { phase: "standing", prevKnee: null, minKneeThisRep: 999, bottomFeatures: null, reps: 0 };
function resetSquat() {
  Object.assign(squat, { phase: "standing", prevKnee: null, minKneeThisRep: 999, bottomFeatures: null, reps: 0 });
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

function updatePhase(f) {
  const k = f.kneeAngle;
  if (k == null) return { bottomFeatures: null, repCompleted: false };
  const delta = squat.prevKnee == null ? 0 : k - squat.prevKnee;
  squat.prevKnee = k;
  let bottomFeatures = null, repCompleted = false;
  switch (squat.phase) {
    case "standing":
      if (k < CONFIG.STANDING_KNEE_ANGLE - 10) {
        squat.phase = "descending"; squat.minKneeThisRep = k; squat.bottomFeatures = null;
      }
      break;
    case "descending":
      if (k < squat.minKneeThisRep) { squat.minKneeThisRep = k; squat.bottomFeatures = { ...f }; }
      if (delta > CONFIG.BOTTOM_ENTER_DELTA) { squat.phase = "ascending"; bottomFeatures = squat.bottomFeatures; }
      break;
    case "ascending":
      if (k >= CONFIG.STANDING_KNEE_ANGLE) { squat.phase = "standing"; squat.reps += 1; repCompleted = true; }
      if (delta < -CONFIG.BOTTOM_ENTER_DELTA) squat.phase = "descending";
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
  u.lang = "ko-KR";
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
    $("squat-status").textContent = "모델 준비 중…";
    landmarker = await getLandmarker();
    $("squat-status").textContent = "카메라 연결 중…";
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
    $("squat-phase").textContent = "준비";
    initRepDots();
    $("squat-status").textContent = "";
    showView("v-workout");
    $("status").textContent = "동작 인식 중";
    running = true;
    lastVideoTime = -1;
    requestAnimationFrame(loop);
  } catch (err) {
    $("squat-status").textContent = `오류: ${err.message}`;
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
      $("status").textContent = "동작 인식 중";
      const lms = smoothLandmarks(result.landmarks[0]);
      drawer.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, { color: "#6fb6ff", lineWidth: 3 });
      drawer.drawLandmarks(lms, { color: "#ffd54f", radius: 4 });

      const f = computeSquatFeatures(lms);
      $("knee-angle").textContent = f.kneeAngle != null ? `${f.kneeAngle.toFixed(0)}°` : "–";
      $("hip-angle").textContent = f.hipAngle != null ? `${f.hipAngle.toFixed(0)}°` : "–";
      $("trunk-lean").textContent = `${f.trunkLean.toFixed(0)}°`;
      $("foot-angle").textContent = f.footAngle != null ? `${f.footAngle.toFixed(0)}°` : "–";
      $("knee-ratio").textContent = f.kneeAnkleRatio != null ? f.kneeAnkleRatio.toFixed(2) : "–";

      const { bottomFeatures, repCompleted } = updatePhase(f);
      const phaseKo = { standing: "서 있음", descending: "내려가는 중", ascending: "올라오는 중" };
      $("squat-phase").textContent = phaseKo[squat.phase] ?? squat.phase;
      $("rep-count").textContent = squat.reps;

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
      smoothedLms = null;
      squat.prevKnee = null;
      $("status").textContent = "전신이 45° 각도로 화면에 들어오게 서 주세요";
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
  if (total > 0 && bad("good") === total) return "전부 정상 자세였어요. 다음엔 목표 횟수를 늘려볼까요?";
  if (bad("depth") > 0 && bad("heel") > 0)
    return "깊이 부족과 뒤꿈치 들림이 함께 나타나면 발목 유연성이 부족한 경우가 많아요. 카프 스트레칭을 추천해요.";
  if (bad("depth") > 0 && bad("back") > 0)
    return "깊이 부족과 허리 굽음이 함께 나타나면 고관절 유연성이나 코어 힘이 부족한 경우가 많아요. 플랭크와 힙 브릿지를 추천해요.";
  if (bad("knee") > 0)
    return "무릎이 안쪽으로 모이는 습관은 엉덩이 옆 근육이 약할 때 자주 나타나요. 클램쉘 운동을 추천해요.";
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
    ? `${CLASSES[errors[0][0]].name} <small>${errors[0][1]}회</small>` : "없음 👍";

  const bar = $("dist-bar");
  const legend = $("dist-legend");
  bar.innerHTML = "";
  legend.innerHTML = "";
  const order = ["good", "depth", "back", "heel", "knee"];
  const shown = order.filter((l) => (counts[l] ?? 0) > 0);
  bar.setAttribute("aria-label", shown.map((l) => `${CLASSES[l].name} ${counts[l]}회`).join(", "));
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
    const angle = r.kneeAngle != null ? `무릎 ${r.kneeAngle.toFixed(0)}°` : "";
    const lab = r.label === "good" ? "정상" : `${CLASSES[r.label].name} — ${CLASSES[r.label].advice}`;
    li.innerHTML = `<span class="n">${i + 1}회</span><i style="background:${CLASSES[r.label].color}"></i>` +
      `<span class="lab">${lab}</span><span class="ang">${angle}</span>`;
    list.appendChild(li);
  });
}
$("btn-report-done").addEventListener("click", () => { showView("v-shell"); showTab("log"); });

// ---------- 기록 탭 ----------
function fmtDate(iso) {
  const d = new Date(iso);
  return `${d.getMonth() + 1}월 ${d.getDate()}일 ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
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
    pList.innerHTML = '<li class="empty">아직 분석 기록이 없어요</li>';
  }

  const sessions = store.get("sessions", []);
  const trend = $("trend-chart");
  trend.innerHTML = "";
  if (sessions.length === 0) {
    trend.innerHTML = '<p class="empty">스쿼트 세션을 완료하면 표시돼요</p>';
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
    sList.innerHTML = '<li class="empty">아직 운동 기록이 없어요</li>';
  } else {
    for (const s of [...sessions].reverse().slice(0, 10)) {
      const li = document.createElement("li");
      li.innerHTML = `<span>스쿼트 ${s.total}회 중 <b>${s.good}회 정상</b></span><span class="s-date">${fmtDate(s.date)}</span>`;
      sList.appendChild(li);
    }
  }

  const profile = store.get("profile", {});
  const info = bmiInfo(profile.height, profile.weight);
  $("profile-line").textContent = info
    ? `키 ${profile.height}cm · 몸무게 ${profile.weight}kg · BMI ${info.bmi} (${info.cat})`
    : "미입력";
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
  "v-exercise": () => { showView("v-shell"); showTab("home"); },
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
getLandmarker().catch((e) => console.warn("포즈 모델 사전 로딩 실패(사용 시 재시도):", e));
loadPostureModel().then((m) => {
  postureModel = m;
  $("engine").textContent = m ? `ML (트리 ${m.trees.length}개)` : "규칙 기반";
});
