import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const STORAGE_KEY = "sign-tracker-browser-examples-v1";
const WINDOW_FRAMES = 30, WINDOW_MS = 1000;
let landmarker, stream, latestFrame = new Array(126).fill(0), history = [], examples = loadExamples();
let capturing = false, lastInference = 0, predictions = [], stableLabel = "", stableUntil = 0;

function loadExamples() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(saved) ? saved.filter((x) => typeof x.label === "string" && Array.isArray(x.frames) && x.frames.length === WINDOW_FRAMES) : [];
  } catch { return []; }
}
function saveExamples() { localStorage.setItem(STORAGE_KEY, JSON.stringify(examples)); renderExamples(); }
function showMessage(text, error = false) { const el = $("message"); el.textContent = text; el.classList.toggle("error", error); }

function normalizedHands(result) {
  const out = new Array(126).fill(0);
  for (let h = 0; h < Math.min(2, result.landmarks?.length || 0); h++) {
    const points = result.landmarks[h];
    const handed = result.handedness?.[h]?.[0]?.categoryName || "Left";
    // The camera preview is mirrored, so use handedness to keep the two slots consistent.
    const slot = handed.toLowerCase() === "left" ? 0 : 1;
    const wrist = points[0];
    let scale = Math.hypot(points[5].x - points[17].x, points[5].y - points[17].y);
    if (scale < 1e-4) scale = 1;
    for (let i = 0; i < 21; i++) {
      const p = points[i], offset = slot * 63 + i * 3;
      out[offset] = (p.x - wrist.x) / scale;
      out[offset + 1] = (p.y - wrist.y) / scale;
      out[offset + 2] = (p.z - wrist.z) / scale;
    }
  }
  return out;
}

function drawLandmarks(result) {
  const width = video.videoWidth, height = video.videoHeight;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  ctx.clearRect(0, 0, width, height);
  const links = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  for (const points of result.landmarks || []) {
    ctx.strokeStyle = "rgba(200,239,119,.75)"; ctx.lineWidth = Math.max(2, width / 400);
    for (const [a, b] of links) { ctx.beginPath(); ctx.moveTo(points[a].x * width, points[a].y * height); ctx.lineTo(points[b].x * width, points[b].y * height); ctx.stroke(); }
    for (let i = 0; i < points.length; i++) { ctx.beginPath(); ctx.fillStyle = i === 0 ? "#ffffff" : "#c8ef77"; ctx.arc(points[i].x * width, points[i].y * height, Math.max(2.4, width / 250), 0, Math.PI * 2); ctx.fill(); }
  }
}

function resampledWindow(now) {
  const start = now - WINDOW_MS;
  const relevant = history.filter((item) => item.t >= start && item.t <= now);
  if (relevant.length < 2 || relevant[0].t > start + 100) return null;
  const frames = [];
  for (let i = 0; i < WINDOW_FRAMES; i++) {
    const target = start + (i / (WINDOW_FRAMES - 1)) * WINDOW_MS;
    let right = relevant.findIndex((item) => item.t >= target);
    if (right < 0) right = relevant.length - 1;
    const left = Math.max(0, right - 1), a = relevant[left], b = relevant[right];
    const ratio = b.t === a.t ? 0 : Math.max(0, Math.min(1, (target - a.t) / (b.t - a.t)));
    frames.push(a.v.map((v, j) => v + (b.v[j] - v) * ratio));
  }
  return frames;
}

function classify(frames) {
  if (!examples.length) return null;
  const handPresent = (frame, slot) => {
    const start = slot * 63;
    for (let i = start; i < start + 63; i++) if (Math.abs(frame[i]) > 1e-5) return true;
    return false;
  };
  const candidateDistances = examples.map((example) => {
    let total = 0, count = 0;
    for (let f = 0; f < WINDOW_FRAMES; f++) {
      const a = frames[f], b = example.frames[f];
      for (let j = 0; j < 126; j++) {
        // Ignore empty hand slots; still penalize a missing hand lightly.
        const handOffset = j < 63 ? 0 : 63;
        const presentA = handPresent(a, handOffset / 63);
        const presentB = handPresent(b, handOffset / 63);
        if (!presentA && !presentB) continue;
        if (presentA !== presentB) { total += 0.20; count++; continue; }
        const delta = a[j] - b[j]; total += delta * delta; count++;
      }
    }
    return { label: example.label, distance: count ? total / count : Infinity };
  }).sort((a, b) => a.distance - b.distance);
  const best = candidateDistances[0];
  const confidence = Math.exp(-best.distance / 0.24);
  return { ...best, confidence };
}

function recognize(now) {
  const frames = resampledWindow(now);
  if (!frames) return;
  const result = classify(frames);
  if (!result || result.confidence < 0.42 || result.label === "idle") {
    predictions = []; $("recognized").classList.remove("show"); return;
  }
  predictions.push(result.label); if (predictions.length > 5) predictions.shift();
  const counts = predictions.reduce((acc, label) => (acc[label] = (acc[label] || 0) + 1, acc), {});
  if (counts[result.label] >= 3) { stableLabel = result.label; stableUntil = now + 1100; }
  if (stableLabel && now < stableUntil) { $("recognizedLabel").textContent = stableLabel; $("recognized").classList.add("show"); }
}

async function frameLoop() {
  if (!stream) return;
  if (video.readyState >= 2 && landmarker) {
    const now = performance.now();
    if (now - lastInference > 30) {
      lastInference = now;
      try {
        const result = landmarker.detectForVideo(video, now);
        latestFrame = normalizedHands(result); drawLandmarks(result);
        history.push({ t: now, v: latestFrame });
        while (history.length && history[0].t < now - 2500) history.shift();
        if (!capturing) recognize(now);
      } catch (error) { showMessage(`Hand tracking stopped: ${error.message}`, true); }
    }
  }
  requestAnimationFrame(frameLoop);
}

async function startCamera() {
  $("cameraButton").disabled = true;
  showMessage("Loading hand tracking model…");
  try {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.55, minHandPresenceConfidence: 0.55, minTrackingConfidence: 0.5,
    });
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await new Promise((resolve) => video.onloadedmetadata = resolve);
    await video.play();
    $("videoPlaceholder").style.display = "none";
    $("cameraStatus").classList.add("active"); $("cameraStatus").innerHTML = "<i></i> CAMERA LIVE";
    $("cameraButton").textContent = "Stop camera"; $("cameraButton").disabled = false;
    $("recordButton").disabled = false;
    showMessage("Camera is ready. Add labeled examples to teach a sign.");
    requestAnimationFrame(frameLoop);
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop()); stream = null;
    const reason = error.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access in your browser’s site settings." : error.message || "Could not start the camera.";
    showMessage(reason, true); $("cameraButton").disabled = false; $("cameraButton").textContent = "Enable camera ↗";
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null; video.srcObject = null; ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("videoPlaceholder").style.display = "flex"; $("cameraStatus").classList.remove("active"); $("cameraStatus").innerHTML = "<i></i> CAMERA OFF";
  $("cameraButton").textContent = "Enable camera ↗"; $("recordButton").disabled = true; $("recognized").classList.remove("show");
  showMessage("Camera stopped.");
}

async function recordExample() {
  const label = $("labelInput").value.trim().toLocaleLowerCase();
  if (!label || !stream || capturing) return;
  capturing = true; predictions = [];
  const start = performance.now(), frames = [];
  $("recordButton").disabled = true; $("recordButton").textContent = "Recording…";
  showMessage(`Recording “${label}” — hold or perform the sign now.`);
  await new Promise((resolve) => {
    const collect = () => {
      const elapsed = performance.now() - start;
      $("progress").style.width = `${Math.min(100, elapsed / WINDOW_MS * 100)}%`;
      if (elapsed >= WINDOW_MS) { frames.push([...latestFrame]); resolve(); return; }
      frames.push([...latestFrame]); requestAnimationFrame(collect);
    };
    requestAnimationFrame(collect);
  });
  const sampled = [];
  for (let i = 0; i < WINDOW_FRAMES; i++) sampled.push(frames[Math.min(frames.length - 1, Math.floor(i * frames.length / WINDOW_FRAMES))]);
  if (!sampled.some((frame) => frame.some((value) => value !== 0))) {
    showMessage("No hand was detected. Keep your hand in view and try again.", true);
  } else {
    examples.push({ label, frames: sampled });
    try {
      saveExamples();
      showMessage(`Saved example for “${label}”. Record a few more variations for better results.`);
    } catch {
      examples.pop();
      showMessage("The browser ran out of local storage. Clear some examples in this browser and try again.", true);
    }
  }
  $("progress").style.width = "0%"; capturing = false; $("recordButton").disabled = false; $("recordButton").innerHTML = '<span class="record-dot"></span> Record 1 sec';
}

function renderExamples() {
  const count = examples.length;
  $("exampleCount").textContent = `${count} EXAMPLE${count === 1 ? "" : "S"}`;
  $("clearButton").disabled = !count;
  const list = $("exampleList");
  if (!count) { list.innerHTML = '<div class="empty-state">Your recorded sign examples<br>will appear here.</div>'; return; }
  const counts = examples.reduce((acc, x) => (acc[x.label] = (acc[x.label] || 0) + 1, acc), {});
  list.innerHTML = Object.entries(counts).map(([label, n]) => `<div class="example"><b>${escapeHTML(label)}</b><small>${n} EXAMPLE${n === 1 ? "" : "S"}</small><button class="remove-label" data-remove="${escapeHTML(label)}" aria-label="Delete all examples for ${escapeHTML(label)}" title="Delete this label and its examples">×</button></div>`).join("");
}
function escapeHTML(value) { return value.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]); }

$("cameraButton").addEventListener("click", () => stream ? stopCamera() : startCamera());
$("recordButton").addEventListener("click", recordExample);
$("clearButton").addEventListener("click", () => { examples = []; localStorage.removeItem(STORAGE_KEY); renderExamples(); showMessage("Examples cleared from this device."); });
$("exampleList").addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove]");
  if (!button) return;
  const label = button.dataset.remove;
  examples = examples.filter((example) => example.label !== label);
  saveExamples();
  showMessage(`Deleted “${label}” and its recorded examples.`);
});
document.querySelectorAll("[data-label]").forEach((button) => button.addEventListener("click", () => $("labelInput").value = button.dataset.label));
$("labelInput").addEventListener("keydown", (event) => { if (event.key === "Enter" && !$("recordButton").disabled) recordExample(); });
window.addEventListener("beforeunload", () => stream?.getTracks().forEach((track) => track.stop()));
renderExamples();
