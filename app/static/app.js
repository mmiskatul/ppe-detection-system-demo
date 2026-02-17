const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");

const imageFileInput = document.getElementById("image-file");
const detectImageBtn = document.getElementById("detect-image-btn");
const imageOriginal = document.getElementById("image-original");
const imageAnnotated = document.getElementById("image-annotated");

const videoFileInput = document.getElementById("video-file");
const detectVideoBtn = document.getElementById("detect-video-btn");
const videoPreview = document.getElementById("video-preview");
const videoSummary = document.getElementById("video-summary");

const video = document.getElementById("video-live");
const overlay = document.getElementById("overlay-live");
const captureBtn = document.getElementById("capture-btn");
const capturedFrameCanvas = document.getElementById("captured-frame");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start-btn");
const transportSelect = document.getElementById("transport");
const confInput = document.getElementById("conf");
const iouInput = document.getElementById("iou");
const confValue = document.getElementById("conf-value");
const iouValue = document.getElementById("iou-value");
// const legendEl = document.getElementById("legend");

const overlayCtx = overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { alpha: false, desynchronized: true });
const FRAME_INTERVAL_MS = 90;
const MAX_CAPTURE_WIDTH = 640;
const JPEG_QUALITY = 0.6;

let ws = null;
let sio = null;
let classes = [];
let chart = null;
let sending = false;
let inFlight = false;
let imageObj = null;
let latestRealtimeDetections = [];
let latestRealtimeSize = { w: 0, h: 0 };

function updateSliderText() {
  confValue.textContent = Number(confInput.value).toFixed(2);
  iouValue.textContent = Number(iouInput.value).toFixed(2);
}

confInput.addEventListener("input", updateSliderText);
iouInput.addEventListener("input", updateSliderText);
updateSliderText();

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.target).classList.add("active");
  });
});

async function loadClasses() {
  const resp = await fetch("/classes");
  const data = await resp.json();
  classes = data.classes;
  renderLegend();
  initChart();
}

function renderLegend() {
  legendEl.innerHTML = "";
  classes.forEach((c) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span class="swatch" style="background:${c.color}"></span><span>${c.class_name}</span>`;
    legendEl.appendChild(item);
  });
}

// function initChart() {
//   const ctx = document.getElementById("classChart");
//   chart = new Chart(ctx, {
//     type: "bar",
//     data: {
//       labels: classes.map((c) => c.class_name),
//       datasets: [
//         {
//           label: "Live Count",
//           data: classes.map(() => 0),
//           backgroundColor: classes.map((c) => c.color),
//           borderWidth: 0,
//         },
//       ],
//     },
//     options: {
//       responsive: true,
//       animation: false,
//       scales: {
//         y: {
//           beginAtZero: true,
//           ticks: { stepSize: 1 },
//         },
//       },
//       plugins: {
//         legend: { display: false },
//       },
//     },
//   });
// }

function updateChart(counts) {
  if (!chart) return;
  chart.data.datasets[0].data = classes.map((c) => counts[c.class_name] || 0);
  chart.update();
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function toDisplayItemName(rawClass) {
  const name = rawClass.toLowerCase().trim();
  const mapping = {
    boots: "Boot",
    gloves: "Gloves",
    glasses: "Glasses",
    helmet: "Helmet",
    vest: "Vest",
    "no boots": "Boot",
    "no gloves": "Gloves",
    "no glasses": "Glasses",
    "no helmet": "Helmet",
    "no vest": "Vest",
  };
  return mapping[name] || rawClass;
}

function uniqueList(arr) {
  return [...new Set(arr)];
}

function buildReferenceSummary(detections) {
  const violations = [];
  const positives = [];
  detections.forEach((det) => {
    const className = det.class_name.toLowerCase().trim();
    if (className.startsWith("no ")) {
      violations.push(toDisplayItemName(className));
    } else if (className !== "person") {
      positives.push(toDisplayItemName(className));
    }
  });

  const badList = uniqueList(violations);
  const goodList = uniqueList(positives);
  const isAlert = badList.length > 0;
  const list = isAlert ? badList : goodList;
  const prefix = isAlert ? "No " : "";

  return {
    isAlert,
    color: isAlert ? "#d81e06" : "#11c784",
    icon: isAlert ? "⚠" : "🛡",
    text: list.length ? `${prefix}${list.join(", ")} Detected` : "No PPE Class Detected",
  };
}

function choosePrimaryBBox(detections) {
  if (!detections.length) return null;
  const personDetections = detections.filter((d) => d.class_name.toLowerCase() === "person");
  if (personDetections.length) {
    return personDetections.reduce((a, b) => (a.confidence > b.confidence ? a : b)).bbox;
  }
  return detections.reduce(
    (acc, d) => ({
      x1: Math.min(acc.x1, d.bbox.x1),
      y1: Math.min(acc.y1, d.bbox.y1),
      x2: Math.max(acc.x2, d.bbox.x2),
      y2: Math.max(acc.y2, d.bbox.y2),
    }),
    { x1: Number.POSITIVE_INFINITY, y1: Number.POSITIVE_INFINITY, x2: 0, y2: 0 },
  );
}

function drawReferenceOverlay(ctx, detections, scaleX, scaleY) {
  if (!detections.length) return;
  const bbox = choosePrimaryBBox(detections);
  if (!bbox) return;

  const x1 = bbox.x1 * scaleX;
  const y1 = bbox.y1 * scaleY;
  const x2 = bbox.x2 * scaleX;
  const y2 = bbox.y2 * scaleY;
  const w = x2 - x1;
  const h = y2 - y1;

  const summary = buildReferenceSummary(detections);
  ctx.strokeStyle = summary.color;
  ctx.lineWidth = 3;
  drawRoundedRect(ctx, x1, y1, w, h, 20);
  ctx.stroke();

  const text = `${summary.icon} ${summary.text}`;
  ctx.font = "600 24px Trebuchet MS";
  const padX = 16;
  const bubbleH = 46;
  const rawW = ctx.measureText(text).width + padX * 2;
  const bubbleW = Math.min(Math.max(rawW, 240), ctx.canvas.width - 20);
  const bubbleX = Math.max(10, Math.min(x1, ctx.canvas.width - bubbleW - 10));
  const bubbleY = Math.max(10, y1 - bubbleH / 2);

  ctx.fillStyle = summary.color;
  drawRoundedRect(ctx, bubbleX, bubbleY, bubbleW, bubbleH, 22);
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  const maxChars = 58;
  const clipped = text.length > maxChars ? `${text.slice(0, maxChars - 3)}...` : text;
  ctx.fillText(clipped, bubbleX + padX, bubbleY + 31);
}

function drawReferenceFrame(canvas, image, detections) {
  const ctx = canvas.getContext("2d");
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  drawReferenceOverlay(ctx, detections, 1, 1);
}

function drawRealtimeDetections(ctx, detections, scaleX, scaleY) {
  detections.forEach((det) => {
    const isAlert = det.class_name.toLowerCase().startsWith("no ");
    const color = isAlert ? "#ef4444" : "#10b981";
    const box = det.bbox;
    const x1 = box.x1 * scaleX;
    const y1 = box.y1 * scaleY;
    const x2 = box.x2 * scaleX;
    const y2 = box.y2 * scaleY;
    const w = x2 - x1;
    const h = y2 - y1;
    const icon = isAlert ? "!" : "OK";
    const label = `${icon} ${det.class_name.toUpperCase()} ${(det.confidence * 100).toFixed(0)}%`;

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    drawRoundedRect(ctx, x1, y1, w, h, 10);
    ctx.stroke();

    ctx.font = "bold 12px Trebuchet MS";
    const textW = ctx.measureText(label).width;
    const padX = 8;
    const boxH = 20;
    const labelY = Math.max(4, y1 - boxH + 2);
    ctx.fillStyle = color;
    drawRoundedRect(ctx, x1, labelY, textW + padX * 2, boxH, 10);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + padX, labelY + 14);
  });
}

function connectWs() {
  disconnectRealtime();
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${window.location.host}/ws/live`);
  ws.onopen = () => {
    statusEl.textContent = "Status: connected (WebSocket)";
  };
  ws.onclose = () => {
    statusEl.textContent = "Status: disconnected (WebSocket)";
    sending = false;
    inFlight = false;
  };
  ws.onerror = () => {
    statusEl.textContent = "Status: error (WebSocket)";
    inFlight = false;
  };
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type !== "detections") return;
    latestRealtimeDetections = payload.detections || [];
    latestRealtimeSize = {
      w: payload.image_width || video.videoWidth || 0,
      h: payload.image_height || video.videoHeight || 0,
    };
    updateChart(payload.counts || {});
    inFlight = false;
  };
}

function connectSocketIo() {
  disconnectRealtime();
  if (typeof io === "undefined") {
    statusEl.textContent = "Status: Socket.IO client not loaded";
    return;
  }
  sio = io(window.location.origin, { transports: ["websocket", "polling"] });
  sio.on("connect", () => {
    statusEl.textContent = "Status: connected (Socket.IO)";
  });
  sio.on("disconnect", () => {
    statusEl.textContent = "Status: disconnected (Socket.IO)";
    sending = false;
    inFlight = false;
  });
  sio.on("error", () => {
    statusEl.textContent = "Status: error (Socket.IO)";
    inFlight = false;
  });
  sio.on("detections", (payload) => {
    latestRealtimeDetections = payload.detections || [];
    latestRealtimeSize = {
      w: payload.image_width || video.videoWidth || 0,
      h: payload.image_height || video.videoHeight || 0,
    };
    updateChart(payload.counts || {});
    inFlight = false;
  });
}

function disconnectRealtime() {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (sio) {
    sio.disconnect();
    sio = null;
  }
}

function drawRealtimeFrame() {
  if (video.videoWidth && video.videoHeight) {
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }
    overlayCtx.drawImage(video, 0, 0, overlay.width, overlay.height);
    const baseW = latestRealtimeSize.w || overlay.width;
    const baseH = latestRealtimeSize.h || overlay.height;
    const scaleX = baseW ? overlay.width / baseW : 1;
    const scaleY = baseH ? overlay.height / baseH : 1;
    drawRealtimeDetections(overlayCtx, latestRealtimeDetections, scaleX, scaleY);
  }
  requestAnimationFrame(drawRealtimeFrame);
}

imageFileInput.addEventListener("change", () => {
  const file = imageFileInput.files && imageFileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    imageObj = img;
    imageOriginal.width = img.width;
    imageOriginal.height = img.height;
    imageOriginal.getContext("2d").drawImage(img, 0, 0);
  };
  img.src = url;
});

detectImageBtn.addEventListener("click", async () => {
  const file = imageFileInput.files && imageFileInput.files[0];
  if (!file || !imageObj) return;

  const form = new FormData();
  form.append("file", file);

  const resp = await fetch(`/detect/image?conf=${Number(confInput.value)}&iou=${Number(iouInput.value)}`, {
    method: "POST",
    body: form,
  });
  const data = await resp.json();
  drawReferenceFrame(imageAnnotated, imageObj, data.detections || []);
  updateChart(data.counts || {});
});

videoFileInput.addEventListener("change", () => {
  const file = videoFileInput.files && videoFileInput.files[0];
  if (!file) return;
  videoPreview.src = URL.createObjectURL(file);
});

detectVideoBtn.addEventListener("click", async () => {
  const file = videoFileInput.files && videoFileInput.files[0];
  if (!file) return;

  videoSummary.textContent = "Processing video...";
  const form = new FormData();
  form.append("file", file);

  const resp = await fetch(`/detect/video?conf=${Number(confInput.value)}&iou=${Number(iouInput.value)}&frame_stride=5&max_frames=120`, {
    method: "POST",
    body: form,
  });
  const data = await resp.json();
  videoSummary.textContent = JSON.stringify(data, null, 2);
  updateChart(data.totals || {});
});

function sendFramesLoop() {
  if (!sending) return;
  if (inFlight) {
    setTimeout(sendFramesLoop, FRAME_INTERVAL_MS);
    return;
  }

  if (video.videoWidth && video.videoHeight) {
    const scale = Math.min(1, MAX_CAPTURE_WIDTH / video.videoWidth);
    const targetW = Math.max(1, Math.floor(video.videoWidth * scale));
    const targetH = Math.max(1, Math.floor(video.videoHeight * scale));

    if (captureCanvas.width !== targetW || captureCanvas.height !== targetH) {
      captureCanvas.width = targetW;
      captureCanvas.height = targetH;
    }

    captureCtx.drawImage(video, 0, 0, targetW, targetH);
    const image = captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
    const payload = {
      type: "frame",
      image,
      conf: Number(confInput.value),
      iou: Number(iouInput.value),
    };

    if (transportSelect.value === "websocket" && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
      inFlight = true;
    } else if (transportSelect.value === "socketio" && sio && sio.connected) {
      sio.emit("frame", payload);
      inFlight = true;
    }
  }

  setTimeout(sendFramesLoop, FRAME_INTERVAL_MS);
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

startBtn.addEventListener("click", async () => {
  if (sending) return;
  await startCamera();
  if (transportSelect.value === "socketio") {
    connectSocketIo();
  } else {
    connectWs();
  }
  sending = true;
  sendFramesLoop();
});

captureBtn.addEventListener("click", () => {
  if (!video.videoWidth || !video.videoHeight) return;
  capturedFrameCanvas.width = video.videoWidth;
  capturedFrameCanvas.height = video.videoHeight;
  const ctx = capturedFrameCanvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(video, 0, 0, capturedFrameCanvas.width, capturedFrameCanvas.height);
  const baseW = latestRealtimeSize.w || capturedFrameCanvas.width;
  const baseH = latestRealtimeSize.h || capturedFrameCanvas.height;
  const scaleX = baseW ? capturedFrameCanvas.width / baseW : 1;
  const scaleY = baseH ? capturedFrameCanvas.height / baseH : 1;
  drawReferenceOverlay(ctx, latestRealtimeDetections, scaleX, scaleY);
});

loadClasses();
drawRealtimeFrame();
