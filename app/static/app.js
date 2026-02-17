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
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start-btn");
const transportSelect = document.getElementById("transport");
const confInput = document.getElementById("conf");
const iouInput = document.getElementById("iou");
const confValue = document.getElementById("conf-value");
const iouValue = document.getElementById("iou-value");
const legendEl = document.getElementById("legend");

const overlayCtx = overlay.getContext("2d");
let ws = null;
let sio = null;
let classes = [];
let colorsByClass = {};
let chart = null;
let sending = false;
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
  colorsByClass = {};
  classes.forEach((c) => {
    colorsByClass[c.class_name] = c.color;
  });
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

function initChart() {
  const ctx = document.getElementById("classChart");
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: classes.map((c) => c.class_name),
      datasets: [
        {
          label: "Live Count",
          data: classes.map(() => 0),
          backgroundColor: classes.map((c) => c.color),
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1 },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

function drawDetectionsOnCanvas(canvas, image, detections) {
  const ctx = canvas.getContext("2d");
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  drawStyledDetections(ctx, detections, 1, 1);
}

function drawStyledDetections(ctx, detections, scaleX, scaleY) {
  detections.forEach((det) => {
    const isAlert = det.class_name.toLowerCase().startsWith("no ");
    const c = isAlert ? "#ef4444" : "#10b981";
    const box = det.bbox;
    const x1 = box.x1 * scaleX;
    const y1 = box.y1 * scaleY;
    const x2 = box.x2 * scaleX;
    const y2 = box.y2 * scaleY;
    const w = x2 - x1;
    const h = y2 - y1;
    const icon = isAlert ? "!" : "✓";
    const label = `${icon} ${det.class_name.toUpperCase()}`;

    ctx.strokeStyle = c;
    ctx.lineWidth = 3;
    drawRoundedRect(ctx, x1, y1, w, h, 10);
    ctx.stroke();

    ctx.font = "bold 12px Trebuchet MS";
    const textW = ctx.measureText(label).width;
    const padX = 8;
    const boxH = 20;
    const labelY = Math.max(4, y1 - boxH + 2);
    ctx.fillStyle = c;
    drawRoundedRect(ctx, x1, labelY, textW + padX * 2, boxH, 10);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + padX, labelY + 14);
  });
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
  };
  ws.onerror = () => {
    statusEl.textContent = "Status: error (WebSocket)";
  };
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type !== "detections") return;
    latestRealtimeDetections = payload.detections || [];
    latestRealtimeSize = {
      w: payload.image_width || video.videoWidth || 0,
      h: payload.image_height || video.videoHeight || 0,
    };
    updateChart(payload.counts);
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
  });
  sio.on("error", (payload) => {
    statusEl.textContent = `Status: error (Socket.IO) ${JSON.stringify(payload)}`;
  });
  sio.on("detections", (payload) => {
    latestRealtimeDetections = payload.detections || [];
    latestRealtimeSize = {
      w: payload.image_width || video.videoWidth || 0,
      h: payload.image_height || video.videoHeight || 0,
    };
    updateChart(payload.counts || {});
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

function updateChart(counts) {
  if (!chart) return;
  chart.data.datasets[0].data = classes.map((c) => counts[c.class_name] || 0);
  chart.update();
}

function drawDetections(detections) {
  if (video.videoWidth && video.videoHeight) {
    overlayCtx.drawImage(video, 0, 0, overlay.width, overlay.height);
  } else {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }
  const baseW = latestRealtimeSize.w || overlay.width;
  const baseH = latestRealtimeSize.h || overlay.height;
  const scaleX = baseW ? overlay.width / baseW : 1;
  const scaleY = baseH ? overlay.height / baseH : 1;
  drawStyledDetections(overlayCtx, detections, scaleX, scaleY);
}

function renderRealtimeOverlayLoop() {
  if (video.videoWidth && video.videoHeight) {
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }
    drawDetections(latestRealtimeDetections);
  }
  requestAnimationFrame(renderRealtimeOverlayLoop);
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
  drawDetectionsOnCanvas(imageAnnotated, imageObj, data.detections || []);
  updateChart(data.counts || {});
});

videoFileInput.addEventListener("change", () => {
  const file = videoFileInput.files && videoFileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
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
  if (video.videoWidth && video.videoHeight) {
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }

    const temp = document.createElement("canvas");
    temp.width = video.videoWidth;
    temp.height = video.videoHeight;
    const tempCtx = temp.getContext("2d");
    tempCtx.drawImage(video, 0, 0);
    const image = temp.toDataURL("image/jpeg", 0.7);
    const payload = {
      type: "frame",
      image,
      conf: Number(confInput.value),
      iou: Number(iouInput.value),
    };
    if (transportSelect.value === "websocket" && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    } else if (transportSelect.value === "socketio" && sio && sio.connected) {
      sio.emit("frame", payload);
    }
  }
  setTimeout(sendFramesLoop, 250);
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

loadClasses();
renderRealtimeOverlayLoop();
