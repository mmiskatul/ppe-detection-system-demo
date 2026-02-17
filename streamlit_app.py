import base64
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import av
import cv2
import numpy as np
import requests
import socketio
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer


DEFAULT_API_BASE = "http://localhost:8080"

# Color map for each class
CLASS_COLORS = {
    "boots": (255, 50, 50),
    "glasses": (50, 50, 255),
    "gloves": (50, 255, 50),
    "helmet": (255, 255, 0),
    "no boots": (255, 100, 0),
    "no glasses": (150, 0, 255),
    "no gloves": (0, 255, 255),
    "no helmet": (255, 0, 255),
    "no vest": (200, 255, 20),
    "person": (255, 192, 203),
    "vest": (0, 150, 150),
}


def api_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def get_class_color(class_name: str) -> tuple:
    return CLASS_COLORS.get(class_name.lower().strip(), (0, 255, 0))


def draw_detections(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        bbox = det.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if None in (x1, y1, x2, y2):
            continue
        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0)
        color = get_class_color(class_name)
        
        # Draw bounding box border (thicker)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Determine status icon
        status_icon = "✓" if not class_name.startswith("no ") else "⚠"
        label = f"{status_icon} {class_name.upper()} {confidence:.0%}"
        
        # Calculate label background
        label_y = max(y1 - 35, 5)
        label_height = 30
        label_box = [x1, label_y, x1 + len(label) * 7 + 20, label_y + label_height]
        
        # Draw label background with color
        draw.rectangle(label_box, fill=color, outline=color)
        
        # Draw label text in white
        draw.text(
            (x1 + 8, label_y + 5),
            label,
            fill=(255, 255, 255),
        )
    return image


def call_image_detection(base: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes)}
    response = requests.post(api_url(base, "/detect/image"), files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def call_video_detection(base: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes)}
    response = requests.post(api_url(base, "/detect/video"), files=files, timeout=300)
    response.raise_for_status()
    return response.json()


class SocketIOVideoProcessor(VideoProcessorBase):
    def __init__(self, base: str, conf: float, iou: float, interval: float = 0.3):
        self.base = base
        self.conf = conf
        self.iou = iou
        self.interval = interval
        self.last_sent = 0.0
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_payload: Dict[str, Any] = {}
        self.lock = threading.Lock()

        self.sio = socketio.Client()
        self._connect_socket()

    def _connect_socket(self) -> None:
        @self.sio.event
        def connect():
            return None

        @self.sio.event
        def detections(data):
            with self.lock:
                self.latest_payload = data
                self.latest_detections = data.get("detections", [])

        @self.sio.event
        def error(data):
            with self.lock:
                self.latest_payload = {"error": data}
                self.latest_detections = []

        try:
            self.sio.connect(self.base, transports=["websocket", "polling"])
        except Exception:
            with self.lock:
                self.latest_payload = {"error": "Socket.IO connection failed"}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        if self.sio.connected and (now - self.last_sent) >= self.interval:
            self.last_sent = now
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                payload = base64.b64encode(buf.tobytes()).decode("ascii")
                try:
                    self.sio.emit(
                        "frame", {"image": payload, "conf": self.conf, "iou": self.iou}
                    )
                except Exception:
                    pass

        with self.lock:
            dets = list(self.latest_detections)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        annotated = draw_detections(pil, dets)
        out = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


def render_json(title: str, data: Dict[str, Any]) -> None:
    st.subheader(title)
    st.json(data)


def show_color_legend() -> None:
    st.sidebar.markdown("## Class Color Legend")
    for class_name, color in CLASS_COLORS.items():
        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        st.sidebar.markdown(
            f"<span style='color:{hex_color}'>●</span> **{class_name}**",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="YOLO Detection UI", layout="wide")
    st.title("YOLO Detection UI")
    show_color_legend()

    base = st.text_input("API base URL", value=DEFAULT_API_BASE)
    tab_image, tab_video, tab_realtime = st.tabs(["Image", "Video", "Realtime"])

    with tab_image:
        st.header("Image Detection")
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file:
            image_bytes = image_file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption="Original", use_container_width=True)
            if st.button("Detect image"):
                with st.spinner("Running detection..."):
                    result = call_image_detection(base, image_bytes, image_file.name)
                detections = result.get("detections", [])
                annotated = draw_detections(image.copy(), detections)
                st.image(annotated, caption="Annotated", use_container_width=True)
                st.markdown(
                    f"Request ID: {result.get('request_id')} | Saved: {result.get('saved')}"
                )
                render_json("Detections JSON", result)

    with tab_video:
        st.header("Video Detection")
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
            if st.button("Detect video"):
                with st.spinner("Running video detection..."):
                    result = call_video_detection(base, video_bytes, video_file.name)
                st.markdown(
                    f"Request ID: {result.get('request_id')} | Saved: {result.get('saved')}"
                )
                render_json("Video JSON", result)

                frames = result.get("frames", [])
                if frames:
                    st.subheader("Annotated Frame Preview")
                    frame_index = st.slider(
                        "Frame index", 0, max(0, len(frames) - 1), 0
                    )
                    frame_info = frames[frame_index]
                    temp_path = Path(".streamlit_temp_video")
                    temp_path.write_bytes(video_bytes)
                    cap = cv2.VideoCapture(str(temp_path))
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ok, frame = cap.read()
                        if ok:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_image = Image.fromarray(frame_rgb)
                            annotated = draw_detections(
                                frame_image, frame_info.get("detections", [])
                            )
                            st.image(
                                annotated,
                                caption=f"Frame {frame_index}",
                                use_container_width=True,
                            )
                    finally:
                        cap.release()
                        temp_path.unlink(missing_ok=True)

    with tab_realtime:
        st.header("Realtime Detection (Socket.IO)")
        conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01, key="rt_conf")
        iou = st.slider("IOU", 0.0, 1.0, 0.45, 0.01, key="rt_iou")
        st.caption("Allow camera access when prompted. Frames are sent to Socket.IO.")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key="realtime",
            rtc_configuration=rtc_config,
            video_processor_factory=lambda: SocketIOVideoProcessor(
                base=base, conf=conf, iou=iou, interval=0.3
            ),
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx and ctx.video_processor:
            with ctx.video_processor.lock:
                payload = dict(ctx.video_processor.latest_payload)
            if payload:
                render_json("Realtime JSON", payload)


if __name__ == "__main__":
    main()
