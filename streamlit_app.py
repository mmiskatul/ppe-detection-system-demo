import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import socketio
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


DEFAULT_API_BASE = "http://localhost:8080"


def api_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


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
        label = f"{det.get('class_name', 'unknown')} {det.get('confidence', 0):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(y1 - 12, 0)), label, fill=(0, 255, 0))
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


def socketio_frame_detection(base: str, image: Image.Image) -> Dict[str, Any]:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")

    sio = socketio.Client()
    results: Dict[str, Any] = {}

    @sio.event
    def connect():
        sio.emit("frame", {"image": payload})

    @sio.event
    def detections(data):
        results.update(data)
        sio.disconnect()

    @sio.event
    def error(data):
        results.update({"error": data})
        sio.disconnect()

    sio.connect(base, transports=["websocket", "polling"])
    sio.wait()
    return results


def render_json(title: str, data: Dict[str, Any]) -> None:
    st.subheader(title)
    st.json(data)


def main() -> None:
    st.set_page_config(page_title="YOLO Detection UI", layout="wide")
    st.title("YOLO Detection UI")

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
        rt_file = st.file_uploader(
            "Upload an image for realtime", type=["jpg", "jpeg", "png"], key="rt"
        )
        if rt_file:
            rt_bytes = rt_file.read()
            image = Image.open(BytesIO(rt_bytes)).convert("RGB")
            st.image(image, caption="Original", use_container_width=True)
            if st.button("Send realtime frame"):
                with st.spinner("Sending frame..."):
                    result = socketio_frame_detection(base, image)
                detections = result.get("detections", [])
                annotated = draw_detections(image.copy(), detections)
                st.image(annotated, caption="Annotated", use_container_width=True)
                render_json("Realtime JSON", result)


if __name__ == "__main__":
    try:
        import cv2
    except Exception as exc:
        raise SystemExit("opencv-python is required for video frame preview") from exc
    main()
