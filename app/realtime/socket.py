import asyncio
import base64
import time
import uuid
from typing import Any

import cv2
import numpy as np
import socketio
from fastapi import FastAPI

from ..core.config import settings
from ..core.model import resolve_model_path
from ..services.detection import detect_on_image, save_detection

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")


def create_socket_app(app: FastAPI) -> socketio.ASGIApp:
    return socketio.ASGIApp(sio, other_asgi_app=app)


@sio.event
async def connect(sid, environ):
    await sio.emit("status", {"status": "connected"}, to=sid)


@sio.event
async def disconnect(sid):
    return None


@sio.event
async def frame(sid, data: Any):
    if not isinstance(data, dict) or "image" not in data:
        await sio.emit("error", {"error": "Missing image"}, to=sid)
        return
    try:
        raw = base64.b64decode(data["image"])
    except Exception:
        await sio.emit("error", {"error": "Invalid base64"}, to=sid)
        return

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        await sio.emit("error", {"error": "Unsupported image format"}, to=sid)
        return

    conf = float(data.get("conf", settings.yolo_conf))
    iou = float(data.get("iou", settings.yolo_iou))
    detections = await asyncio.to_thread(detect_on_image, image, conf, iou)

    payload = {
        "request_id": str(uuid.uuid4()),
        "type": "realtime",
        "model": str(resolve_model_path(settings)),
        "created_at": time.time(),
        "detections": [d.model_dump() for d in detections],
    }
    saved = await asyncio.to_thread(save_detection, payload)

    await sio.emit(
        "detections",
        {
            "model": str(resolve_model_path(settings)),
            "detections": [d.model_dump() for d in detections],
            "saved": saved,
        },
        to=sid,
    )
