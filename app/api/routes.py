import asyncio
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..core.config import settings
from ..core.model import resolve_model_path
from ..schemas import (
    HealthResponse,
    ImageDetectionResponse,
    VideoDetectionResponse,
    VideoFrameDetection,
)
from ..services.detection import detect_on_image, load_image_from_upload, save_detection
from ..services.video import detect_on_video

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=str(resolve_model_path(settings)))


@router.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf: float = settings.yolo_conf,
    iou: float = settings.yolo_iou,
) -> ImageDetectionResponse:
    request_id = str(uuid.uuid4())
    image = load_image_from_upload(file)
    detections = await asyncio.to_thread(detect_on_image, image, conf, iou)

    payload = {
        "request_id": request_id,
        "type": "image",
        "model": str(resolve_model_path(settings)),
        "created_at": time.time(),
        "detections": [d.model_dump() for d in detections],
    }
    saved = await asyncio.to_thread(save_detection, payload)

    return ImageDetectionResponse(
        request_id=request_id,
        model=str(resolve_model_path(settings)),
        detections=detections,
        saved=saved,
    )


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    conf: float = settings.yolo_conf,
    iou: float = settings.yolo_iou,
) -> VideoDetectionResponse:
    request_id = str(uuid.uuid4())
    suffix = Path(file.filename or "video").suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        temp_path = tmp.name

    try:
        frames: List[VideoFrameDetection] = await asyncio.to_thread(
            detect_on_video, temp_path, conf, iou, settings.max_frames
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass

    payload = {
        "request_id": request_id,
        "type": "video",
        "model": str(resolve_model_path(settings)),
        "created_at": time.time(),
        "frames": [f.model_dump() for f in frames],
    }
    saved = await asyncio.to_thread(save_detection, payload)

    return VideoDetectionResponse(
        request_id=request_id,
        model=str(resolve_model_path(settings)),
        frames=frames,
        saved=saved,
    )
