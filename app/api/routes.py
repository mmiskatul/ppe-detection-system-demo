import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.colors import get_class_colors
from app.core.model import get_model_class_map, get_model_class_names
from app.schemas import (
    ClassInfo,
    ClassesResponse,
    DetectionResponse,
    VideoDetectionResponse,
    VideoFrameDetections,
)
from app.services.detection import decode_image_bytes, run_detection

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/classes", response_model=ClassesResponse)
async def classes() -> ClassesResponse:
    class_map = get_model_class_map()
    color_map = get_class_colors()
    all_classes = [
        ClassInfo(class_id=class_id, class_name=class_name, color=color_map[class_name])
        for class_id, class_name in class_map.items()
    ]
    return ClassesResponse(classes=all_classes)


@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
) -> DetectionResponse:
    image_bytes = await file.read()
    image = decode_image_bytes(image_bytes)
    detections, counts = run_detection(image, conf=conf, iou=iou)
    h, w = image.shape[:2]
    return DetectionResponse(
        image_width=w,
        image_height=h,
        detections=detections,
        counts=counts,
    )


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    frame_stride: int = 5,
    max_frames: int = 120,
) -> VideoDetectionResponse:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    class_names = get_model_class_names()
    totals = {name: 0 for name in class_names}
    frames: list[VideoFrameDetections] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid video file")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = 0
        processed_frames = 0
        safe_stride = max(1, frame_stride)
        safe_max = max(1, max_frames)

        while processed_frames < safe_max:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % safe_stride == 0:
                detections, counts = run_detection(frame, conf=conf, iou=iou)
                for class_name, count in counts.items():
                    totals[class_name] = totals.get(class_name, 0) + count
                frames.append(
                    VideoFrameDetections(
                        frame_index=frame_index,
                        detections=detections,
                        counts=counts,
                    )
                )
                processed_frames += 1
            frame_index += 1
    finally:
        cap.release()
        tmp_path.unlink(missing_ok=True)

    return VideoDetectionResponse(
        total_frames=total_frames,
        processed_frames=processed_frames,
        frames=frames,
        totals=totals,
    )
