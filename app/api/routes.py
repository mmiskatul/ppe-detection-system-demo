import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi.responses import FileResponse
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
from app.services.detection import annotate_detections, decode_image_bytes, run_detection, smooth_detections

router = APIRouter()


def _render_annotated_video(
    *,
    input_path: Path,
    conf: float,
    iou: float,
    frame_stride: int,
    max_frames: int | None,
    include_frames: bool,
    smooth: bool,
    smooth_alpha: float,
) -> tuple[Path, int, int, int, float, int, dict[str, int], list[VideoFrameDetections]]:
    class_names = get_model_class_names()
    totals = {name: 0 for name in class_names}
    frames: list[VideoFrameDetections] = []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        safe_fps = video_fps if video_fps > 0 else 25.0

        output_dir = Path("app/static/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"annotated_{uuid4().hex}.mp4"
        output_path = output_dir / output_name

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            safe_fps,
            (video_width, video_height),
        )
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Unable to initialize annotated video writer")

        frame_index = 0
        processed_frames = 0
        safe_stride = 1 if smooth else max(1, frame_stride)
        safe_max = max_frames if max_frames is None else max(1, max_frames)
        previous_detections: list = []

        try:
            while True:
                if safe_max is not None and processed_frames >= safe_max:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                frame_to_write = frame
                if frame_index % safe_stride == 0:
                    detections, counts = run_detection(frame, conf=conf, iou=iou)
                    if smooth:
                        detections = smooth_detections(
                            previous_detections,
                            detections,
                            alpha=smooth_alpha,
                        )
                    previous_detections = detections
                    frame_to_write = annotate_detections(frame, detections)
                    for class_name, count in counts.items():
                        totals[class_name] = totals.get(class_name, 0) + count
                    if include_frames:
                        frames.append(
                            VideoFrameDetections(
                                frame_index=frame_index,
                                detections=detections,
                                counts=counts,
                            )
                        )
                    processed_frames += 1
                elif previous_detections:
                    frame_to_write = annotate_detections(frame, previous_detections)
                writer.write(frame_to_write)
                frame_index += 1
        finally:
            writer.release()
    finally:
        cap.release()

    return output_path, total_frames, processed_frames, video_width, video_height, safe_fps, totals, frames


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
    frame_stride: int = 1,
    max_frames: int | None = None,
    include_frames: bool = False,
    smooth: bool = True,
    smooth_alpha: float = 0.65,
) -> VideoDetectionResponse:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        (
            output_path,
            total_frames,
            processed_frames,
            video_width,
            video_height,
            safe_fps,
            totals,
            frames,
        ) = _render_annotated_video(
            input_path=tmp_path,
            conf=conf,
            iou=iou,
            frame_stride=frame_stride,
            max_frames=max_frames,
            include_frames=include_frames,
            smooth=smooth,
            smooth_alpha=smooth_alpha,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return VideoDetectionResponse(
        total_frames=total_frames,
        processed_frames=processed_frames,
        video_width=video_width,
        video_height=video_height,
        video_fps=safe_fps,
        annotated_video_url=f"/static/outputs/{output_path.name}",
        frames=frames,
        totals=totals,
    )


@router.post("/detect/video/file")
async def detect_video_file(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    frame_stride: int = 1,
    max_frames: int | None = None,
    smooth: bool = True,
    smooth_alpha: float = 0.65,
) -> FileResponse:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        (
            output_path,
            total_frames,
            processed_frames,
            video_width,
            video_height,
            safe_fps,
            totals,
            _frames,
        ) = _render_annotated_video(
            input_path=tmp_path,
            conf=conf,
            iou=iou,
            frame_stride=frame_stride,
            max_frames=max_frames,
            include_frames=False,
            smooth=smooth,
            smooth_alpha=smooth_alpha,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    headers = {
        "X-Total-Frames": str(total_frames),
        "X-Processed-Frames": str(processed_frames),
        "X-Video-Width": str(video_width),
        "X-Video-Height": str(video_height),
        "X-Video-FPS": f"{safe_fps:.3f}",
        "X-Totals": str(totals),
    }
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"annotated_{Path(file.filename or 'video').stem}.mp4",
        headers=headers,
    )
