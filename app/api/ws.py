import json

from fastapi import HTTPException
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas import WsDetectionResponse, WsFrameRequest
from app.services.detection import decode_base64_image, run_detection

router = APIRouter()


@router.websocket("/ws/live")
async def live_detection_ws(websocket: WebSocket) -> None:
    settings = get_settings()
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = WsFrameRequest.model_validate_json(raw)
            except ValidationError as exc:
                await websocket.send_text(json.dumps({"type": "error", "detail": exc.errors()}))
                continue

            try:
                image = decode_base64_image(payload.image)
            except HTTPException as exc:
                await websocket.send_text(json.dumps({"type": "error", "detail": exc.detail}))
                continue
            conf = payload.conf if payload.conf is not None else settings.yolo_conf
            iou = payload.iou if payload.iou is not None else settings.yolo_iou
            detections, counts = run_detection(image, conf=conf, iou=iou)
            h, w = image.shape[:2]
            response = WsDetectionResponse(
                image_width=w,
                image_height=h,
                detections=detections,
                counts=counts,
            )
            await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        return
