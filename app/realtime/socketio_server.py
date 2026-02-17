import socketio
from fastapi import HTTPException

from app.core.config import get_settings
from app.schemas import WsDetectionResponse, WsFrameRequest
from app.services.detection import decode_base64_image, run_detection


def create_socket_server() -> socketio.AsyncServer:
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    settings = get_settings()

    @sio.event
    async def connect(sid, environ, auth):  # noqa: ANN001, ARG001
        return

    @sio.event
    async def disconnect(sid):  # noqa: ANN001
        return

    @sio.on("frame")
    async def frame(sid, data):  # noqa: ANN001
        try:
            payload = WsFrameRequest.model_validate(data)
            image = decode_base64_image(payload.image)
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
            await sio.emit("detections", response.model_dump(), to=sid)
        except HTTPException as exc:
            await sio.emit("error", {"detail": exc.detail}, to=sid)
        except Exception as exc:  # noqa: BLE001
            await sio.emit("error", {"detail": str(exc)}, to=sid)

    return sio
