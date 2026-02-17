import base64
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.colors import get_class_colors
from app.core.model import get_model_class_map, get_model_class_names, model_predict
from app.schemas import BBox, Detection


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


def decode_base64_image(encoded: str) -> np.ndarray:
    try:
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc
    return decode_image_bytes(image_bytes)


def run_detection(image_bgr: np.ndarray, conf: float, iou: float) -> Tuple[List[Detection], Dict[str, int]]:
    class_map = get_model_class_map()
    class_names = get_model_class_names()
    color_map = get_class_colors()

    counts: Dict[str, int] = {name: 0 for name in class_names}
    result = model_predict(image_bgr, conf=conf, iou=iou)

    detections: List[Detection] = []
    boxes = result.boxes
    if boxes is None:
        return detections, counts

    for box in boxes:
        cls_id = int(box.cls.item())
        class_name = class_map.get(cls_id, f"class_{cls_id}")
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        if class_name not in counts:
            counts[class_name] = 0
        counts[class_name] += 1

        detections.append(
            Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=confidence,
                color=color_map.get(class_name, "#00ff00"),
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
        )

    return detections, counts
