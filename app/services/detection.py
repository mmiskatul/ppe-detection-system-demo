from typing import List

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from ..core.config import settings
from ..core.database import get_collection
from ..core.model import get_model
from ..schemas import BBox, DetectionItem


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    data = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    return image


def parse_detections(result) -> List[DetectionItem]:
    detections: List[DetectionItem] = []
    names = result.names or {}
    boxes = result.boxes
    if boxes is None:
        return detections

    allowed = set(settings.allowed_classes)
    xyxy = boxes.xyxy.cpu().tolist()
    conf = boxes.conf.cpu().tolist()
    cls = boxes.cls.cpu().tolist()

    for box, score, class_id in zip(xyxy, conf, cls):
        class_name = str(names.get(int(class_id), int(class_id)))
        if class_name not in allowed:
            continue
        detections.append(
            DetectionItem(
                class_name=class_name,
                class_id=int(class_id),
                confidence=float(score),
                bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
            )
        )
    return detections


def detect_on_image(image: np.ndarray, conf: float, iou: float) -> List[DetectionItem]:
    model = get_model()
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    if not results:
        return []
    return parse_detections(results[0])


def save_detection(payload: dict) -> bool:
    collection = get_collection()
    if collection is None:
        return False
    try:
        collection.insert_one(payload)
        return True
    except Exception:
        return False
