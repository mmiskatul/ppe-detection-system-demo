import base64
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.colors import get_class_colors
from app.core.config import get_settings
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
    settings = get_settings()
    class_map = get_model_class_map()
    class_names = get_model_class_names()
    color_map = get_class_colors()

    counts: Dict[str, int] = {name: 0 for name in class_names}
    result = model_predict(image_bgr, conf=conf, iou=iou, imgsz=settings.yolo_imgsz)

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


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    raw = color_hex.lstrip("#")
    if len(raw) != 6:
        return (0, 255, 0)
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    return (b, g, r)


def annotate_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        x1 = int(round(det.bbox.x1))
        y1 = int(round(det.bbox.y1))
        x2 = int(round(det.bbox.x2))
        y2 = int(round(det.bbox.y2))
        color = _hex_to_bgr(det.color)
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_y = max(y1 - 6, text_h + 6)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - baseline - 6),
            (x1 + text_w + 8, text_y + baseline - 4),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return annotated


def _bbox_iou(a: BBox, b: BBox) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a.x2 - a.x1)) * max(0.0, (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1)) * max(0.0, (b.y2 - b.y1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def smooth_detections(
    previous: List[Detection],
    current: List[Detection],
    *,
    alpha: float = 0.65,
    iou_threshold: float = 0.25,
) -> List[Detection]:
    if not previous:
        return current

    alpha = max(0.0, min(1.0, alpha))
    used_prev: set[int] = set()
    smoothed: List[Detection] = []

    for cur in current:
        best_idx = -1
        best_iou = 0.0
        for idx, prev in enumerate(previous):
            if idx in used_prev or prev.class_id != cur.class_id:
                continue
            iou_val = _bbox_iou(prev.bbox, cur.bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            prev = previous[best_idx]
            used_prev.add(best_idx)
            smoothed.append(
                Detection(
                    class_id=cur.class_id,
                    class_name=cur.class_name,
                    confidence=float(alpha * cur.confidence + (1.0 - alpha) * prev.confidence),
                    color=cur.color,
                    bbox=BBox(
                        x1=float(alpha * cur.bbox.x1 + (1.0 - alpha) * prev.bbox.x1),
                        y1=float(alpha * cur.bbox.y1 + (1.0 - alpha) * prev.bbox.y1),
                        x2=float(alpha * cur.bbox.x2 + (1.0 - alpha) * prev.bbox.x2),
                        y2=float(alpha * cur.bbox.y2 + (1.0 - alpha) * prev.bbox.y2),
                    ),
                )
            )
        else:
            smoothed.append(cur)
    return smoothed
