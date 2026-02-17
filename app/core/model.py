from functools import lru_cache
from threading import Lock
from typing import Dict, List

from ultralytics import YOLO

from app.core.config import get_settings

_predict_lock = Lock()


@lru_cache
def get_model() -> YOLO:
    settings = get_settings()
    return YOLO(settings.yolo_model_path)


def get_model_class_names() -> List[str]:
    model = get_model()
    names = model.names
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    return list(names)


def model_predict(image_bgr, conf: float, iou: float, imgsz: int | None = None):
    model = get_model()
    kwargs = {"conf": conf, "iou": iou, "verbose": False}
    if imgsz:
        kwargs["imgsz"] = imgsz
    with _predict_lock:
        return model.predict(image_bgr, **kwargs)[0]


def get_model_class_map() -> Dict[int, str]:
    model = get_model()
    names = model.names
    if isinstance(names, dict):
        return dict(sorted(names.items()))
    return {idx: name for idx, name in enumerate(names)}
