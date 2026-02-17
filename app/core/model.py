from functools import lru_cache

from ultralytics import YOLO

from .config import Settings, resolve_model_path, settings


@lru_cache
def get_model() -> YOLO:
    model_path = resolve_model_path(settings)
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
    return YOLO(str(model_path))


def get_model_path(settings_obj: Settings) -> str:
    return str(resolve_model_path(settings_obj))
