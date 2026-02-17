from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    base_dir: Path = Path(__file__).resolve().parents[2]
    yolo_model_path: Path = Field(default_factory=lambda: Path("yolo11l.pt"))
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    max_frames: int = 300

    mongodb_uri: Optional[str] = None
    mongodb_db: str = "yolo_model_project_db"
    mongodb_collection: str = "yolo_model_project_detections"

    allowed_classes: List[str] = [
        "boots",
        "glasses",
        "gloves",
        "helmet",
        "no boots",
        "no glasses",
        "no gloves",
        "no helmet",
        "no vest",
        "person",
        "vest",
    ]


def resolve_model_path(settings: Settings) -> Path:
    if settings.yolo_model_path.is_absolute():
        return settings.yolo_model_path
    return (settings.base_dir / settings.yolo_model_path).resolve()


settings = Settings()
