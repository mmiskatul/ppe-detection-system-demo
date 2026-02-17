from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "YOLO Live Detection API"
    yolo_model_path: str = "best.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    ws_frame_interval_ms: int = 250


@lru_cache
def get_settings() -> Settings:
    return Settings()
