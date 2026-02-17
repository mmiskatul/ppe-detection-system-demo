from typing import List

from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox


class ImageDetectionResponse(BaseModel):
    request_id: str
    model: str
    detections: List[DetectionItem]
    saved: bool = False


class VideoFrameDetection(BaseModel):
    frame_index: int
    detections: List[DetectionItem]


class VideoDetectionResponse(BaseModel):
    request_id: str
    model: str
    frames: List[VideoFrameDetection]
    saved: bool = False


class HealthResponse(BaseModel):
    status: str
    model: str
