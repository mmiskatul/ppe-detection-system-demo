from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    color: str
    bbox: BBox


class ClassInfo(BaseModel):
    class_id: int
    class_name: str
    color: str


class ClassesResponse(BaseModel):
    classes: List[ClassInfo]


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    detections: List[Detection]
    counts: Dict[str, int]


class VideoFrameDetections(BaseModel):
    frame_index: int
    detections: List[Detection]
    counts: Dict[str, int]


class VideoDetectionResponse(BaseModel):
    total_frames: int
    processed_frames: int
    frames: List[VideoFrameDetections]
    totals: Dict[str, int]


class WsFrameRequest(BaseModel):
    type: Literal["frame"] = "frame"
    image: str = Field(..., description="Base64 image string (optional data URL prefix)")
    conf: Optional[float] = None
    iou: Optional[float] = None


class WsDetectionResponse(BaseModel):
    type: Literal["detections"] = "detections"
    image_width: int
    image_height: int
    detections: List[Detection]
    counts: Dict[str, int]
