from typing import List

import cv2

from ..schemas import VideoFrameDetection
from .detection import detect_on_image


def detect_on_video(
    path: str, conf: float, iou: float, max_frames: int
) -> List[VideoFrameDetection]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to read video")

    frames: List[VideoFrameDetection] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detect_on_image(frame, conf, iou)
            frames.append(
                VideoFrameDetection(frame_index=frame_index, detections=detections)
            )
            frame_index += 1
            if max_frames > 0 and frame_index >= max_frames:
                break
    finally:
        cap.release()

    return frames
