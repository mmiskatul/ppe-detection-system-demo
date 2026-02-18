# YOLO Detection API (FastAPI + Pydantic + WebSocket)

Backend and frontend for realtime YOLO inference using `best.pt`.

## Detected Classes in `best.pt`

1. boots
2. glasses
3. gloves
4. helmet
5. no boots
6. no glasses
7. no gloves
8. no helmet
9. no vest
10. person
11. vest

## Features

- `POST /detect/image` for image detection
- `POST /detect/video` for video detection JSON + annotated video URL
- `POST /detect/video/file` for direct annotated MP4 file response
- `GET /classes` for all classes + standard fixed color map
- `GET /health` for health checks
- `WS /ws/live` for live detection over WebSocket
- Browser frontend at `/`:
  - live webcam inference
  - per-class colored bounding boxes
  - all-class bar plot (live counts)

## Run

```bash
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open: `http://localhost:8080`

## Environment Variables

- `YOLO_MODEL_PATH` (default: `best.pt`)
- `YOLO_CONF` (default: `0.25`)
- `YOLO_IOU` (default: `0.45`)
- `WS_FRAME_INTERVAL_MS` (default: `250`)
