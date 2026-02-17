# YOLO Detection API

FastAPI service for image, video, and realtime (Socket.IO) detection using a YOLO model, with optional MongoDB persistence.

## Features

- Image detection endpoint
- Video detection endpoint (per-frame results)
- Realtime detection via Socket.IO
- MongoDB persistence for requests/results (optional)

## Project Structure

- app/main.py: app factory and ASGI socket app
- app/api/routes.py: REST endpoints
- app/core/config.py: settings and env config
- app/core/database.py: MongoDB helpers
- app/core/model.py: YOLO model loader
- app/services/detection.py: detection logic
- app/services/video.py: video processing
- app/realtime/socket.py: Socket.IO handlers
- app/schemas.py: Pydantic schemas

## Requirements

- Python 3.10+ recommended
- A YOLO weights file (default: yolo11l.pt at repo root)

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a .env file (or set environment variables). Example:

```bash
MONGODB_URI=mongodb+srv://<user>:<pass>@cluster0.ugubgo0.mongodb.net/?appName=Cluster0
MONGODB_DB=yolo_model_project_db
MONGODB_COLLECTION=yolo_model_project_detections
YOLO_MODEL_PATH=yolo11l.pt
YOLO_CONF=0.25
YOLO_IOU=0.45
MAX_FRAMES=300
```

Notes:
- If MONGODB_URI is empty, results are not stored.
- Set MAX_FRAMES=0 to process the full video.

## Run

```bash
uvicorn app.main:socket_app --host 0.0.0.0 --port 8000
```

Health check:

```bash
GET http://localhost:8000/health
```

## REST API

### POST /detect/image

- Content-Type: multipart/form-data
- Field: file (image)
- Optional query params: conf, iou

### POST /detect/video

- Content-Type: multipart/form-data
- Field: file (video)
- Optional query params: conf, iou

## Realtime Socket.IO

Connect to the same server and send a base64-encoded image:

Event: frame
Payload:

```json
{
  "image": "<base64>"
}
```

Response event: detections

## Allowed Classes

- boots
- glasses
- gloves
- helmet
- no boots
- no glasses
- no gloves
- no helmet
- no vest
- person
- vest
