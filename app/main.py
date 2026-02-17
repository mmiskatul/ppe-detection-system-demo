from fastapi import FastAPI

from .api.routes import router as api_router
from .realtime.socket import create_socket_app


def create_app() -> FastAPI:
    app = FastAPI(title="YOLO Detection API")
    app.include_router(api_router)
    return app


app = create_app()
socket_app = create_socket_app(app)
