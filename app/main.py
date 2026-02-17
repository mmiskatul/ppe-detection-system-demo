from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import socketio

from app.api.routes import router as api_router
from app.api.ws import router as ws_router
from app.core.config import get_settings
from app.realtime.socketio_server import create_socket_server

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.include_router(api_router)
app.include_router(ws_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("app/static/index.html")


sio = create_socket_server()
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
