import argparse

import uvicorn


def run(host: str = "0.0.0.0", port: int = 8090, reload: bool = False) -> None:
    print(f"Starting backend + frontend on http://127.0.0.1:{port}")
    print(f"Frontend: http://127.0.0.1:{port}/")
    print(f"Health:   http://127.0.0.1:{port}/health")
    print(f"Docs:     http://127.0.0.1:{port}/docs")
    uvicorn.run("app.main:socket_app", host=host, port=port, reload=reload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO FastAPI backend and frontend (served by FastAPI static files)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(host=args.host, port=args.port, reload=args.reload)
