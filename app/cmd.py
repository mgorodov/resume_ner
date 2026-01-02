import uvicorn
from app.asgi.app import create_asgi_app
from app.config.settings import settings


def main():
    asgi_app = create_asgi_app()
    uvicorn.run(app=asgi_app, host=settings.uvicorn.host, port=settings.uvicorn.port)


if __name__ == "__main__":
    main()
