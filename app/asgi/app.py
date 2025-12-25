from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.asgi.routers import auth, forward, history, stats, users
from app.db.helper import DatabaseHelper
from app.ml.ner_service import NERService
from app.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_helper = DatabaseHelper(settings.db.url, settings.db.echo)
    app.state.ner_service = NERService()
    try:
        yield
    finally:
        await app.state.db_helper.dispose()


def create_asgi_app() -> FastAPI:
    app = FastAPI(
        title="resume-ner",
        openapi_url="/swagger.json",
        lifespan=lifespan,
    )

    app.include_router(auth.router)
    app.include_router(forward.router)
    app.include_router(history.router)
    app.include_router(stats.router)
    app.include_router(users.router)

    # TODO: think about CORS
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    # )
    return app
