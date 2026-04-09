"""SAMURAI REST API entry point.

Startup sequence:
1. Load ``Settings`` from environment.
2. Build ``ModelConfig`` and resolve checkpoint / device.
3. Load SAM2 predictor (GPU-heavy, done once).
4. Mount routers and register exception handlers.
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings, get_settings
from api.exceptions import (
    SessionNotFoundError,
    VideoNotFoundError,
    VideoProcessingError,
    PropagationError,
    session_not_found_handler,
    video_not_found_handler,
    video_processing_error_handler,
    propagation_error_handler,
)
from api.ml.loader import build_model_config, load_image_predictor, load_predictor
from api.repositories.session_repo import SessionRepository
from api.repositories.video_repo import VideoRepository
from api.routers import health, images, sessions, videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the SAM2 predictor and shared repositories on startup.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI while the server is running.
    """

    settings: Settings = get_settings()
    model_config = build_model_config(settings)

    logger.info("Loading SAM2 predictor (model_size=%s)…", settings.model_size)
    predictor = load_predictor(model_config)
    image_predictor = load_image_predictor(model_config)
    logger.info("SAM2 predictor ready on device %s.", predictor.device)

    app.state.settings = settings
    app.state.predictor = predictor
    app.state.image_predictor = image_predictor
    app.state.session_repo = SessionRepository()
    app.state.video_repo = VideoRepository()

    settings.data_path.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down — releasing SAM2 resources.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        Configured ``FastAPI`` instance ready to be served.
    """

    app = FastAPI(
        title="SAMURAI Tracking API",
        summary="Zero-shot video object tracking powered by SAM 2.1 with motion-aware memory.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(SessionNotFoundError, session_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(VideoNotFoundError, video_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(VideoProcessingError, video_processing_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PropagationError, propagation_error_handler)  # type: ignore[arg-type]

    app.include_router(health.router)
    app.include_router(images.router)
    app.include_router(videos.router)
    app.include_router(sessions.router)

    return app


app = create_app()
