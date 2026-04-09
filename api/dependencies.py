"""FastAPI dependency factories.

All dependencies are injected from the application state set during lifespan.
"""

from fastapi import Request
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.config import Settings
from api.repositories.session_repo import SessionRepository
from api.repositories.video_repo import VideoRepository


def get_settings(request: Request) -> Settings:
    """Return application settings from app state.

    Args:
        request: Current HTTP request.

    Returns:
        Application ``Settings`` instance.
    """

    return request.app.state.settings


def get_predictor(request: Request) -> SAM2VideoPredictor:
    """Return the shared SAM2 video predictor from app state.

    Args:
        request: Current HTTP request.

    Returns:
        Loaded ``SAM2VideoPredictor``.
    """

    return request.app.state.predictor


def get_image_predictor(request: Request) -> SAM2ImagePredictor:
    """Return the shared SAM2 image predictor from app state."""

    return request.app.state.image_predictor


def get_session_repo(request: Request) -> SessionRepository:
    """Return the session repository from app state.

    Args:
        request: Current HTTP request.

    Returns:
        Application-scoped ``SessionRepository``.
    """

    return request.app.state.session_repo


def get_video_repo(request: Request) -> VideoRepository:
    """Return the video repository from app state.

    Args:
        request: Current HTTP request.

    Returns:
        Application-scoped ``VideoRepository``.
    """

    return request.app.state.video_repo
