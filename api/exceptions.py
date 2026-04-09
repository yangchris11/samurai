"""Domain exceptions and FastAPI exception handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse


class SessionNotFoundError(Exception):
    """Raised when a session_id does not exist in the repository."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' not found")
        self.session_id = session_id


class VideoNotFoundError(Exception):
    """Raised when a video_id does not exist in the repository."""

    def __init__(self, video_id: str) -> None:
        super().__init__(f"Video '{video_id}' not found")
        self.video_id = video_id


class VideoProcessingError(Exception):
    """Raised when video upload or FFmpeg processing fails."""


class PropagationError(Exception):
    """Raised when SAM2 propagation fails."""


async def session_not_found_handler(
    _request: Request, exc: SessionNotFoundError
) -> JSONResponse:
    """Return 404 JSON response for missing sessions."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def video_not_found_handler(
    _request: Request, exc: VideoNotFoundError
) -> JSONResponse:
    """Return 404 JSON response for missing videos."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def video_processing_error_handler(
    _request: Request, exc: VideoProcessingError
) -> JSONResponse:
    """Return 422 JSON response for video processing failures."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def propagation_error_handler(
    _request: Request, exc: PropagationError
) -> JSONResponse:
    """Return 500 JSON response for propagation failures."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})
