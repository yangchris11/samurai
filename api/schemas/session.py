"""Session resource DTOs."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request body for creating a new tracking session.

    Attributes:
        video_id: ID of the video to track objects in.
    """

    video_id: str


class SessionDTO(BaseModel):
    """Active tracking session metadata.

    Attributes:
        session_id: Unique session identifier (UUID).
        video_id: ID of the video associated with the session.
        num_frames: Total number of frames in the video.
        created_at: UTC timestamp when the session was created.
    """

    session_id: str
    video_id: str
    num_frames: int
    created_at: datetime


class CloseSessionResponse(BaseModel):
    """Result of a session close request.

    Attributes:
        success: Whether the session was found and removed.
    """

    success: bool


class AddPointsRequest(BaseModel):
    """Request body for adding point prompts to a frame.

    Points use image-space coordinates (pixels). Provide either ``points``
    or ``box``, not both.

    Attributes:
        object_id: Identifier for the object being tracked (must be >= 0).
        points: List of (x, y) pixel coordinates.
        labels: Corresponding point labels: 1 = positive (foreground), 0 = negative (background).
        clear_old_points: Replace any previously added points for this object on this frame.
        normalize_coords: Treat coordinates as normalised [0, 1] fractions instead of pixels.
    """

    object_id: int = Field(ge=0)
    points: list[tuple[float, float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)
    clear_old_points: bool = True
    normalize_coords: bool = False

    model_config = {"json_schema_extra": {"example": {
        "object_id": 0,
        "points": [[320, 240]],
        "labels": [1],
        "clear_old_points": True,
        "normalize_coords": False,
    }}}


class AddBoxRequest(BaseModel):
    """Request body for adding a bounding box prompt to a frame.

    Attributes:
        object_id: Identifier for the object being tracked (must be >= 0).
        box: Bounding box as ``[x_min, y_min, x_max, y_max]`` in pixels.
    """

    object_id: int = Field(ge=0)
    box: tuple[float, float, float, float]

    model_config = {"json_schema_extra": {"example": {
        "object_id": 0,
        "box": [100, 80, 300, 260],
    }}}


class AddMaskRequest(BaseModel):
    """Request body for adding a polygon mask prompt to a frame.

    Attributes:
        object_id: Identifier for the object being tracked (must be >= 0).
        polygon: Polygon points as ``[[x, y], ...]`` in pixels.
    """

    object_id: int = Field(ge=0)
    polygon: list[tuple[float, float]] = Field(min_length=3)

    model_config = {"json_schema_extra": {"example": {
        "object_id": 0,
        "polygon": [[100, 80], [300, 90], [280, 260], [120, 240]],
    }}}


class PropagateRequest(BaseModel):
    """Request body for video propagation.

    Attributes:
        start_frame_index: Frame index to start propagation from (default: 0).
        direction: Propagation direction — ``forward``, ``backward``, or ``both``.
        max_frames: Maximum number of frames to track; ``null`` means all frames.
    """

    start_frame_index: int = Field(default=0, ge=0)
    direction: str = Field(default="both", pattern="^(forward|backward|both)$")
    max_frames: int | None = Field(default=None, gt=0)
    close_on_complete: bool = Field(default=True)
