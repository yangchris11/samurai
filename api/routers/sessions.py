"""Session management and tracking endpoints."""

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.config import Settings
from api.dependencies import get_predictor, get_session_repo, get_settings, get_video_repo
from api.ml import inference
from api.ml.loader import autocast_context
from api.repositories.session_repo import SessionRecord, SessionRepository
from api.repositories.video_repo import VideoRepository
from api.schemas.session import (
    AddBoxRequest,
    AddMaskRequest,
    AddPointsRequest,
    CloseSessionResponse,
    CreateSessionRequest,
    PropagateRequest,
    SessionDTO,
)
from api.schemas.tracking import (
    CancelPropagationResponse,
    ClearPromptsResponse,
    FramePromptsDTO,
    RemoveObjectResponse,
    TrackingFrameDTO,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])

_propagation_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="propagation")


def _session_to_dto(record: SessionRecord) -> SessionDTO:
    """Convert a ``SessionRecord`` to a ``SessionDTO``.

    Args:
        record: Active session record.

    Returns:
        API-facing session DTO.
    """

    return SessionDTO(
        session_id=record.session_id,
        video_id=record.video_id,
        num_frames=record.num_frames,
        created_at=record.created_at,
    )


async def _stream_propagation(
    predictor: SAM2VideoPredictor,
    record: SessionRecord,
    request: PropagateRequest,
    score_thresh: float,
    on_complete: "Callable[[], None] | None" = None,
) -> AsyncGenerator[bytes, None]:
    """Stream per-frame tracking results as NDJSON lines.

    Runs the synchronous SAM2 propagation generator in a thread pool and
    forwards results to the async response stream via a queue.

    Args:
        predictor: Loaded SAM2 video predictor.
        record: Active session containing inference state and cancellation flag.
        request: Propagation parameters (start frame, direction, max frames).
        score_thresh: Binary mask threshold.

    Yields:
        UTF-8 encoded NDJSON lines, one per frame.
    """

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[TrackingFrameDTO | BaseException | None] = asyncio.Queue(maxsize=16)

    def produce() -> None:
        try:
            ctx = autocast_context(predictor.device)
            with ctx:
                for frame_dto in inference.propagate_in_video(
                    predictor=predictor,
                    state=record.inference_state,
                    start_frame_index=request.start_frame_index,
                    direction=request.direction,
                    max_frames=request.max_frames,
                    score_thresh=score_thresh,
                ):
                    if record.canceled:
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(frame_dto), loop).result()
        except Exception as exc:  # noqa: BLE001
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    record.canceled = False
    _propagation_executor.submit(produce)

    while True:
        item = await queue.get()

        if item is None:
            break

        if isinstance(item, BaseException):
            logger.exception("Propagation error in session %s", record.session_id, exc_info=item)
            error_payload = {
                "error": {
                    "type": item.__class__.__name__,
                    "message": str(item),
                }
            }
            yield (json.dumps(error_payload) + "\n").encode()
            break

        yield (json.dumps(item.model_dump()) + "\n").encode()

    if on_complete is not None:
        on_complete()


@router.post(
    "",
    response_model=SessionDTO,
    status_code=201,
    summary="Create tracking session",
    description=(
        "Initialise a SAM2 tracking session for the given video. "
        "The video frames are loaded into GPU memory during this call."
    ),
)
async def create_session(
    body: CreateSessionRequest,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    video_repo: VideoRepository = Depends(get_video_repo),
) -> SessionDTO:
    """Create a new tracking session for the specified video."""

    video = video_repo.get(body.video_id)
    source_path = Path(video.video_path)
    if source_path.is_dir():
        init_path = source_path
        logger.info("Initialising session %s from frame directory %s", body.video_id, init_path)
    else:
        init_path = source_path
        logger.info("Initialising session %s directly from video file %s", body.video_id, init_path)

    offload = predictor.device.type == "mps"

    def _init() -> inference.InferenceState:
        ctx = autocast_context(predictor.device)
        with ctx:
            return inference.init_state(predictor, str(init_path), offload_video_to_cpu=offload)

    state = await asyncio.to_thread(_init)

    record = SessionRecord(
        session_id=str(uuid.uuid4()),
        video_id=body.video_id,
        inference_state=state,
        num_frames=state["num_frames"],
        created_at=datetime.now(timezone.utc),
    )
    session_repo.add(record)

    logger.info(
        "Session %s created for video %s (%d frames)",
        record.session_id, body.video_id, record.num_frames,
    )

    return _session_to_dto(record)


@router.delete(
    "/{session_id}",
    response_model=CloseSessionResponse,
    summary="Close tracking session",
    description="Release GPU memory and remove the session.",
)
def close_session(
    session_id: str,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    video_repo: VideoRepository = Depends(get_video_repo),
) -> CloseSessionResponse:
    """Close a session, release GPU resources, and delete associated video files."""

    session = session_repo.get(session_id)
    video_id = session.video_id

    success = session_repo.remove(session_id, predictor)

    if success:
        try:
            video = video_repo.get(video_id)
            upload_dir = Path(video.video_path).parent
            shutil.rmtree(upload_dir, ignore_errors=True)
            video_repo.remove(video_id)
        except Exception:
            pass

    logger.info("Session %s closed (found=%s)", session_id, success)

    return CloseSessionResponse(success=success)


@router.post(
    "/{session_id}/frames/{frame_index}/points",
    response_model=FramePromptsDTO,
    summary="Add point prompts",
    description=(
        "Add foreground/background point prompts to a specific frame. "
        "Returns the updated segmentation mask for that frame immediately."
    ),
)
def add_points(
    session_id: str,
    frame_index: int,
    body: AddPointsRequest,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    settings: Settings = Depends(get_settings),
) -> FramePromptsDTO:
    """Add point prompts and return the updated frame mask."""

    record = session_repo.get(session_id)

    ctx = autocast_context(predictor.device)
    with ctx:
        return inference.add_points(
            predictor=predictor,
            state=record.inference_state,
            frame_index=frame_index,
            object_id=body.object_id,
            points=body.points,
            labels=body.labels,
            clear_old_points=body.clear_old_points,
            normalize_coords=body.normalize_coords,
            score_thresh=settings.score_thresh,
        )


@router.post(
    "/{session_id}/frames/{frame_index}/box",
    response_model=FramePromptsDTO,
    summary="Add bounding box prompt",
    description=(
        "Add a bounding box prompt to a specific frame. "
        "Returns the updated segmentation mask for that frame immediately."
    ),
)
def add_box(
    session_id: str,
    frame_index: int,
    body: AddBoxRequest,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    settings: Settings = Depends(get_settings),
) -> FramePromptsDTO:
    """Add a bounding box prompt and return the updated frame mask."""

    record = session_repo.get(session_id)

    ctx = autocast_context(predictor.device)
    with ctx:
        return inference.add_box(
            predictor=predictor,
            state=record.inference_state,
            frame_index=frame_index,
            object_id=body.object_id,
            box=body.box,
            score_thresh=settings.score_thresh,
        )


@router.post(
    "/{session_id}/frames/{frame_index}/mask",
    response_model=FramePromptsDTO,
    summary="Add polygon mask prompt",
    description=(
        "Add a polygon-derived mask prompt to a specific frame. "
        "Returns the updated segmentation mask for that frame immediately."
    ),
)
def add_mask(
    session_id: str,
    frame_index: int,
    body: AddMaskRequest,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    settings: Settings = Depends(get_settings),
) -> FramePromptsDTO:
    """Add a polygon mask prompt and return the updated frame mask."""

    record = session_repo.get(session_id)
    video_width = int(record.inference_state["video_width"])
    video_height = int(record.inference_state["video_height"])

    mask_image = Image.new("L", (video_width, video_height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([(float(x), float(y)) for x, y in body.polygon], fill=1)
    mask = np.array(mask_image, dtype=np.uint8)

    ctx = autocast_context(predictor.device)
    with ctx:
        return inference.add_mask(
            predictor=predictor,
            state=record.inference_state,
            frame_index=frame_index,
            object_id=body.object_id,
            mask=mask,
            score_thresh=settings.score_thresh,
        )


@router.delete(
    "/{session_id}/frames/{frame_index}/prompts",
    response_model=FramePromptsDTO,
    summary="Clear frame prompts",
    description="Remove all prompts for one object on a specific frame.",
)
def clear_frame_prompts(
    session_id: str,
    frame_index: int,
    object_id: int,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    settings: Settings = Depends(get_settings),
) -> FramePromptsDTO:
    """Clear all prompts for one object on the given frame."""

    record = session_repo.get(session_id)

    ctx = autocast_context(predictor.device)
    with ctx:
        return inference.clear_frame_prompts(
            predictor=predictor,
            state=record.inference_state,
            frame_index=frame_index,
            object_id=object_id,
            score_thresh=settings.score_thresh,
        )


@router.delete(
    "/{session_id}/prompts",
    response_model=ClearPromptsResponse,
    summary="Clear all prompts",
    description="Remove all prompts across all frames in the session (reset tracking state).",
)
def clear_all_prompts(
    session_id: str,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> ClearPromptsResponse:
    """Reset all tracking prompts for the session."""

    record = session_repo.get(session_id)
    inference.reset_state(predictor, record.inference_state)

    return ClearPromptsResponse(success=True)


@router.delete(
    "/{session_id}/objects/{object_id}",
    response_model=RemoveObjectResponse,
    summary="Remove tracked object",
    description=(
        "Remove an object from the tracking state. "
        "Returns updated masks for all frames where the object appeared."
    ),
)
def remove_object(
    session_id: str,
    object_id: int,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    settings: Settings = Depends(get_settings),
) -> RemoveObjectResponse:
    """Remove an object and return updated per-frame masks."""

    record = session_repo.get(session_id)

    ctx = autocast_context(predictor.device)
    with ctx:
        updated_frames = inference.remove_object(
            predictor=predictor,
            state=record.inference_state,
            object_id=object_id,
            score_thresh=settings.score_thresh,
        )

    return RemoveObjectResponse(updated_frames=updated_frames)


@router.post(
    "/{session_id}/propagate",
    summary="Propagate tracking through video",
    description=(
        "Run SAM2 tracking from the prompted frames through the entire video. "
        "Results are streamed as **newline-delimited JSON** "
        "(``Content-Type: application/x-ndjson``), one object per line."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/x-ndjson": {}},
            "description": "Stream of ``TrackingFrameDTO`` objects, one per line.",
        }
    },
)
async def propagate(
    session_id: str,
    body: PropagateRequest,
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    session_repo: SessionRepository = Depends(get_session_repo),
    video_repo: VideoRepository = Depends(get_video_repo),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream per-frame tracking results as NDJSON."""

    record = session_repo.get(session_id)

    logger.info(
        "Starting propagation for session %s: direction=%s start=%d close_on_complete=%s",
        session_id, body.direction, body.start_frame_index, body.close_on_complete,
    )

    on_complete: Callable[[], None] | None = None
    if body.close_on_complete:
        def on_complete() -> None:
            video_id = record.video_id
            session_repo.remove(session_id, predictor)
            try:
                video = video_repo.get(video_id)
                shutil.rmtree(Path(video.video_path).parent, ignore_errors=True)
                video_repo.remove(video_id)
            except Exception:
                pass
            logger.info("Session %s auto-closed after propagation", session_id)

    return StreamingResponse(
        _stream_propagation(predictor, record, body, settings.score_thresh, on_complete),
        media_type="application/x-ndjson",
    )


@router.delete(
    "/{session_id}/propagate",
    response_model=CancelPropagationResponse,
    summary="Cancel propagation",
    description="Signal an in-progress propagation to stop at the next frame boundary.",
)
def cancel_propagation(
    session_id: str,
    session_repo: SessionRepository = Depends(get_session_repo),
) -> CancelPropagationResponse:
    """Cancel an active propagation for the given session."""

    session_repo.mark_canceled(session_id)

    logger.info("Propagation canceled for session %s", session_id)

    return CancelPropagationResponse(success=True)
