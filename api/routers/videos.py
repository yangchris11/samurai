"""Video upload and listing endpoints."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from api.config import Settings
from api.dependencies import get_settings, get_video_repo
from api.exceptions import VideoProcessingError
from api.repositories.video_repo import VideoRecord, VideoRepository
from api.schemas.video import (
    ChunkedUploadChunkResponse,
    ChunkedUploadCompleteRequest,
    ChunkedUploadInitRequest,
    ChunkedUploadInitResponse,
    VideoDTO,
    VideoListDTO,
    VideoUploadResponse,
)

router = APIRouter(prefix="/videos", tags=["Videos"])

_MAX_CHUNK_SIZE_BYTES = 1024 * 1024


def _video_to_dto(record: VideoRecord) -> VideoDTO:
    """Convert a ``VideoRecord`` to a ``VideoDTO``.

    Args:
        record: Stored video metadata.

    Returns:
        API-facing video DTO with URL fields populated.
    """

    return VideoDTO(
        id=record.id,
        filename=record.filename,
        width=record.width,
        height=record.height,
        duration_sec=record.duration_sec,
        frame_count=record.frame_count,
        url=f"/videos/{record.id}/stream",
        poster_url=f"/videos/{record.id}/poster",
    )


def _upload_dir(base_path: Path, upload_id: str) -> Path:
    return base_path / ".chunk_uploads" / upload_id


def _upload_meta_path(base_path: Path, upload_id: str) -> Path:
    return _upload_dir(base_path, upload_id) / "meta.json"


def _load_upload_meta(base_path: Path, upload_id: str) -> dict[str, object]:
    meta_path = _upload_meta_path(base_path, upload_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found.")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_upload_meta(base_path: Path, upload_id: str, meta: dict[str, object]) -> None:
    meta_path = _upload_meta_path(base_path, upload_id)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _probe_video(video_path: Path) -> tuple[int, int, float, int]:
    """Extract width, height, duration, and frame count from a video file using ffprobe.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Tuple of ``(width, height, duration_sec, frame_count)``.

    Raises:
        VideoProcessingError: If ffprobe fails or returns unexpected output.
    """

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,nb_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(
            f"ffprobe failed for '{video_path.name}': {exc.stderr.strip()}"
        ) from exc

    parts = result.stdout.strip().split(",")

    try:
        width, height, duration = int(parts[0]), int(parts[1]), float(parts[2])
    except (IndexError, ValueError) as exc:
        raise VideoProcessingError(
            f"Unexpected ffprobe output for '{video_path.name}': {result.stdout!r}"
        ) from exc

    try:
        frame_count = int(parts[3])
    except (IndexError, ValueError):
        frame_count = 0

    return width, height, duration, frame_count



def _extract_poster(video_path: Path, poster_path: Path) -> None:
    """Extract the first frame of a video as a JPEG poster image.

    Args:
        video_path: Source video file path.
        poster_path: Destination JPEG path.

    Raises:
        VideoProcessingError: If ffmpeg fails to extract the frame.
    """

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(poster_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(
            f"ffmpeg poster extraction failed for '{video_path.name}': {exc.stderr.decode().strip()}"
        ) from exc


def _trim_video(
    source: Path,
    dest: Path,
    start_sec: float,
    duration_sec: float | None,
    ffmpeg_threads: int,
) -> None:
    """Trim a video to the specified time range and re-encode to dest.

    Args:
        source: Input video path.
        dest: Output video path.
        start_sec: Start offset in seconds.
        duration_sec: Duration to keep; ``None`` means until end of file.
        ffmpeg_threads: Number of threads for ffmpeg.

    Raises:
        VideoProcessingError: If ffmpeg trimming fails.
    """

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(source),
        "-threads", str(ffmpeg_threads),
        "-c", "copy",
    ]

    if duration_sec is not None:
        cmd += ["-t", str(duration_sec)]

    cmd.append(str(dest))

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(
            f"ffmpeg trim failed for '{source.name}': {exc.stderr.decode().strip()}"
        ) from exc


def _finalize_uploaded_video(
    source_path: Path,
    filename: str,
    start_sec: float,
    duration_sec: float | None,
    settings: Settings,
    video_repo: VideoRepository,
) -> VideoUploadResponse:
    video_id = str(uuid.uuid4())
    upload_dir: Path = settings.data_path / video_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename or "video.mp4").suffix.lower() or ".mp4"
    raw_path = upload_dir / f"raw{suffix}"
    final_path = upload_dir / "video.mp4"
    poster_path = upload_dir / "poster.jpg"

    try:
        shutil.move(str(source_path), str(raw_path))
    except OSError as exc:
        raise VideoProcessingError(
            f"Failed to save uploaded file '{filename}': {exc}"
        ) from exc

    needs_convert = suffix != ".mp4"
    needs_trim = start_sec > 0.0 or duration_sec is not None
    if needs_trim or needs_convert:
        _trim_video(raw_path, final_path, start_sec, duration_sec, settings.ffmpeg_threads)
        raw_path.unlink()
    else:
        raw_path.rename(final_path)

    width, height, actual_duration, frame_count = _probe_video(final_path)

    if actual_duration > settings.max_upload_duration_sec:
        shutil.rmtree(upload_dir)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Video duration {actual_duration:.1f}s exceeds the "
                f"limit of {settings.max_upload_duration_sec:.0f}s."
            ),
        )

    _extract_poster(final_path, poster_path)

    frames_dir = upload_dir / "frames"

    record = VideoRecord(
        id=video_id,
        filename=filename or "video.mp4",
        video_path=str(final_path),
        frames_path=str(frames_dir),
        poster_path=str(poster_path),
        width=width,
        height=height,
        duration_sec=actual_duration,
        frame_count=frame_count,
    )
    video_repo.add(record)

    return VideoUploadResponse(video=_video_to_dto(record))


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=201,
    summary="Upload a video",
    description=(
        "Upload an MP4 video for tracking. "
        "Optional ``start_sec`` and ``duration_sec`` trim the video before storing."
    ),
)
async def upload_video(
    file: UploadFile,
    start_sec: float = Form(default=0.0, ge=0.0),
    duration_sec: float | None = Form(default=None, gt=0.0),
    settings: Settings = Depends(get_settings),
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoUploadResponse:
    """Accept a video upload, optionally trim it, and store metadata."""

    try:
        temp_dir = settings.data_path / ".single_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4()}{Path(file.filename or 'video.mp4').suffix.lower() or '.mp4'}"
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as exc:
        raise VideoProcessingError(
            f"Failed to save uploaded file '{file.filename}': {exc}"
        ) from exc

    return _finalize_uploaded_video(
        source_path=temp_path,
        filename=file.filename or "video.mp4",
        start_sec=start_sec,
        duration_sec=duration_sec,
        settings=settings,
        video_repo=video_repo,
    )


@router.post(
    "/upload/init",
    response_model=ChunkedUploadInitResponse,
    status_code=201,
    summary="Start a chunked video upload",
)
def init_chunked_upload(
    body: ChunkedUploadInitRequest,
    settings: Settings = Depends(get_settings),
) -> ChunkedUploadInitResponse:
    upload_id = str(uuid.uuid4())
    upload_dir = _upload_dir(settings.data_path, upload_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    _save_upload_meta(settings.data_path, upload_id, {
        "filename": body.filename,
        "start_sec": body.start_sec,
        "duration_sec": body.duration_sec,
        "received_chunks": [],
    })
    return ChunkedUploadInitResponse(
        upload_id=upload_id,
        chunk_size_bytes=_MAX_CHUNK_SIZE_BYTES,
    )


@router.put(
    "/upload/{upload_id}/chunk/{chunk_index}",
    response_model=ChunkedUploadChunkResponse,
    summary="Upload one video chunk",
)
async def upload_video_chunk(
    upload_id: str,
    chunk_index: int,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ChunkedUploadChunkResponse:
    if chunk_index < 0:
        raise HTTPException(status_code=422, detail="chunk_index must be >= 0.")

    chunk = await request.body()
    if not chunk:
        raise HTTPException(status_code=422, detail="Chunk body is empty.")
    if len(chunk) > _MAX_CHUNK_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Chunk exceeds {_MAX_CHUNK_SIZE_BYTES} bytes.",
        )

    meta = _load_upload_meta(settings.data_path, upload_id)
    upload_dir = _upload_dir(settings.data_path, upload_id)
    chunk_path = upload_dir / f"{chunk_index:08d}.part"
    chunk_path.write_bytes(chunk)

    received_chunks = {int(idx) for idx in meta.get("received_chunks", [])}
    received_chunks.add(chunk_index)
    meta["received_chunks"] = sorted(received_chunks)
    _save_upload_meta(settings.data_path, upload_id, meta)

    return ChunkedUploadChunkResponse(
        upload_id=upload_id,
        chunk_index=chunk_index,
        received_chunks=len(received_chunks),
    )


@router.post(
    "/upload/{upload_id}/complete",
    response_model=VideoUploadResponse,
    summary="Finalize a chunked video upload",
)
def complete_chunked_upload(
    upload_id: str,
    body: ChunkedUploadCompleteRequest = Body(...),
    settings: Settings = Depends(get_settings),
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoUploadResponse:
    meta = _load_upload_meta(settings.data_path, upload_id)
    upload_dir = _upload_dir(settings.data_path, upload_id)
    received_chunks = [int(idx) for idx in meta.get("received_chunks", [])]
    expected_chunks = list(range(body.total_chunks))
    if received_chunks != expected_chunks:
        raise HTTPException(
            status_code=422,
            detail="Missing uploaded chunks; cannot finalize upload.",
        )

    suffix = Path(body.filename or "video.mp4").suffix.lower() or ".mp4"
    assembled_path = upload_dir / f"assembled{suffix}"
    with assembled_path.open("wb") as output:
        for chunk_index in expected_chunks:
            chunk_path = upload_dir / f"{chunk_index:08d}.part"
            with chunk_path.open("rb") as chunk_file:
                shutil.copyfileobj(chunk_file, output)

    try:
        return _finalize_uploaded_video(
            source_path=assembled_path,
            filename=body.filename,
            start_sec=body.start_sec,
            duration_sec=body.duration_sec,
            settings=settings,
            video_repo=video_repo,
        )
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _probe_frames(frames_dir: Path) -> tuple[int, int, int]:
    """Read width, height, and frame count from a directory of JPEG frames.

    Args:
        frames_dir: Directory containing zero-padded JPEG files.

    Returns:
        Tuple of ``(width, height, frame_count)``.

    Raises:
        VideoProcessingError: If the directory is empty or the first frame is unreadable.
    """

    frames = sorted(frames_dir.glob("*.jpg"))

    if not frames:
        raise VideoProcessingError(f"No JPEG frames found in '{frames_dir}'.")

    try:
        with Image.open(frames[0]) as img:
            width, height = img.size
    except OSError as exc:
        raise VideoProcessingError(
            f"Cannot read first frame '{frames[0].name}': {exc}"
        ) from exc

    return width, height, len(frames)


def _save_frames(files: list[UploadFile], frames_dir: Path) -> Path:
    """Save uploaded image files as sorted zero-padded JPEGs.

    Files are sorted by their original filename before saving so that the
    caller controls ordering by naming convention (e.g. ``frame_001.jpg``).

    Args:
        files: Uploaded image files.
        frames_dir: Destination directory (must already exist).

    Returns:
        Path to the first saved frame (used as the poster).

    Raises:
        VideoProcessingError: If any file cannot be read or saved.
    """

    sorted_files = sorted(files, key=lambda f: f.filename or "")
    first_frame_path: Path | None = None

    for idx, upload in enumerate(sorted_files):
        dest = frames_dir / f"{idx:05d}.jpg"

        try:
            with Image.open(upload.file) as img:
                rgb = img.convert("RGB")
                rgb.save(dest, format="JPEG", quality=95)
        except OSError as exc:
            raise VideoProcessingError(
                f"Failed to save frame '{upload.filename}': {exc}"
            ) from exc

        if idx == 0:
            first_frame_path = dest

    if first_frame_path is None:
        raise VideoProcessingError("No frames were saved.")

    return first_frame_path


@router.post(
    "/upload-frames",
    response_model=VideoUploadResponse,
    status_code=201,
    summary="Upload a frame sequence",
    description=(
        "Upload N image files as an ordered frame sequence for tracking. "
        "Files are sorted by filename before saving — use zero-padded names "
        "(e.g. ``frame_000.jpg``, ``frame_001.jpg``) to control order. "
        "JPEG, PNG, and WebP are accepted."
    ),
)
async def upload_frames(
    files: list[UploadFile],
    settings: Settings = Depends(get_settings),
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoUploadResponse:
    """Accept a frame sequence upload and store it as a trackable video source."""

    if not files:
        raise HTTPException(status_code=422, detail="At least one frame is required.")

    for upload in files:
        if upload.content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported media type '{upload.content_type}' for file "
                    f"'{upload.filename}'. Upload JPEG, PNG, or WebP images."
                ),
            )

    video_id = str(uuid.uuid4())
    frames_dir: Path = settings.data_path / video_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    poster_path = settings.data_path / video_id / "poster.jpg"

    first_frame = _save_frames(files, frames_dir)
    width, height, frame_count = _probe_frames(frames_dir)

    shutil.copy(first_frame, poster_path)

    record = VideoRecord(
        id=video_id,
        filename=f"{frame_count}_frames",
        video_path=str(frames_dir),
        frames_path=str(frames_dir),
        poster_path=str(poster_path),
        width=width,
        height=height,
        duration_sec=None,
        frame_count=frame_count,
    )
    video_repo.add(record)

    return VideoUploadResponse(video=_video_to_dto(record))


@router.get(
    "",
    response_model=VideoListDTO,
    summary="List all videos",
    description="Return metadata for every uploaded video.",
)
def list_videos(
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoListDTO:
    """Return all stored video records."""

    records = video_repo.list_all()

    return VideoListDTO(
        items=[_video_to_dto(r) for r in records],
        total=len(records),
    )


@router.get(
    "/{video_id}",
    response_model=VideoDTO,
    summary="Get video metadata",
    description="Return metadata for a single video by ID.",
)
def get_video(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoDTO:
    """Return metadata for the requested video."""

    record = video_repo.get(video_id)
    return _video_to_dto(record)


@router.get(
    "/{video_id}/stream",
    summary="Stream video file",
    description="Serve the raw video file for playback.",
    response_class=FileResponse,
)
def stream_video(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repo),
) -> FileResponse:
    """Serve the video file for the requested ID."""

    record = video_repo.get(video_id)

    if Path(record.video_path).is_dir():
        raise HTTPException(
            status_code=422,
            detail="This video is a frame sequence and cannot be streamed.",
        )

    return FileResponse(record.video_path, media_type="video/mp4")


@router.get(
    "/{video_id}/poster",
    summary="Get poster image",
    description="Serve the first-frame poster JPEG for a video.",
    response_class=FileResponse,
)
def get_poster(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repo),
) -> FileResponse:
    """Serve the poster image for the requested video."""

    record = video_repo.get(video_id)
    return FileResponse(record.poster_path, media_type="image/jpeg")
