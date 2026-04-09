"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SAMURAI API runtime configuration.

    Attributes:
        app_root: Project root directory containing ``sam2/`` and ``checkpoints/``.
        model_size: SAM2 model variant — tiny, small, base_plus, or large.
        device: PyTorch device string (``cuda``, ``cuda:0``, ``mps``, ``cpu``).
        dtype: Autocast dtype for CUDA inference — ``float16`` or ``bfloat16``.
        score_thresh: Minimum mask score threshold for binary mask extraction.
        data_path: Root directory for uploaded videos and posters.
        max_upload_duration_sec: Maximum allowed video length in seconds.
        ffmpeg_threads: FFmpeg thread count for video processing.
    """

    model_config = SettingsConfigDict(env_prefix="SAMURAI_", env_file=".env", env_file_encoding="utf-8")

    app_root: Path = Path("sam2")
    model_size: str = "base_plus"
    device: str = "cuda"
    dtype: str = "bfloat16"
    score_thresh: float = 0.0
    data_path: Path = Path("data/uploads")
    max_upload_duration_sec: float = 10.0
    ffmpeg_threads: int = 4


def get_settings() -> Settings:
    """Build application settings from environment variables."""
    return Settings()
