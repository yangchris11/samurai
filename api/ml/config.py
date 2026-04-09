"""ML model configuration types."""

from pathlib import Path

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """SAM2/SAMURAI video predictor configuration.

    Attributes:
        checkpoint_path: Absolute path to the ``.pt`` checkpoint file.
        model_cfg: Hydra config name relative to the sam2 package config dir.
        device: PyTorch device string (e.g. ``cuda``, ``cuda:0``, ``cpu``).
        dtype: Autocast dtype string — ``float16`` or ``bfloat16``.
        score_thresh: Minimum logit threshold for binary mask extraction.
    """

    checkpoint_path: Path
    model_cfg: str
    device: str
    dtype: str = "bfloat16"
    score_thresh: float = 0.0
