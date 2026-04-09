"""SAM2 predictor loading for video and image inference."""

import contextlib
import logging
from pathlib import Path

import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.config import Settings
from api.ml.config import ModelConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_NAMES: dict[str, str] = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}

_MODEL_CFGS: dict[str, str] = {
    "tiny": "configs/samurai/sam2.1_hiera_t.yaml",
    "small": "configs/samurai/sam2.1_hiera_s.yaml",
    "base_plus": "configs/samurai/sam2.1_hiera_b+.yaml",
    "large": "configs/samurai/sam2.1_hiera_l.yaml",
}


def build_model_config(settings: Settings) -> ModelConfig:
    """Build a ``ModelConfig`` from application settings.

    Args:
        settings: Application settings containing ``app_root``, ``model_size``,
            ``device``, ``dtype``, and ``score_thresh``.

    Returns:
        Fully resolved ``ModelConfig`` ready to be passed to ``load_predictor``.

    Raises:
        ValueError: If ``settings.model_size`` is not a supported variant.
    """
    if settings.model_size not in _CHECKPOINT_NAMES:
        raise ValueError(
            f"Unsupported model_size '{settings.model_size}'. "
            f"Choose one of: {sorted(_CHECKPOINT_NAMES)}"
        )

    checkpoint_path = (
        settings.app_root / "checkpoints" / _CHECKPOINT_NAMES[settings.model_size]
    )

    return ModelConfig(
        checkpoint_path=checkpoint_path,
        model_cfg=_MODEL_CFGS[settings.model_size],
        device=settings.device,
        dtype=settings.dtype,
        score_thresh=settings.score_thresh,
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve the best available device, falling back gracefully.

    If the requested device is unavailable, falls back to MPS, then CPU.

    Args:
        requested: PyTorch device string (e.g. ``cuda``, ``cuda:0``).

    Returns:
        The best available ``torch.device``.
    """
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if torch.backends.mps.is_available():
        logger.warning("CUDA not available; falling back to MPS.")
        return torch.device("mps")
    logger.warning("GPU not available; falling back to CPU.")
    return torch.device("cpu")


def configure_gpu(device: torch.device) -> None:
    """Apply GPU-specific performance settings.

    Enables TF32 on Ampere-class CUDA GPUs and logs a warning for MPS devices.

    Args:
        device: The resolved PyTorch device.
    """
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        logger.warning(
            "MPS support is preliminary. SAM2 is trained with CUDA and may produce "
            "numerically different results on MPS."
        )


def load_predictor(config: ModelConfig) -> SAM2VideoPredictor:
    """Load the SAM2 video predictor from checkpoint.

    Args:
        config: Model configuration with checkpoint path, hydra config name,
            device, and dtype settings.

    Returns:
        A loaded ``SAM2VideoPredictor`` in eval mode on the target device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If checkpoint loading fails.
    """
    checkpoint: Path = config.checkpoint_path
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint}'. "
            "Run the SAM2 download script first."
        )

    device = resolve_device(config.device)
    configure_gpu(device)

    logger.info(
        f"Loading SAM2 predictor: cfg={config.model_cfg} "
        f"ckpt={checkpoint.name} device={device}"
    )

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        config.model_cfg,
        str(checkpoint),
        device=device,
    )

    logger.info("SAM2 predictor loaded successfully.")
    return predictor


def load_image_predictor(config: ModelConfig) -> SAM2ImagePredictor:
    """Load the SAM2 image predictor from checkpoint."""
    checkpoint: Path = config.checkpoint_path
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint}'. "
            "Run the SAM2 download script first."
        )

    device = resolve_device(config.device)
    configure_gpu(device)

    logger.info(
        "Loading SAM2 image predictor: cfg=%s ckpt=%s device=%s",
        config.model_cfg,
        checkpoint.name,
        device,
    )

    model = build_sam2(
        config.model_cfg,
        str(checkpoint),
        device=device,
    )
    predictor = SAM2ImagePredictor(model, mask_threshold=config.score_thresh)
    logger.info("SAM2 image predictor loaded successfully.")
    return predictor


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:  # type: ignore[type-arg]
    """Return the appropriate autocast context for the given device.

    Args:
        device: Target PyTorch device.

    Returns:
        A ``torch.autocast`` context for CUDA, or a no-op context otherwise.
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
