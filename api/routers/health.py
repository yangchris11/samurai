"""Health check router."""

import torch
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthDTO(BaseModel):
    """Server health status.

    Attributes:
        status: Always ``"ok"`` when the server is reachable.
        cuda_available: Whether a CUDA-capable GPU is visible to PyTorch.
        gpu_memory_allocated_mb: GPU memory currently allocated in MiB (0 if no CUDA).
        gpu_memory_reserved_mb: GPU memory currently reserved in MiB (0 if no CUDA).
    """

    status: str
    cuda_available: bool
    gpu_memory_allocated_mb: int
    gpu_memory_reserved_mb: int


@router.get(
    "/health",
    response_model=HealthDTO,
    summary="Health check",
    description="Return server status and basic GPU memory statistics.",
)
def health() -> HealthDTO:
    """Return server health and GPU memory usage."""

    cuda = torch.cuda.is_available()
    allocated = torch.cuda.memory_allocated() // 1024**2 if cuda else 0
    reserved = torch.cuda.memory_reserved() // 1024**2 if cuda else 0

    return HealthDTO(
        status="ok",
        cuda_available=cuda,
        gpu_memory_allocated_mb=allocated,
        gpu_memory_reserved_mb=reserved,
    )
