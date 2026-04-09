"""Static image segmentation DTOs."""

from pydantic import BaseModel, Field

from api.schemas.tracking import BoundingBoxDTO, RLEMaskDTO


class ImageSegmentationPrompt(BaseModel):
    """A box prompt for segmenting one object in a static image."""

    label: str | None = None
    box: list[int] = Field(..., min_length=4, max_length=4)
    point: list[int] | None = Field(default=None, min_length=2, max_length=2)


class ImageSegmentationObjectDTO(BaseModel):
    """SAM2 segmentation result for one prompted object in a static image."""

    label: str | None = None
    bbox: BoundingBoxDTO | None
    mask: RLEMaskDTO | None
    polygon: list[list[int]] | None = None


class ImageSegmentationResponse(BaseModel):
    """Response with segmentation results for all prompts."""

    detections: list[ImageSegmentationObjectDTO] = Field(default_factory=list)
