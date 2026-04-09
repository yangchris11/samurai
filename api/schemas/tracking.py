"""Tracking result DTOs."""

from pydantic import BaseModel, Field


class RLEMaskDTO(BaseModel):
    """Run-length encoded segmentation mask (pycocotools format).

    Attributes:
        size: Mask dimensions as ``[height, width]``.
        counts: RLE-encoded mask bytes decoded as UTF-8 string.
    """

    size: list[int] = Field(min_length=2, max_length=2)
    counts: str


class BoundingBoxDTO(BaseModel):
    """Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x: Left edge in pixels.
        y: Top edge in pixels.
        width: Box width in pixels.
        height: Box height in pixels.
    """

    x: int
    y: int
    width: int
    height: int


class ObjectTrackDTO(BaseModel):
    """Tracking result for a single object on a single frame.

    Attributes:
        object_id: Identifier of the tracked object.
        mask: RLE-encoded segmentation mask; ``null`` when the object is occluded.
        bbox: Tight bounding box derived from the mask; ``null`` when mask is empty.
    """

    object_id: int
    mask: RLEMaskDTO | None
    bbox: BoundingBoxDTO | None
    polygon: list[list[int]] | None = None


class TrackingFrameDTO(BaseModel):
    """Tracking results for all objects on a single video frame.

    Attributes:
        frame_index: Zero-based frame index.
        objects: Per-object tracking results.
    """

    frame_index: int
    objects: list[ObjectTrackDTO]


class FramePromptsDTO(BaseModel):
    """Masks returned immediately after adding prompts to a frame.

    This is the same shape as ``TrackingFrameDTO`` and represents the
    updated segmentation on the prompted frame only.

    Attributes:
        frame_index: Zero-based frame index that was prompted.
        objects: Per-object segmentation results on that frame.
    """

    frame_index: int
    objects: list[ObjectTrackDTO]


class ClearPromptsResponse(BaseModel):
    """Response after clearing prompts from a session.

    Attributes:
        success: Whether prompts were successfully removed.
    """

    success: bool


class RemoveObjectResponse(BaseModel):
    """Response after removing an object from the tracking state.

    Contains updated masks for all frames where the object appeared,
    reflecting the remaining objects after removal.

    Attributes:
        updated_frames: Per-frame tracking results after the object was removed.
    """

    updated_frames: list[TrackingFrameDTO]


class CancelPropagationResponse(BaseModel):
    """Response after cancelling an in-progress propagation.

    Attributes:
        success: Whether the cancellation signal was sent.
    """

    success: bool
