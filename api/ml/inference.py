"""Pure inference functions wrapping SAM2 predictors.

All functions are stateless with respect to the predictor — the mutable
``inference_state`` is owned by the caller (session repository) and passed
explicitly. IO and device context management are handled at the call site.
"""

import logging
from collections.abc import Generator
from typing import TypedDict

import cv2
import numpy as np
import pycocotools.mask as mask_util
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.schemas.image import ImageSegmentationObjectDTO
from api.schemas.tracking import (
    BoundingBoxDTO,
    FramePromptsDTO,
    ObjectTrackDTO,
    RLEMaskDTO,
    TrackingFrameDTO,
)

logger = logging.getLogger(__name__)


class InferenceState(TypedDict):
    """Opaque SAM2 session state returned by ``predictor.init_state``."""

    num_frames: int
    obj_ids: list[int]


def init_state(
    predictor: SAM2VideoPredictor,
    video_path: str,
    offload_video_to_cpu: bool = False,
) -> InferenceState:
    """Initialise per-video inference state.

    Args:
        predictor: Loaded SAM2 video predictor.
        video_path: Absolute path to the video file or a directory of JPEG frames.
        offload_video_to_cpu: Offload frame tensors to CPU to reduce GPU VRAM usage.

    Returns:
        Initialised inference state dict owned by the caller.
    """

    state: InferenceState = predictor.init_state(
        video_path,
        offload_video_to_cpu=offload_video_to_cpu,
    )
    return state


def reset_state(predictor: SAM2VideoPredictor, state: InferenceState) -> None:
    """Clear all prompts in a session without releasing frame memory.

    Args:
        predictor: Loaded SAM2 video predictor.
        state: Inference state to reset.
    """

    predictor.reset_state(state)


def add_points(
    predictor: SAM2VideoPredictor,
    state: InferenceState,
    frame_index: int,
    object_id: int,
    points: list[tuple[float, float]],
    labels: list[int],
    clear_old_points: bool,
    normalize_coords: bool,
    score_thresh: float,
) -> FramePromptsDTO:
    """Add point prompts to a frame and return the updated mask.

    Args:
        predictor: Loaded SAM2 video predictor.
        state: Active session inference state.
        frame_index: Zero-based frame index to add prompts on.
        object_id: Object identifier to associate with the prompts.
        points: List of ``(x, y)`` pixel (or normalised) coordinates.
        labels: Per-point labels; 1 = foreground, 0 = background.
        clear_old_points: Replace existing prompts for this object on this frame.
        normalize_coords: Treat ``points`` as normalised [0, 1] fractions.
        score_thresh: Binary threshold for mask logits.

    Returns:
        Updated segmentation result for the prompted frame.
    """

    points_arr = np.array(points, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_index,
        obj_id=object_id,
        points=points_arr,
        labels=labels_arr,
        clear_old_points=clear_old_points,
        normalize_coords=normalize_coords,
    )

    masks_binary = (masks > score_thresh)[:, 0].cpu().numpy()

    return FramePromptsDTO(
        frame_index=int(frame_idx),
        objects=_build_object_tracks(object_ids, masks_binary),
    )


def add_box(
    predictor: SAM2VideoPredictor,
    state: InferenceState,
    frame_index: int,
    object_id: int,
    box: tuple[float, float, float, float],
    score_thresh: float,
) -> FramePromptsDTO:
    """Add a bounding box prompt to a frame and return the updated mask.

    Args:
        predictor: Loaded SAM2 video predictor.
        state: Active session inference state.
        frame_index: Zero-based frame index to add the prompt on.
        object_id: Object identifier to associate with the box.
        box: Bounding box as ``(x_min, y_min, x_max, y_max)`` in pixels.
        score_thresh: Binary threshold for mask logits.

    Returns:
        Updated segmentation result for the prompted frame.
    """

    box_arr = np.array(box, dtype=np.float32)

    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_index,
        obj_id=object_id,
        box=box_arr,
    )

    masks_binary = (masks > score_thresh)[:, 0].cpu().numpy()

    return FramePromptsDTO(
        frame_index=int(frame_idx),
        objects=_build_object_tracks(object_ids, masks_binary),
    )


def add_mask(
    predictor: SAM2VideoPredictor,
    state: InferenceState,
    frame_index: int,
    object_id: int,
    mask: np.ndarray,
    score_thresh: float,
) -> FramePromptsDTO:
    """Add a binary mask prompt to a frame and return the updated mask."""

    frame_idx, object_ids, masks = predictor.add_new_mask(
        inference_state=state,
        frame_idx=frame_index,
        obj_id=object_id,
        mask=mask.astype(np.uint8),
    )

    masks_binary = (masks > score_thresh)[:, 0].cpu().numpy()

    return FramePromptsDTO(
        frame_index=int(frame_idx),
        objects=_build_object_tracks(object_ids, masks_binary),
    )


def clear_frame_prompts(
    predictor: SAM2VideoPredictor,
    state: InferenceState,
    frame_index: int,
    object_id: int,
    score_thresh: float,
) -> FramePromptsDTO:
    """Clear all prompts for one object on a specific frame.

    Args:
        predictor: Loaded SAM2 video predictor.
        state: Active session inference state.
        frame_index: Zero-based frame index to clear.
        object_id: Object whose prompts are removed.
        score_thresh: Binary threshold for mask logits.

    Returns:
        Updated segmentation result for the affected frame.
    """

    frame_idx, object_ids, masks = predictor.clear_all_prompts_in_frame(
        state, frame_index, object_id
    )

    masks_binary = (masks > score_thresh)[:, 0].cpu().numpy()

    return FramePromptsDTO(
        frame_index=int(frame_idx),
        objects=_build_object_tracks(object_ids, masks_binary),
    )


def remove_object(
    predictor: SAM2VideoPredictor,
    state: InferenceState,
    object_id: int,
    score_thresh: float,
) -> list[TrackingFrameDTO]:
    """Remove an object from the tracking state and return updated frame masks.

    Args:
        predictor: Loaded SAM2 video predictor.
        state: Active session inference state.
        object_id: Object to remove.
        score_thresh: Binary threshold for mask logits.

    Returns:
        Per-frame tracking results after the object was removed.
    """

    new_obj_ids, updated_frames = predictor.remove_object(state, object_id)

    results: list[TrackingFrameDTO] = []
    for frame_index, video_res_masks in updated_frames:
        masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
        results.append(
            TrackingFrameDTO(
                frame_index=int(frame_index),
                objects=_build_object_tracks(new_obj_ids, masks_binary),
            )
        )

    return results


def propagate_in_video(
    predictor: SAM2VideoPredictor,
    state: InferenceState,
    start_frame_index: int,
    direction: str,
    max_frames: int | None,
    score_thresh: float,
) -> Generator[TrackingFrameDTO, None, None]:
    """Propagate prompts through the video and yield per-frame results.

    This is a synchronous generator. Call it inside a thread (not the event loop)
    to avoid blocking async coroutines.

    Args:
        predictor: Loaded SAM2 video predictor.
        state: Active session inference state (must have prompts added first).
        start_frame_index: Frame to begin propagation from.
        direction: ``"forward"``, ``"backward"``, or ``"both"``.
        max_frames: Maximum frames to process per direction; ``None`` means all.
        score_thresh: Binary threshold for mask logits.

    Yields:
        One ``TrackingFrameDTO`` per processed frame.
    """

    run_forward = direction in ("forward", "both")
    run_backward = direction in ("backward", "both")
    yielded_frame_indices: set[int] = set()
    yielded_count = 0

    def _yield_tracking_frames(reverse: bool) -> Generator[TrackingFrameDTO, None, None]:
        nonlocal yielded_count
        yielded_this_direction = 0
        logger.info(
            "SAMURAI propagate start: start_frame=%d direction=%s reverse=%s max_frames=%s",
            start_frame_index,
            direction,
            reverse,
            max_frames,
        )
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state=state,
            start_frame_idx=start_frame_index,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        ):
            if frame_idx in yielded_frame_indices:
                continue
            yielded_frame_indices.add(int(frame_idx))
            masks_binary = (masks > score_thresh)[:, 0].cpu().numpy()
            yielded_count += 1
            yielded_this_direction += 1
            if yielded_this_direction <= 5 or yielded_this_direction % 50 == 0:
                logger.info(
                    "SAMURAI propagate frame: reverse=%s frame_index=%d yielded_total=%d",
                    reverse,
                    int(frame_idx),
                    yielded_count,
                )
            yield TrackingFrameDTO(
                frame_index=int(frame_idx),
                objects=_build_object_tracks(obj_ids, masks_binary),
            )
        logger.info(
            "SAMURAI propagate end: reverse=%s yielded_this_direction=%d yielded_total=%d",
            reverse,
            yielded_this_direction,
            yielded_count,
        )

    if run_forward:
        yield from _yield_tracking_frames(reverse=False)

    if run_backward:
        yield from _yield_tracking_frames(reverse=True)


def _encode_mask(mask: np.ndarray) -> RLEMaskDTO:
    """Encode a binary mask array as a pycocotools RLE DTO.

    Args:
        mask: Boolean or uint8 array of shape ``(H, W)``.

    Returns:
        RLE-encoded mask DTO.
    """

    mask_f: np.ndarray = np.asfortranarray(mask.astype(np.uint8))
    rle: dict[str, object] = mask_util.encode(mask_f)
    counts_bytes: bytes = rle["counts"]  # type: ignore[assignment]
    return RLEMaskDTO(
        size=list(rle["size"]),  # type: ignore[arg-type]
        counts=counts_bytes.decode("utf-8"),
    )


def _mask_to_bbox(mask: np.ndarray) -> BoundingBoxDTO | None:
    """Compute a tight bounding box from a binary mask.

    Args:
        mask: Boolean array of shape ``(H, W)``.

    Returns:
        Bounding box DTO, or ``None`` if the mask is empty.
    """

    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None

    y_min, x_min = indices.min(axis=0).tolist()
    y_max, x_max = indices.max(axis=0).tolist()

    return BoundingBoxDTO(
        x=int(x_min),
        y=int(y_min),
        width=int(x_max - x_min),
        height=int(y_max - y_min),
    )


def _mask_to_polygon(mask: np.ndarray) -> list[list[int]] | None:
    """Approximate a binary mask contour as a polygon."""

    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() == 0:
        return None

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    epsilon = max(1.5, 0.002 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = [[int(point[0][0]), int(point[0][1])] for point in approx]
    return points if len(points) >= 3 else None


def _build_object_tracks(
    object_ids: list[int],
    masks: np.ndarray,
) -> list[ObjectTrackDTO]:
    """Build a list of ``ObjectTrackDTO`` from parallel object-id and mask arrays.

    Args:
        object_ids: Object identifiers aligned with ``masks`` first axis.
        masks: Boolean array of shape ``(N, H, W)``.

    Returns:
        List of per-object tracking results.
    """

    results: list[ObjectTrackDTO] = []
    for obj_id, mask in zip(object_ids, masks):
        has_object = bool(mask.any())
        results.append(
            ObjectTrackDTO(
                object_id=int(obj_id),
                mask=_encode_mask(mask) if has_object else None,
                bbox=_mask_to_bbox(mask) if has_object else None,
                polygon=_mask_to_polygon(mask) if has_object else None,
            )
        )
    return results


def segment_image_with_boxes(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    prompts: list[tuple[str | None, tuple[float, float, float, float], tuple[float, float] | None]],
    score_thresh: float,
) -> list[ImageSegmentationObjectDTO]:
    """Segment a static image with SAM2 using box prompts."""

    predictor.set_image(image)
    results: list[ImageSegmentationObjectDTO] = []

    for label, box, point in prompts:
        point_coords = None
        point_labels = None
        if point is not None:
            point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
        masks, ious, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
            return_logits=False,
            normalize_coords=False,
        )
        if len(masks) == 0:
            results.append(ImageSegmentationObjectDTO(label=label, bbox=None, mask=None, polygon=None))
            continue

        best_index = int(np.argmax(ious)) if len(ious) > 1 else 0
        mask = masks[best_index] > score_thresh
        has_object = bool(mask.any())
        results.append(
            ImageSegmentationObjectDTO(
                label=label,
                bbox=_mask_to_bbox(mask) if has_object else None,
                mask=_encode_mask(mask) if has_object else None,
                polygon=_mask_to_polygon(mask) if has_object else None,
            )
        )

    return results
