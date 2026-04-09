"""Static image segmentation endpoints."""

import json
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.config import Settings
from api.dependencies import get_predictor, get_settings
from api.ml import inference
from api.ml.loader import autocast_context
from api.schemas.image import ImageSegmentationPrompt, ImageSegmentationResponse

router = APIRouter(prefix="/images", tags=["Images"])


@router.post(
    "/segment-by-boxes",
    response_model=ImageSegmentationResponse,
    summary="Segment a static image using box prompts",
)
async def segment_image_by_boxes(
    file: UploadFile = File(...),
    prompts_json: str = Form(...),
    predictor: SAM2VideoPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> ImageSegmentationResponse:
    """Segment a static image using the same SAMURAI video predictor flow as videos."""

    try:
        prompts_raw = json.loads(prompts_json)
        prompts = [ImageSegmentationPrompt.model_validate(item) for item in prompts_raw]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid prompts_json: {exc}") from exc

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    detections = []
    offload = predictor.device.type == "mps"

    with tempfile.TemporaryDirectory(prefix="samurai_image_") as temp_dir:
        frames_dir = Path(temp_dir)
        frame_path = frames_dir / "00000.jpg"
        pil_image.save(frame_path, format="JPEG", quality=95)

        ctx = autocast_context(predictor.device)
        with ctx:
            state = inference.init_state(
                predictor,
                str(frames_dir),
                offload_video_to_cpu=offload,
            )
            try:
                for object_id, prompt in enumerate(prompts, start=1):
                    frame_result = inference.add_box(
                        predictor=predictor,
                        state=state,
                        frame_index=0,
                        object_id=object_id,
                        box=(
                            float(prompt.box[0]),
                            float(prompt.box[1]),
                            float(prompt.box[2]),
                            float(prompt.box[3]),
                        ),
                        score_thresh=settings.score_thresh,
                    )
                    tracked_object = frame_result.objects[0] if frame_result.objects else None
                    detections.append(
                        {
                            "label": prompt.label,
                            "bbox": None if tracked_object is None else tracked_object.bbox,
                            "mask": None if tracked_object is None else tracked_object.mask,
                            "polygon": None if tracked_object is None else tracked_object.polygon,
                        }
                    )
            finally:
                inference.reset_state(predictor, state)

    return ImageSegmentationResponse.model_validate({"detections": detections})
