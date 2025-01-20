import warnings
from collections import OrderedDict

import torch
import numpy as np
import cv2

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores


class SAM2CameraPredictor(SAM2Base):
    """
    A variant of SAM2 predictor designed for real-time or streaming camera data.
    Unlike SAM2VideoPredictor, this class does not load a full video file at once.
    Instead, frames can be added on the fly and tracked incrementally.
    """

    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        clear_non_cond_mem_for_multi_obj=False,
        # Optional: whether to treat subsequent corrections also as conditioning frames.
        # If True, the frame that receives new prompts is treated as cond_frame_outputs.
        # If False, only the initially designated frames are "cond_frame_outputs."
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        """
        Args:
          fill_hole_area (int): If >0, fill holes up to this size in predicted masks.
          non_overlap_masks (bool): Whether to enforce a non-overlapping constraint
                                    among multiple objects in the final output.
          clear_non_cond_mem_around_input (bool): If True, after a prompt is added,
                                                 non-conditioning memory around that
                                                 frame is cleared (for single-object).
          clear_non_cond_mem_for_multi_obj (bool): If True, the same clearing behavior
                                                   applies also for multi-object tracking.
          add_all_frames_to_correct_as_cond (bool): If True, frames that receive new
                                                    prompts are considered "conditioning."
        """
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

        # A dictionary holding the entire "inference_state":
        #   - images
        #   - feature caches
        #   - prompts, mask inputs, output dicts, etc.
        self.inference_state = None

    @torch.inference_mode()
    def load_first_frame(self, frame, offload_state_to_cpu=False):
        """
        Initialize the state with the first frame from the camera/stream.

        Args:
            frame (np.ndarray or PIL.Image): The first frame (RGB or BGR).
            offload_state_to_cpu (bool): Whether to store large state tensors on CPU.
        """
        # Convert frame to a consistent shape & type
        if isinstance(frame, np.ndarray):
            # If BGR, convert to RGB to be consistent with SAM2 training
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]
            frame_tensor = self._preprocess_numpy(frame)
        else:
            # Assume it's a PIL.Image
            frame = frame.convert("RGB")
            W, H = frame.size
            frame = np.array(frame)
            frame_tensor = self._preprocess_numpy(frame)  # (3, H, W)

        device = self.device
        if offload_state_to_cpu:
            storage_device = torch.device("cpu")
        else:
            storage_device = device

        # Initialize the inference_state
        self.inference_state = {}
        self.inference_state["images"] = [frame_tensor]      # store the first frame
        self.inference_state["num_frames"] = 1
        self.inference_state["video_height"] = H
        self.inference_state["video_width"] = W
        self.inference_state["device"] = device
        self.inference_state["storage_device"] = storage_device
        self.inference_state["offload_state_to_cpu"] = offload_state_to_cpu

        # Per-object prompt storages
        self.inference_state["point_inputs_per_obj"] = {}
        self.inference_state["mask_inputs_per_obj"] = {}

        # A cache for recently visited frames' features, so we don't re-run the backbone
        self.inference_state["cached_features"] = {}
        # Values that don't change across frames
        self.inference_state["constants"] = {}

        # Object ID mappings
        self.inference_state["obj_id_to_idx"] = OrderedDict()
        self.inference_state["obj_idx_to_id"] = OrderedDict()
        self.inference_state["obj_ids"] = []

        # The main data structure storing model outputs
        self.inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        # Sliced (single-object) views of the outputs
        self.inference_state["output_dict_per_obj"] = {}
        # Temporary outputs when user is adding new prompts
        self.inference_state["temp_output_dict_per_obj"] = {}

        # Track which frames have consolidated (finalized) prompts
        self.inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        self.inference_state["tracking_has_started"] = False
        self.inference_state["frames_already_tracked"] = {}

        # Optionally warm up with frame 0
        self._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

    def _preprocess_numpy(self, frame_np):
        """
        Preprocess the incoming frame into a (3, self.image_size, self.image_size) float32 Tensor.
        The standard SAM2 approach is to resize to `self.image_size` and apply ImageNet normalization.
        Modify as needed if your deployment uses different transforms.

        Args:
            frame_np (np.ndarray): shape (H, W, 3), assumed to be RGB.

        Returns:
            (torch.Tensor): shape (3, image_size, image_size)
        """
        frame_resized = cv2.resize(frame_np, (self.image_size, self.image_size))
        # Convert [0,255] -> [0,1] if needed
        frame_resized = frame_resized.astype(np.float32) / 255.0

        # Basic mean-std normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_normalized = (frame_resized - mean) / std

        # HWC -> CHW
        frame_chw = frame_normalized.transpose(2, 0, 1)  # (3, H, W)
        return torch.from_numpy(frame_chw)

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """
        Add a set of prompt points (and optional box) to a specified frame.
        Similar to add_new_points_or_box(...) in SAM2VideoPredictor, but the
        `inference_state` is accessed as self.inference_state, and we do not check
        if frame_idx is beyond loaded frames, since we expect you track frames in order.

        Args:
            frame_idx (int): Index of the frame in self.inference_state["images"].
            obj_id (int): Identifier for the object (client-side) being prompted.
            points (Tensor or ndarray): shape (N,2) or (B,N,2), user-clicked points.
            labels (Tensor or ndarray): shape (N,) or (B,N,), 1=foreground, 0=background.
            clear_old_points (bool): If True, clear any old points for this object+frame.
            normalize_coords (bool): If True, treat points as (x, y) in original frame coords
                                     and normalize them to [0,1] domain before scaling.
            box (None or (4,) shape): A bounding box, if you prefer box-based prompt.
        Returns:
            (frame_idx, [obj_ids], mask_logits_resized_to_video)
        """
        inference_state = self.inference_state
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must both be provided or both be None.")
        if points is None and box is None:
            raise ValueError("At least one of `points` or `box` must be non-empty.")

        # Convert inputs to Tensors
        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int32)
        else:
            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
            labels = labels.unsqueeze(0)

        # If `box` is provided, we add it as the first two "points" with labels (2,3).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "Box prompt must be used with clear_old_points=True."
                )
            if inference_state["tracking_has_started"]:
                warnings.warn(
                    "Adding a box after tracking started may not always refine properly. "
                    "Consider using `reset_state` first if you want box-based initialization."
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32)
            box = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32).reshape(1, 2)
            points = torch.cat([box, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        # Normalize if needed
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        if normalize_coords and points.numel() > 0:
            points = points / torch.tensor([video_W, video_H]).to(points.device)

        # Scale to model image size
        points = points * self.image_size
        device = inference_state["device"]
        points = points.to(device)
        labels = labels.to(device)

        # Possibly combine with old prompts
        if not clear_old_points:
            old_points = point_inputs_per_frame.get(frame_idx, None)
        else:
            old_points = None
        point_inputs = concat_points(old_points, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs

        # Clear any old mask-based input on that frame
        mask_inputs_per_frame.pop(frame_idx, None)

        # Identify if this is an initial conditioning frame
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]

        # Decide if the user-supplied prompts on this frame classify it as a
        # "cond_frame_output" or "non_cond_frame_output"
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Possibly retrieve the previous (predicted) mask logits as input to the mask decoder
        prev_sam_mask_logits = None
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = torch.clamp(
                prev_out["pred_masks"].to(device), -32.0, 32.0
            )

        # Now run a single-frame forward pass on the mask decoder
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,  # we do memory encoding later
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Consolidate across all objects => get final mask in original resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """
        A simple alias for backward compatibility.
        """
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(self, frame_idx, obj_id, mask):
        """
        Add a full-resolution mask as input on a specific frame.
        This is similar to `add_new_mask(...)` in SAM2VideoPredictor.
        """
        inference_state = self.inference_state
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2, "Mask must be a 2D array or tensor."

        mask_h, mask_w = mask.shape
        device = inference_state["device"]
        mask_inputs_orig = mask.float().unsqueeze(0).unsqueeze(0).to(device)

        # Resize if needed
        if mask_h != self.image_size or mask_w != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        # Store & remove any old point-based input on that frame
        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)

        # Check if it's an init cond frame or correction
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]

        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Consolidate & resize to original
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state, frame_idx, is_cond=is_cond,
            run_mem_encoder=False, consolidate_at_video_res=True
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def propagate_in_video_preflight(self):
        """
        Called before actual tracking on new frames. Consolidates all new prompts
        into the main state (including memory encoding). Once this is called,
        `tracking_has_started=True`, so no new objects can be introduced.
        """
        inference_state = self.inference_state
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_inds = inference_state["consolidated_frame_inds"]

        for is_cond in [False, True]:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

            # 1) gather all frames that have new prompts
            prompt_frames = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                prompt_frames.update(obj_temp_output_dict[storage_key].keys())
            consolidated_inds[storage_key].update(prompt_frames)

            # 2) consolidate
            for fidx in prompt_frames:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx=fidx,
                    is_cond=is_cond,
                    run_mem_encoder=True
                )
                output_dict[storage_key][fidx] = consolidated_out
                self._add_output_per_object(
                    inference_state, fidx, consolidated_out, storage_key
                )

                # Optionally clear memory around this frame
                clear_mem = (
                    self.clear_non_cond_mem_around_input
                    and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1)
                )
                if clear_mem:
                    self._clear_non_cond_mem_around_input(inference_state, fidx)

            # 3) done with those temp frames
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # If a frame is upgraded to "cond_frame_outputs", remove it from "non_cond_frame_outputs"
        for fidx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(fidx, None)
        for obj_out_dict in inference_state["output_dict_per_obj"].values():
            for fidx in obj_out_dict["cond_frame_outputs"]:
                obj_out_dict["non_cond_frame_outputs"].pop(fidx, None)
        for fidx in consolidated_inds["cond_frame_outputs"]:
            consolidated_inds["non_cond_frame_outputs"].discard(fidx)

        # Double-check that all frames with prompts are in consolidated_frame_inds
        # and vice versa
        all_consolidated = (
            consolidated_inds["cond_frame_outputs"]
            | consolidated_inds["non_cond_frame_outputs"]
        )
        all_prompted = set()
        for pm in inference_state["point_inputs_per_obj"].values():
            all_prompted.update(pm.keys())
        for mk in inference_state["mask_inputs_per_obj"].values():
            all_prompted.update(mk.keys())
        assert all_prompted == all_consolidated, (
            "Mismatch in consolidated vs. prompted frames:\n"
            f"  all_prompted={all_prompted}\n  all_consolidated={all_consolidated}"
        )

    @torch.inference_mode()
    def track(self, new_frame):
        """
        Process one new frame from the camera feed, applying the previously consolidated
        memory. The result is a multi-object mask, returned in the same resolution as input.

        Args:
            new_frame (np.ndarray or PIL.Image): The new camera frame.

        Returns:
            (obj_ids, mask_logits): A list of object IDs and a batched 4D mask tensor
                                    in (B,1,H,W) shape (original resolution).
        """
        # If tracking hasn't started, it means we must "consolidate" prompts now
        inference_state = self.inference_state
        if not inference_state["tracking_has_started"]:
            self.propagate_in_video_preflight()

        # Preprocess & store
        if isinstance(new_frame, np.ndarray) and new_frame.shape[-1] == 3:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            H, W = new_frame.shape[:2]
            frame_tensor = self._preprocess_numpy(new_frame)
        else:
            # e.g. PIL image
            new_frame = new_frame.convert("RGB")
            W, H = new_frame.size
            new_frame_np = np.array(new_frame)
            frame_tensor = self._preprocess_numpy(new_frame_np)

        inference_state["images"].append(frame_tensor)
        frame_idx = inference_state["num_frames"]
        inference_state["num_frames"] += 1

        # Prepare memory for multi-object pass
        batch_size = self._get_obj_num(inference_state)
        output_dict = inference_state["output_dict"]

        # Actually run the track step
        current_out = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            batch_size=batch_size,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )[0]  # we only need the "compact_current_out"

        # Save to non_cond_frame_outputs
        output_dict["non_cond_frame_outputs"][frame_idx] = current_out
        self._add_output_per_object(inference_state, frame_idx, current_out,
                                    "non_cond_frame_outputs")

        # Mark the frame as tracked
        inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}

        # Resize predicted mask to original resolution
        pred_masks_gpu = current_out["pred_masks"]
        _, video_res_masks = self._get_orig_video_res_output(inference_state,
                                                             pred_masks_gpu)
        return inference_state["obj_ids"], video_res_masks

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # If tracking has started, we can't add new objects
        if inference_state["tracking_has_started"]:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"Existing object ids: {inference_state['obj_ids']}."
            )
        # else, create a new object slot
        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])

        # Initialize storages for that new object
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        return obj_idx

    def _obj_idx_to_id(self, inference_state, obj_idx):
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def reset_state(self):
        """
        Reset all tracking results and prompts throughout the entire session.
        """
        self._reset_tracking_results(self.inference_state)
        # Clear object IDs
        s = self.inference_state
        s["obj_id_to_idx"].clear()
        s["obj_idx_to_id"].clear()
        s["obj_ids"].clear()
        s["point_inputs_per_obj"].clear()
        s["mask_inputs_per_obj"].clear()
        s["output_dict_per_obj"].clear()
        s["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        # Wipe out all per-frame inputs and outputs
        for obj_dict in inference_state["point_inputs_per_obj"].values():
            obj_dict.clear()
        for obj_dict in inference_state["mask_inputs_per_obj"].values():
            obj_dict.clear()
        for obj_dict in inference_state["output_dict_per_obj"].values():
            obj_dict["cond_frame_outputs"].clear()
            obj_dict["non_cond_frame_outputs"].clear()
        for obj_dict in inference_state["temp_output_dict_per_obj"].values():
            obj_dict["cond_frame_outputs"].clear()
            obj_dict["non_cond_frame_outputs"].clear()

        outd = inference_state["output_dict"]
        outd["cond_frame_outputs"].clear()
        outd["non_cond_frame_outputs"].clear()

        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()

        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the predicted masks (B,1,Hm,Wm) up to the original camera frame resolution (H,W).
        Optionally apply a non-overlap constraint among objects.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)

        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """
        Single-step forward pass on frame `frame_idx`. 
        This includes obtaining image features, updating memory if needed, 
        and predicting the new masks.
        """
        # 1) Retrieve image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # 2) Run track_step (from SAM2Base)
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # 3) Optionally offload large outputs to CPU
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)

        # 4) Possibly fill holes in predicted mask
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)

        # 5) Expand `maskmem_pos_enc` from single object to batch if needed
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is small, keep on GPU
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        best_iou_score = current_out.get("best_iou_score", None)

        # 6) Return compact dict
        compact_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "best_iou_score": best_iou_score,
        }
        return compact_out, pred_masks_gpu

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate "temp_output_dict_per_obj[frame_idx]" for each object into a
        single multi-object output. Optionally re-run the memory encoder if needed.
        Optionally do the consolidation at the original image resolution
        (e.g., for editing at full res).
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        if consolidate_at_video_res:
            # For immediate user feedback at native resolution
            assert not run_mem_encoder, "Memory encoder must run at model resolution."
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            mask_key = "pred_masks"

        # Initialize "consolidated_out" with placeholders
        device = inference_state["device"]
        storage_device = inference_state["storage_device"]
        shape4 = (batch_size, 1, consolidated_H, consolidated_W)
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            mask_key: torch.full(
                shape4,
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=storage_device,
            ),
            "obj_ptr": torch.full(
                (batch_size, self.hidden_dim),
                NO_OBJ_SCORE,
                dtype=torch.float32,
                device=device,
            ),
            "object_score_logits": torch.full(
                (batch_size, 1),
                10.0,  # Logit of +10 => near 1.0 in sigmoids
                dtype=torch.float32,
                device=device,
            ),
        }

        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_out_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = obj_out_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_out_dict["non_cond_frame_outputs"].get(frame_idx, None)

            if out is None:
                # No data for this object => fill with empty pointers if memory is needed
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(inference_state, frame_idx)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue

            # Merge the predicted mask from "out" into `consolidated_out`
            obj_mask = out["pred_masks"]
            cons_mask = consolidated_out[mask_key]
            if obj_mask.shape[-2:] == cons_mask.shape[-2:]:
                cons_mask[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Need to resize
                resized = torch.nn.functional.interpolate(
                    obj_mask,
                    size=cons_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                cons_mask[obj_idx : obj_idx + 1] = resized

            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # If run_mem_encoder => re-encode memory after possibly applying non-overlap constraint
        if run_mem_encoder:
            # Upscale to model resolution
            hi_res = torch.nn.functional.interpolate(
                consolidated_out[mask_key].to(device),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                hi_res = self._apply_non_overlapping_constraints(hi_res)
            # Actually encode memory
            maskmem_feats, maskmem_pos = self._run_memory_encoder(
                inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=hi_res,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,
            )
            consolidated_out["maskmem_features"] = maskmem_feats
            consolidated_out["maskmem_pos_enc"] = maskmem_pos

        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """
        Build a dummy object pointer from an empty mask for frames that do not have
        that object. This ensures the model's memory is consistent in multi-object mode.
        """
        empty_mask = torch.zeros(
            (1, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"]
        )
        # Acquire image features for this frame, run a "track_step"
        _, _, feats, pos_embeds, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size=1
        )
        out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=feats,
            current_vision_pos_embeds=pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=empty_mask,
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return out["obj_ptr"]

    def _add_output_per_object(self, inference_state, frame_idx, current_out, storage_key):
        """
        Slice out each object's portion of the multi-object output into
        `output_dict_per_obj[obj_idx][storage_key][frame_idx]`.
        """
        maskmem_feats = current_out["maskmem_features"]
        maskmem_pos = current_out["maskmem_pos_enc"]
        preds = current_out["pred_masks"]
        ptr = current_out["obj_ptr"]
        obj_scores = current_out["object_score_logits"]

        for obj_idx, obj_dict in inference_state["output_dict_per_obj"].items():
            slc = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": preds[slc],
                "obj_ptr": ptr[slc],
                "object_score_logits": obj_scores[slc],
            }
            if maskmem_feats is not None:
                obj_out["maskmem_features"] = maskmem_feats[slc]
            if maskmem_pos is not None:
                obj_out["maskmem_pos_enc"] = [p[slc] for p in maskmem_pos]
            inference_state["output_dict_per_obj"][obj_idx][storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-cond memory around a newly prompted frame to avoid confusion
        from outdated memory. Useful if you do local corrections on single-object, etc.
        """
        r = self.memory_temporal_stride_for_eval
        start = frame_idx - r * self.num_maskmem
        end = frame_idx + r * self.num_maskmem
        non_cond = inference_state["output_dict"]["non_cond_frame_outputs"]
        for t in range(start, end + 1):
            non_cond.pop(t, None)
            for obj_out_dict in inference_state["output_dict_per_obj"].values():
                obj_out_dict["non_cond_frame_outputs"].pop(t, None)

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Retrieves (or computes) the backbone feature for a given frame_idx,
        expands it for `batch_size` objects.
        """
        cached = inference_state["cached_features"].get(frame_idx, (None, None))
        image, backbone_out = cached
        if backbone_out is None:
            # Not in cache => compute
            device = inference_state["device"]
            # Retrieve the N-th frame
            image = inference_state["images"][frame_idx].to(device).unsqueeze(0).float()
            backbone_out = self.forward_image(image)
            inference_state["cached_features"][frame_idx] = (image, backbone_out)

        # Expand for each object
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_bk = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_bk["backbone_fpn"]):
            expanded_bk["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_bk["vision_pos_enc"]):
            expanded_bk["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)

        # Prepare final
        feats = self._prepare_backbone_features(expanded_bk)
        return (expanded_image,) + feats

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size, high_res_masks,
        object_score_logits, is_mask_from_pts
    ):
        """
        Rerun memory encoder with updated masks (after applying constraints, etc.).
        Typically only called inside `_consolidate_temp_output_across_obj(...)`.
        """
        # Acquire frame features
        _, _, feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        # Encode
        mem_features, mem_pos_enc = self._encode_new_memory(
            current_vision_feats=feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # Possibly offload
        mem_features = mem_features.to(torch.bfloat16)
        mem_features = mem_features.to(inference_state["storage_device"])
        mem_pos_enc = self._get_maskmem_pos_enc(inference_state,
                                                {"maskmem_pos_enc": mem_pos_enc})
        return mem_features, mem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        Ensures that `maskmem_pos_enc` is shared across frames/objects consistently.
        We store a single copy in `inference_state["constants"]["maskmem_pos_enc"]`.
        """
        constants = inference_state["constants"]
        out_pos_enc = current_out["maskmem_pos_enc"]
        if out_pos_enc is None:
            return None

        if "maskmem_pos_enc" not in constants:
            # Save a single slice as canonical
            # typically out_pos_enc is a list of length #feature-stages
            # each stage has shape (B, C, H, W).
            # We'll store just an object=0 slice as the canonical "pattern".
            stored = [x[0:1].clone() for x in out_pos_enc]
            constants["maskmem_pos_enc"] = stored
        else:
            stored = constants["maskmem_pos_enc"]

        # Now expand the stored version to the current batch size
        bsize = out_pos_enc[0].size(0)
        expanded = [x.expand(bsize, -1, -1, -1) for x in stored]
        return expanded
