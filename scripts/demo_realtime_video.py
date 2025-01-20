import argparse
import time
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_camera_predictor

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_camera_predictor(model_cfg, args.model_path, device="cuda:0", mode='eval')

    cap = cv2.VideoCapture(args.video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.5)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.load_first_frame(frame)
        ann_frame_idx = 0
        ann_obj_id = 1
        bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        predictor.add_new_points_or_box(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=bbox
        )

    cap.release()
    height, width = frame.shape[:2]
    frame_rate = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    cap = cv2.VideoCapture(args.video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out_obj_ids, out_mask_logits = predictor.track(frame)

            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(out_obj_ids, out_mask_logits):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            for obj_id, mask in mask_to_vis.items():
                mask_img = np.zeros((height, width, 3), np.uint8)
                mask_img[mask] = color[(obj_id + 1) % len(color)]
                frame = cv2.addWeighted(frame, 1, mask_img, 0.2, 0)

            for obj_id, bbox in bbox_to_vis.items():
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)

    out.release()
    del predictor
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="data/videos/aquarium.mp4", help="Input video path or directory of frames.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_small.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    args = parser.parse_args()
    main(args)
