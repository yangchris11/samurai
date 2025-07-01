#!/usr/bin/env python3
import os
import time
import argparse

import cv2
import torch
import numpy as np

# --- Replace this import with your repo’s actual model API ---
# e.g. from sam2.model import SamuraiModel
from sam2.samurai import SamuraiModel

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive GPU/CPU demo: draw a box, then segment next 5 frames."
    )
    p.add_argument(
        "--video",
        type=str,
        default=r"C:\Users\vansh\OneDrive\Documents\app proj 1\app project 11\samurai2\myvid1.mp4",
        help="Path to input video file",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to save mask images",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    # 2) Read first frame & let user draw a box
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")
    bbox = cv2.selectROI("Draw bounding box", first, False, False)
    cv2.destroyWindow("Draw bounding box")

    # 3) Prepare device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 4) Load model and move to device
    model = SamuraiModel()            # adjust constructor as needed
    model.to(device)
    model.eval()

    # 5) Ensure output dir exists
    os.makedirs(args.outdir, exist_ok=True)

    # 6) Segment next 5 frames
    for i in range(1, 6):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {i}; stopping early.")
            break

        start = time.time()

        # --- preprocess: convert BGR→RGB, HWC→CHW, normalize if needed ---
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            .float() / 255.0
        )

        # 7) Run inference with the initial box as prompt
        with torch.no_grad():
            # adjust call signature to your repo’s API:
            mask = model.predict(tensor, box=bbox)  
            # expect `mask` as a torch tensor [1,H,W] or numpy array

        # 8) Post-process & save mask
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0).cpu().numpy()
        mask_img = (mask * 255).astype(np.uint8)
        out_path = os.path.join(args.outdir, f"frame_{i}.png")
        cv2.imwrite(out_path, mask_img)

        elapsed = (time.time() - start) * 1000
        print(f"[INFO] Frame {i} → saved to {out_path}  ({elapsed:.1f} ms)")

    cap.release()
    print("[DONE] Segmentation complete.")

if __name__ == "__main__":
    main()
