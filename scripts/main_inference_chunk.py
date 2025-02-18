import argparse
import gc
import os
import os.path as osp
import pdb

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor


def load_test_video_list(testing_list_path):
    with open(testing_list_path, 'r') as f:
        test_videos = [line.strip() for line in f.readlines()]
    return test_videos

def load_gt(gt_path):
    """
    Load the ground truth from the given path
    """
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    # bbox in first frame are prompts
    prompts = {}
    fid = 0
    for line in gt:
        x, y, w, h = map(int, line.split(','))
        prompts[fid] = ((x, y, x+w, y+h), 0)
        fid += 1

    return prompts

def get_ckpt_and_cfg(tracker_name, model_name):
    """
    Get the checkpoint and config file for the given tracker and model
    """
    assert tracker_name in ["sam2.1", "samurai"], "Invalid tracker name"
    assert model_name in ["tiny", "small", "base_plus", "large"], "Invalid model name"
    model_ckpt = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"

    if model_name == "base_plus":
        model_cfg = f"configs/{tracker_name}/sam2.1_hiera_b+.yaml"
    else:
        model_cfg = f"configs/{tracker_name}/sam2.1_hiera_{model_name[0]}.yaml"

    return model_ckpt, model_cfg

def split_list(video_list, num_chunks):
    """
    Split a list into num_chunks chunks
    """
    chunk_size = len(video_list) // num_chunks
    return [video_list[i:i+chunk_size] for i in range(0, len(video_list), chunk_size)]

def inference_chunk(dataset_path, tracker_name, model_name, chunk_videos, result_folder):
    exp_name = "test"

    model_ckpt, model_cfg = get_ckpt_and_cfg(tracker_name, model_name)

    for vid, video in enumerate(chunk_videos):

        cat_name = video.split('-')[0]
        cid_name = video.split('-')[1]
        video_basename = video.strip()
        frame_folder = osp.join(dataset_path, cat_name, video.strip(), "img")
        num_frames = len(os.listdir(osp.join(dataset_path, cat_name, video.strip(), "img")))
        height, width = cv2.imread(osp.join(frame_folder, "00000001.jpg")).shape[:2]

        logger.info(f"Running video [{vid+1}/{len(chunk_videos)}]: {video} with {num_frames} frames ({height}x{width})")

        predictor = build_sam2_video_predictor(model_cfg, model_ckpt, device="cuda:0")

        predictions = []

        # Start processing frames
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True)

            prompts = load_gt(osp.join(dataset_path, cat_name, video.strip(), "groundtruth.txt"))

            bbox, track_label = prompts[0]
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                mask_to_vis = {}
                bbox_to_vis = {}

                assert len(masks) == 1 and len(object_ids) == 1, "Only one object is supported right now"
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                predictions.append(bbox_to_vis)        
            
        os.makedirs(result_folder, exist_ok=True)
        with open(osp.join(result_folder, f'{video_basename}.txt'), 'w') as f:
            for pred in predictions:
                x, y, w, h = pred[0]
                f.write(f"{x},{y},{w},{h}\n")

        del predictor
        del state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/LaSOT-ext")
    parser.add_argument("--tracker_name", type=str, default="samurai")
    parser.add_argument("--model_name", type=str, default="large")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--root_result_folder", type=str, default="results")
    args = parser.parse_args()

    test_videos = load_test_video_list("data/LaSOT-ext/testing_set.txt")
    chunk_video_list = split_list(test_videos, args.num_chunks)

    chunk_videos = chunk_video_list[args.chunk_idx]

    logger.info(f"Chunk ID: {args.chunk_idx}, Number of videos: {len(chunk_videos)} (from {chunk_videos[0]} to {chunk_videos[-1]})")

    exp_result_folder = osp.join(args.root_result_folder, args.tracker_name, f"{args.exp_name}_{args.model_name}")

    inference_chunk(args.dataset_path, args.tracker_name, args.model_name, chunk_videos, exp_result_folder)

if __name__ == "__main__":
    main()
