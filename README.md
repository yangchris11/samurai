<div align="center">
<img align="left" width="100" height="100" src="https://github.com/user-attachments/assets/1834fc25-42ef-4237-9feb-53a01c137e83" alt="">

# SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

[Cheng-Yen Yang](https://yangchris11.github.io), [Hsiang-Wei Huang](https://hsiangwei0903.github.io/), [Wenhao Chai](https://rese1f.github.io/), [Zhongyu Jiang](https://zhyjiang.github.io/#/), [Jenq-Neng Hwang](https://people.ece.uw.edu/hwang/)

[Information Processing Lab, University of Washington](https://ipl-uw.github.io/) 
</div>


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-lasot-ext)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot-ext?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-needforspeed)](https://paperswithcode.com/sota/visual-object-tracking-on-needforspeed?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-otb-2015)](https://paperswithcode.com/sota/visual-object-tracking-on-otb-2015?p=samurai-adapting-segment-anything-model-for-1)

[[Arxiv]](https://arxiv.org/abs/2411.11922) [[Project Page]](https://yangchris11.github.io/samurai/) [[Raw Results]](https://drive.google.com/drive/folders/1ssiDmsC7mw5AiItYQG4poiR1JgRq305y?usp=sharing) 

This repository is the official implementation of SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

https://github.com/user-attachments/assets/9d368ca7-2e9b-4fed-9da0-d2efbf620d88

All rights are reserved to the copyright owners (TM & © Universal (2019)). This clip is not intended for commercial use and is solely for academic demonstration in a research paper. Original source can be found [here](https://www.youtube.com/watch?v=cwUzUzpG8aM&t=4s).

## News
- [ ] **Incoming**: Support vot-challenge toolkit intergration.
- [ ] **Incoming**: Release demo script to support inference on video (with mask prompt).
- [x] **2026/03/12**: Release production-ready [REST API](#rest-api) with FastAPI, streaming NDJSON propagation, web demo, and Docker support.
- [x] **2025/02/18**: Release multi-GPU inference script.
- [x] **2025/01/27**: Release [inference script](https://github.com/yangchris11/samurai/blob/master/sam2/tools/README.md#samurai-vos-inference) on VOS task (SA-V)!
- [x] **2024/11/21**: Release [demo script](https://github.com/yangchris11/samurai?tab=readme-ov-file#demo-on-custom-video) to support inference on video (bounding box prompt).
- [x] **2024/11/20** Release [inference script](https://github.com/yangchris11/samurai?tab=readme-ov-file#main-inference) on VOT task (LaSOT, LaSOT-ext, GOT-10k, UAV123, TrackingNet, OTB100)!
- [x] **2024/11/19**: Release [paper](https://arxiv.org/abs/2411.11922), [code](https://github.com/yangchris11/samurai), and [raw results](https://drive.google.com/drive/folders/1ssiDmsC7mw5AiItYQG4poiR1JgRq305y?usp=sharing)!

## Getting Started

#### SAMURAI Installation 

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. You can install **the SAMURAI version** of SAM 2 on a GPU machine using:
```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

Install other requirements:
```
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

#### SAM 2.1 Checkpoint Download

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

#### Data Preparation

Please prepare the data in the following format:
```
data/LaSOT
├── airplane/
│   ├── airplane-1/
│   │   ├── full_occlusion.txt
│   │   ├── groundtruth.txt
│   │   ├── img
│   │   ├── nlp.txt
│   │   └── out_of_view.txt
│   ├── airplane-2/
│   ├── airplane-3/
│   ├── ...
├── basketball
├── bear
├── bicycle
...
├── training_set.txt
└── testing_set.txt
```

#### Main Inference
```
python scripts/main_inference.py 
```

## Demo on Custom Video

To run the demo with your custom video or frame directory, use the following examples:

**Note:** The `.txt` file contains a single line with the bounding box of the first frame in `x,y,w,h` format while the SAM 2 takes `x1,y1,x2,y2` format as bbox input.

### Input is Video File

```
python scripts/demo.py --video_path <your_video.mp4> --txt_path <path_to_first_frame_bbox.txt>
```

### Input is Frame Folder
```
# Only JPG images are supported
python scripts/demo.py --video_path <your_frame_directory> --txt_path <path_to_first_frame_bbox.txt>
```

## REST API

A production-ready FastAPI is available in the `api/` directory. It exposes the full SAMURAI tracking pipeline over HTTP with streaming NDJSON results, segmentation masks, and OpenAPI docs.

### Running locally (native)

Requires Python 3.10–3.13, ffmpeg, and SAM 2.1 checkpoints (see above).

```bash
# Install dependencies
pip install -e sam2/
pip install -r api/requirements.txt

# Copy and edit config
cp .env.example .env   # set SAMURAI_DEVICE=mps (Apple Silicon) or cuda

# Start the server
./run.sh               # auto-handles PYTHONPATH and sam2 shadowing
./run.sh --reload      # with hot-reload for development
```

Interactive docs: `http://localhost:8000/docs`
Web demo: open `index.html` in a browser.

### Running with Docker

```bash
# Copy and edit config
cp .env.example .env

# CPU-only (default)
docker compose up samurai

# With NVIDIA GPU
docker compose --profile gpu up samurai-gpu
```

The first build downloads SAM 2.1 checkpoints (~1 GB) automatically.
Uploaded videos are stored in a named volume (`samurai_data`) and deleted when a session is closed.

### Configuration

All settings are read from environment variables (or `.env`) prefixed with `SAMURAI_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SAMURAI_DEVICE` | `cuda` | PyTorch device: `cuda`, `mps`, or `cpu` |
| `SAMURAI_MODEL_SIZE` | `base_plus` | Model variant: `tiny`, `small`, `base_plus`, `large` |
| `SAMURAI_DTYPE` | `bfloat16` | Autocast dtype: `float16` or `bfloat16` |
| `SAMURAI_APP_ROOT` | `sam2` | Path to the `sam2/` directory (contains `checkpoints/`) |
| `SAMURAI_DATA_PATH` | `data/uploads` | Temporary storage for uploaded videos |
| `SAMURAI_MAX_UPLOAD_DURATION_SEC` | `10.0` | Maximum accepted video duration in seconds |
| `SAMURAI_SCORE_THRESH` | `0.0` | Minimum mask score threshold |

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status and GPU memory stats |
| `POST` | `/videos/upload` | Upload MP4 or MOV (optional trim via `start_sec` / `duration_sec`) |
| `GET` | `/videos/{id}/stream` | Stream the video file |
| `GET` | `/videos/{id}/poster` | First-frame JPEG poster |
| `POST` | `/sessions` | Create a tracking session — extracts frames and loads into GPU |
| `DELETE` | `/sessions/{id}` | Close session, release GPU memory, delete uploaded files |
| `POST` | `/sessions/{id}/frames/{n}/box` | Add bounding box prompt on frame `n` |
| `POST` | `/sessions/{id}/frames/{n}/points` | Add point prompts on frame `n` |
| `DELETE` | `/sessions/{id}/prompts` | Reset all prompts |
| `POST` | `/sessions/{id}/propagate` | Stream tracking results as NDJSON |
| `DELETE` | `/sessions/{id}/propagate` | Cancel an in-progress propagation |

### Typical workflow

```bash
# 1. Upload a video
VIDEO_ID=$(curl -s -X POST http://localhost:8000/videos/upload \
  -F "file=@clip.mp4" | jq -r '.video.id')

# 2. Create a session (extracts frames, loads GPU)
SESSION_ID=$(curl -s -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d "{\"video_id\": \"$VIDEO_ID\"}" | jq -r '.session_id')

# 3. Add a bounding box prompt on frame 0
curl -s -X POST "http://localhost:8000/sessions/$SESSION_ID/frames/0/box" \
  -H "Content-Type: application/json" \
  -d '{"object_id": 1, "box": [100, 80, 300, 260]}'

# 4. Stream tracking results (one JSON line per frame)
curl -N -X POST "http://localhost:8000/sessions/$SESSION_ID/propagate" \
  -H "Content-Type: application/json" \
  -d '{"direction": "both"}'

# 5. Close session — also deletes uploaded files
curl -X DELETE "http://localhost:8000/sessions/$SESSION_ID"
```

Each line of the propagation stream is a `TrackingFrameDTO`:

```json
{
  "frame_index": 42,
  "objects": [
    {
      "object_id": 1,
      "mask": {"size": [480, 640], "counts": "...RLE..."},
      "bbox": {"x": 110, "y": 85, "width": 195, "height": 170}
    }
  ]
}
```

`mask` is a COCO-format RLE segmentation mask. `bbox` is the tight bounding box derived from the mask.

## FAQs
**Question 1:** Does SAMURAI need training? [issue 34](https://github.com/yangchris11/samurai/issues/34)

**Answer 1:** Unlike real-life samurai, the proposed samurai do not require additional training. It is a zero-shot method, we directly use the weights from SAM 2.1 to conduct VOT experiments. The Kalman filter is used to estimate the current and future state (bounding box location and scale in our case) of a moving object based on measurements over time, it is a common approach that had been adopted in the field of tracking for a long time, which does not require any training. Please refer to code for more detail.

**Question 2:** Does SAMURAI support streaming input (e.g. webcam)?

**Answer 2:** Not yet. The existing code doesn't support live/streaming video as we inherit most of the codebase from the amazing SAM 2. Some discussion that you might be interested in: facebookresearch/sam2#90, facebookresearch/sam2#388 (comment).

**Question 3:** How to use SAMURAI in longer video?

**Answer 3:** See the discussion from sam2 https://github.com/facebookresearch/sam2/issues/264.

**Question 4:** How do you run the evaluation on the VOT benchmarks?

**Answer 4:** For LaSOT, LaSOT-ext, OTB, NFS please refer to the [issue 74](https://github.com/yangchris11/samurai/issues/74) for more details. For GOT-10k-test and TrackingNet, please refer to the official portal for submission.

## Acknowledgment

SAMURAI is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.

The VOT evaluation code is modifed from [VOT Toolkit](https://github.com/votchallenge/toolkit) by Luka Čehovin Zajc.

## Citation

Please consider citing our paper and the wonderful `SAM 2` if you found our work interesting and useful.
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
  author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
  year={2024},
  eprint={2411.11922},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.11922}, 
}
```
