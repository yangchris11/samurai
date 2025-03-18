# Base image with Python >= 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget unzip ffmpeg libjpeg-dev libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install PyTorch and TorchVision
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# Create workspace directory
WORKDIR /workspace

# Clone the SAM 2 repository
RUN git clone https://github.com/MlLearnerAkash/samurai.git && \
    cd samurai/sam2 && pip install -e . && pip install -e ".[notebooks]"

# Install additional Python dependencies
RUN pip install --no-cache-dir matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy

# Download checkpoints
WORKDIR /workspace/samurai/sam2/checkpoints
RUN chmod +x download_ckpts.sh && \
    ./download_ckpts.sh \

# Copy data preparation script or mount your dataset
WORKDIR /workspace/samurai/data
# Make sure to structure your data according to the format described in the instructions

# Set the default working directory
WORKDIR /workspace/samurai

# Default command (you can override this when running the container)
# CMD ["python", "scripts/main_inference.py"