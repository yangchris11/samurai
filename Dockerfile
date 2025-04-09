FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    wget \
    python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace
COPY . .

# Install Python packages
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    pip install hydra-core iopath decord

# Download SAM2 model checkpoints
RUN cd sam2/checkpoints && chmod +x download_ckpts.sh && ./download_ckpts.sh

# Default command
CMD ["/bin/bash"]
