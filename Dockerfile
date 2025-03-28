FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision coqui-ai-TTS

# Copy your training script into the container
COPY train.py /app/train.py
COPY dataset /app/dataset

WORKDIR /app

# Default command to run the training script
CMD ["python3", "train.py", "--dataset", "dataset", "--output", "models/mongol-tts", "--epochs", "1000"]
