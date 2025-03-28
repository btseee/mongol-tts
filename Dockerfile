# Use a CUDA-enabled base image for GPU support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies required for Python and TTS-related libraries, including git
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        gcc g++ \
        make \
        git \
        python3 python3-dev python3-pip python3-venv python3-wheel \
        espeak-ng libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*

# Verify git installation
RUN git --version

# Optionally disable setuptools_scm commit lookup if not needed
ENV SETUPTOOLS_SCM_DISABLE_VERSION=1

# Upgrade pip and setuptools for compatibility with Python packages
RUN pip3 install -U pip setuptools

# Install llvmlite separately to ensure Numba dependencies are handled
RUN pip3 install llvmlite --ignore-installed

# Copy the entire repository into the container
COPY . /app/mongol-tts

# Set the working directory to where the training script resides
WORKDIR /app/mongol-tts

# Install dependencies from requirements.txt, ensuring CUDA support for PyTorch
RUN pip3 install -r /app/mongol-tts/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Define the command to run the training script, outputting to /workspace/models/mongol-tts
CMD ["python", "train.py", "/workspace/models/mongol-tts"]
