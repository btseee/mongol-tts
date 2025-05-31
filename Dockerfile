# 1. Base Image
FROM python:3.10-slim

# Install git, build tools, and libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements.txt first for Docker caching
COPY requirements.txt ./

# 4. Install Python dependencies
# Consider adding --no-cache-dir to pip install to reduce image size further
# RUN pip install --no-cache-dir -r requirements.txt
# However, some of the nvidia packages might be large, let's install normally first
RUN pip install -r requirements.txt

# 5. Copy the rest of the project files
COPY . .

# 6. No default CMD or ENTRYPOINT specified
# Users can run commands like:
# docker run <image_name> python synthesize.py --text "Сайн байна уу" --out_path output.wav
# docker run <image_name> python train.py ...
