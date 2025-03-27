FROM runpod/base:0.4.0-cuda11.8.0

# Copy the entire project
COPY . /app

# Install dependencies
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install --upgrade -r /app/requirements.txt --no-cache-dir

# Set the working directory
WORKDIR /app/

# Run the handler
CMD python3.9 -u handler.py