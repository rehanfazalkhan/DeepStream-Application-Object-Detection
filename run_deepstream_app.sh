#!/bin/bash

# Build the Docker image
docker build -t deepstream-app:latest .

# Replace /path/to/your/video and /path/to/output with your actual video and output directories
VIDEO_PATH="MVI_6835.mp4"
OUTPUT_PATH="output.mp4"

# Run the Docker container
docker run -v "$VIDEO_PATH":/app/path/to/your/video -v "$OUTPUT_PATH":/app/path/to/output deepstream-app:latest
