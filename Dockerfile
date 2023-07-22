# Use a base image with Python and OpenCV installed
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and the DeepStream configuration file
COPY deepstream_app.py deepstream_app_config.txt ./

# Install required dependencies
RUN pip install torch==1.10.0 torchvision==0.11.1 opencv-python-headless==4.5.3.56 numpy==1.22.2 PyGObject==3.42.0

# Run the DeepStream application
CMD ["python", "deepstream_app.py"]
