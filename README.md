# DeepStream-Application-Object-Detection

#setup 
DeepStream Application - Local Setup and Execution
Introduction
This document provides a step-by-step guide on how to create and execute the DeepStream application for object detection in videos on your local system. The application utilizes YOLOv5 for inference and processes the video to detect various objects.

Prerequisites
Before proceeding, make sure you have the following prerequisites in place:

Python: Python should be installed on your system.
Video File: Prepare the video file that you want to process with the DeepStream application.
YOLOv5 Model: Make sure you have the YOLOv5 model file (e.g., yolov5s.pt) in the same directory as the application script.
Step-by-Step Guide
Step 1: Clone the Repository
Clone the repository containing the DeepStream application code to your local machine. The repository should include the following files:

deepstream_app.py: The DeepStream application code.
Step 2: Install Dependencies
Install the required Python packages by running the following command in your terminal:

bash
Copy code
pip install torch torchvision opencv-python-headless
Step 3: Modify Class Labels (Optional)
If you wish to use custom class labels for object detection, open the deepstream_app.py file using a text editor and make changes to the class_labels list.

Step 4: Prepare Input Video and Output Directory
Place the video you want to process in a directory on your local machine. Additionally, create a directory where you want the processed output video to be saved.

Step 5: Run the DeepStream Application
In the terminal, navigate to the project directory containing deepstream_app.py, and execute the following command:

bash
Copy code
python deepstream_app.py /path/to/your/video /path/to/output
Replace /path/to/your/video with the path to the input video file on your local machine, and /path/to/output with the path to the desired output directory.

Step 6: View the Processed Output
Once the DeepStream application completes processing the video, the output video with bounding boxes around detected objects will be saved to the specified output directory.
