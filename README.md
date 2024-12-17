# AI-Powered Object Detection App

This project is an AI-based object detection application that uses YOLOv5 for real-time object detection and integrates text-to-speech (TTS) functionality to announce detected objects and their distances. The application is designed to run on a local webcam, with a simple HTML interface.

## Features

- **Real-Time Object Detection**: Utilizes the YOLOv5 model for accurate object detection.
- **Distance Estimation**: Calculates the distance of detected objects using focal length.
- **Text-to-Speech**: Announces detected objects and their distances audibly.
- **Web Interface**: An `index.html` page for user interaction (expandable for web deployment).

## Technologies Used

- **Python**: For the main application logic (`app.py`).
- **YOLOv5**: Object detection framework.
- **PyTorch**: Backend for YOLOv5 model inference.
- **OpenCV**: For video capture and processing.
- **pyttsx3**: For text-to-speech functionality.
- **HTML**: For a simple web interface (`index.html`).

## Prerequisites

Ensure the following are installed on your system:

- Python 3.8 or higher
- PyTorch (with CUDA if using GPU)
- OpenCV
- pyttsx3
- A webcam for real-time detection

To install dependencies, run:
```bash
pip install torch torchvision torchaudio opencv-python pyttsx3

NOTE: Install yolov5 manually.
