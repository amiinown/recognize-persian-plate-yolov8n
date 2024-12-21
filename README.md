# recognize-persian-plate-yolov8n
Image Processing project | recognize Persian Plate with YOLOv8n

 # Iranian Private Car License Plate Recognition

This project provides a robust solution for recognizing Iranian private car license plates using state-of-the-art technologies. The application supports license plate recognition from various input sources including video files, images, and live webcams or other input cameras.

---

## Features
- **Input Options:**
  1. Video files
  2. Images
  3. Live webcam or other input cameras

- **Technologies Used:**
  1. YOLOv8n for object detection
  2. DeepSort for object tracking
  3. Custom OCR trained on a Persian dataset for accurate text extraction
  4. Enhanced accuracy using combined datasets:
     - [Persian Car Plates Digits Detection](https://www.kaggle.com/datasets/nimapourmoradi/persian-car-plates-digits-detection-yolov8)
     - [Car Plate Detection Dataset](https://www.kaggle.com/datasets/nimapourmoradi/car-plate-detection-yolov8)
     - [Diverse LPD Training-Ready Dataset](https://www.kaggle.com/datasets/fxmikf/diverse-lpd-training-ready)

---

## Prerequisites

1. Install Python **3.12.1** from this address: [Download Python 3.12.1](https://www.python.org/downloads/release/python-3121/)
2. Create and activate a virtual environment:
   - **Windows:**
     ```bash
     python -m venv your_venv_name
     ./your_venv_name/Script/activate
     ```
   - **Linux:**
     ```bash
     python -m venv your_venv_name
     source your_venv_name/bin/activate
     ```
3. Install the required Python modules:
   ```bash
   pip install ultralytics==8.3.51
   pip install opencv-python==4.10.0.84
   pip install pillow==11.0.0
   pip install numpy==2.2.0
   pip install deep-sort-realtime==1.3.2
   ```

4. Install the appropriate version of **Torch** for your system:
   - Visit [PyTorch](https://pytorch.org/get-started/locally/) and select the suitable configuration.
   - Copy the suggested installation command and run it in the terminal.

   **Note:** If you do not have an Nvidia RTX graphics card, use the following command to process operations on the CPU (this will significantly increase processing time):
   ```bash
   pip install torch torchvision torchaudio
   ```

---

## Usage

1. Set the **input_path** and **output_path** variables in the Python script to the desired input and output directories.
2. Run the application:
   ```bash
   python your_script_name.py
   ```

---

## Technical Overview

### Object Detection
- **YOLOv8n**: Lightweight yet effective object detection model for recognizing license plates.

### Object Tracking
- **DeepSort**: Ensures that detected license plates are accurately tracked in video and live streams.

### OCR
- **Custom OCR Model**: Trained specifically on Persian license plates for precise text recognition.

---

## Notes
- This project only recognizes **Iranian private car license plates**.
- Ensure that the input and output paths are correctly configured before running the application.

---
