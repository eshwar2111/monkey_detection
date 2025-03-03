# Monkey Detection using YOLOv8

This project focuses on detecting monkeys in images and videos using **YOLOv8**, a state-of-the-art object detection model. It includes dataset preparation, model training, inference, and deployment.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Exporting the Model](#exporting-the-model)
- [Deployment](#deployment)

## Dataset Preparation
1. **Download Dataset**
   - Use an existing dataset (e.g., [OpenImages](https://storage.googleapis.com/openimages/web/download.html)) or create a custom dataset.
   - If needed, annotate images using [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg).

2. **Organize Data**
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   ├── labels/
   ├── valid/
   │   ├── images/
   │   ├── labels/
   ├── test/
   │   ├── images/
   │   ├── labels/
   ├── data.yaml
   ```
   - Ensure `data.yaml` contains the dataset details:
     ```yaml
     train: ./dataset/train
     val: ./dataset/valid
     test: ./dataset/test
     nc: 1  # Number of classes (monkeys)
     names: ['monkey']
     ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/monkey-detection-yolo.git
   cd monkey-detection-yolo
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python matplotlib
   ```

3. Verify YOLOv8 installation:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   print(model)
   ```

## Training the Model

1. Train YOLOv8 on the dataset:
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640
   ```

2. View training results:
   ```bash
   tensorboard --logdir=runs/detect/train
   ```

## Inference

1. Run inference on an image:
   ```bash
   yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
   ```

2. Run inference on a video:
   ```bash
   yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4
   ```

3. Real-time webcam detection:
   ```bash
   yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=0
   ```

## Exporting the Model

Export the trained model for deployment:
```bash
yolo task=detect mode=export model=runs/detect/train/weights/best.pt format=onnx
```

Supported formats: **ONNX, TFLite, TorchScript, CoreML**

## Deployment

For edge AI devices:
- **TFLite** for mobile applications.
- **ONNX** for inference on NVIDIA Jetson or OpenVINO.
- **TensorRT** for optimized performance on GPUs.

Example TFLite inference:
```python
import tensorflow.lite as tflite
interpreter = tflite.Interpreter(model_path="best.tflite")
```

## License
This project is released under the MIT License.

