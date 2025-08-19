# Bicycle Detection with YOLOv10

A computer vision project that uses YOLOv10 (You Only Look Once) to detect bicycles in images and videos. This project includes both pre-trained YOLOv10 models and custom-trained models specifically optimized for bicycle detection.

## 🚲 Project Overview

This project implements bicycle detection using state-of-the-art YOLOv10 object detection models. It can:
- Detect bicycles in single images
- Process multiple images in batch
- Analyze videos for bicycle detection
- Train custom models on bicycle datasets
- Compare performance between pre-trained and custom models

## ✨ Features

- **Image Detection**: Detect bicycles in single images with bounding boxes
- **Batch Processing**: Process multiple images from a folder
- **Video Analysis**: Detect bicycles in video files with real-time processing
- **Custom Training**: Train YOLOv10 models from scratch on bicycle datasets
- **Model Comparison**: Compare pre-trained YOLOv10 vs custom-trained models
- **GPU Support**: CUDA-enabled for faster inference and training

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Ultralytics
- CUDA-compatible GPU (recommended for training)

## 📦 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd PythonProject
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install torch ultralytics
```

## 🚀 Usage

### 1. Single Image Detection

```python
# Using pre-trained YOLOv10
predict_single_yolo("path/to/image.jpg")

# Using custom-trained model
predict_single_my_model("path/to/image.jpg")
```

### 2. Batch Image Processing

```python
# Process all images in a folder
predict(input_folder="datasets/bicycles/test", output_folder="./results")
```

### 3. Video Detection

```python
# Detect bicycles in a video file
detect_video("path/to/video.mp4")
```

### 4. Model Training

```python
# Train a custom model from scratch
train()
```

## 📁 Project Structure

```
PythonProject/
├── yolo.py                 # Main script with all functions
├── yolov10n.pt            # Pre-trained YOLOv10 weights
├── yolo11n.pt             # Alternative YOLOv11 weights
├── datasets/
│   ├── bicycles-2000/     # Custom bicycle dataset
│   │   ├── train/         # Training images
│   │   ├── valid/         # Validation images
│   │   ├── test/          # Test images
│   │   └── data.yaml      # Dataset configuration
│   └── bicycles/          # Additional bicycle dataset
├── results/                # Output folder for image processing
├── video_results/          # Output folder for video processing
├── ultralytics/            # Ultralytics framework files
└── .venv/                  # Virtual environment
```

## 🎯 Dataset

The project uses a custom bicycle dataset with:
- **Training set**: 2000+ bicycle images
- **Validation set**: For model evaluation
- **Test set**: For inference testing
- **Classes**: 1 (Bicycle)
- **Source**: Roboflow dataset

## 🔧 Configuration

### Training Parameters
- **Epochs**: 100
- **Image Size**: 320x320
- **Batch Size**: 8
- **Model**: YOLOv10n (nano version)

### Model Paths
- **Custom Model**: `ultralytics/runs/detect/my_custom_model_2000/weights/best.pt`
- **Pre-trained**: `yolov10n.pt`

## 📊 Performance

The project includes both pre-trained YOLOv10 models and custom-trained models specifically optimized for bicycle detection. Custom models are trained from scratch on bicycle datasets for improved accuracy in bicycle detection scenarios.

## 🖥️ Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 10GB+ free space for datasets and models

## 🚀 Quick Start

1. **Check GPU availability**:
```python
python yolo.py
```

2. **Run video detection** (default):
```python
# The script runs detect_video("bike.mp4") by default
python yolo.py
```

3. **Train custom model**:
```python
# Uncomment train() in main section
python yolo.py
```

## 📝 Examples

The project includes several example images for testing:
- `bicycle.jpg` - Single bicycle
- `bicycle_wbg.jpg` - Bicycle with background
- `bicycle_with_girl.jpg` - Bicycle with person
- `img.png`, `img_1.png`, `img_2.png` - Various test images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses the CC BY 4.0 license for the dataset and is open source.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv10 implementation
- [Roboflow](https://roboflow.com/) for the bicycle dataset
- PyTorch team for the deep learning framework

## 📞 Support

For questions or issues, please open an issue on the project repository.

---

**Note**: This project is designed for research and educational purposes. Ensure you have proper permissions for any images or videos you process.
