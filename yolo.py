import os
import shutil

import torch

from ultralytics import YOLO

def train():
    # Initialize a YOLOv10 model from scratch (no pretrained weights)
    model = YOLO("yolov10n.yaml")  # just defines architecture, not weights

    # Train on your dataset from scratch
    model.train(
        data="datasets/bicycles/data.yaml",  # your dataset YAML
        epochs=100,
        imgsz=640,
        batch=8,
        name="my_custom_model",
        pretrained=False  # ensures no weights are loaded
    )

def predict_single():
    model = YOLO("ultralytics/runs/detect/my_custom_model6/weights/best.pt")
    model = YOLO("yolov10n.pt")
    results = model("bicycle.jpg")  # replace with your image path
    results[0].show()

def predict(input_folder="datasets/bicycles/test", output_folder="./results"):
    # Clear the output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Load your trained model
    model = YOLO("ultralytics/runs/detect/my_custom_model6/weights/best.pt")

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Run inference on all images in the folder
    results = model.predict(source=input_folder, save=True, project=output_folder, name="")

    print(f"Processed images are saved in: {os.path.join(output_folder)}")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    # Optional CLI: python script.py <input_folder> <output_root> <results_name> <weights_path>
    # train()
    # predict()
    predict_single()
