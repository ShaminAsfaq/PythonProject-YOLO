import os
import shutil

import cv2
import torch

from ultralytics import YOLO

data_folder_150 = "datasets/bicycles/data.yaml"
data_folder_2000 = "datasets/bicycles-2000/data.yaml"
data_folder_5000 = "datasets/bicycles-5000/data.yaml"
data_folder_6000 = "datasets/bicycles-6000/data.yaml"

custom_model_150 = "ultralytics/runs/detect/my_custom_model/weights/best.pt"
custom_model_2000 = "ultralytics/runs/detect/my_custom_model_2000/weights/best.pt"
PRETRAINED_custom_model_2000 = "ultralytics/runs/detect/PRETRAINED_my_custom_model_2000/weights/best.pt"
custom_model_5000 = "ultralytics/runs/detect/my_custom_model_5000/weights/best.pt"
custom_model_6000 = "ultralytics/runs/detect/my_custom_model_6000/weights/best.pt"

model_name_150 = "my_custom_model"
model_name_2000 = "my_custom_model_2000"
model_name_5000 = "my_custom_model_5000"
model_name_6000 = "my_custom_model_6000"

# ONLY CHANGE THIS PART TO SWITCH BETWEEN DATASETS AND MODELS
data_folder = data_folder_6000
custom_model = custom_model_6000
model_name = model_name_6000
predict_folder = "datasets/bicycles-6000/test/images"

def train():
    # Initialize a YOLOv10 model from scratch (no pretrained weights)
    model = YOLO("yolov10n.yaml")  # just defines architecture, not weights

    # Train on your dataset from scratch
    model.train(
        data=data_folder,  # your dataset YAML
        epochs=150,
        imgsz=320,
        patience=20,
        batch=-1,
        name='PRETRAINED_' + model_name,  # name of the run
        project="ultralytics/runs/detect",  # where to save results
        pretrained=True  # ensures no weights are loaded
    )

def predict_single_yolo(image_path):
    model = YOLO("yolov10n.pt")
    results = model(image_path)  # replace with your image path
    results[0].show()

def predict_single_my_model(image_path):
    model = YOLO(custom_model)
    results = model(image_path)  # replace with your image path
    results[0].show()

def predict(input_folder=predict_folder, output_folder="./results"):
    # Clear the output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Load your trained model
    model = YOLO(custom_model)

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Run inference on all images in the folder
    results = model.predict(source=input_folder, save=True, project=output_folder, name="")

    print(f"Processed images are saved in: {os.path.join(output_folder)}")

def detect_video(input_video="short.mp4"):

    output_folder = "./video_results"
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load your trained model
    model = YOLO(custom_model)

    # Run inference on video
    # project=output_folder -> folder to save output
    # save=True -> saves annotated video
    results = model.predict(
        source=input_video,
        save=True,
        project=output_folder,
        name=""
    )

    print(f"Processed video saved in: {os.path.join(output_folder, f'bicycle_detection_video {model_name}')}")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    # Optional CLI: python script.py <input_folder> <output_root> <results_name> <weights_path>

    # detect_video("short.mp4")
    # detect_video("bicycle.mp4")

    # train()

    # predict()

    # predict_single_yolo("img.png")
    # predict_single_my_model("img.png")

    # predict_single_yolo("img_1.png")
    # predict_single_my_model("img_1.png")
    #
    # predict_single_yolo("img_2.png")
    # predict_single_my_model("img_2.png")
    #
    # predict_single_yolo("bicycle.jpg")
    # predict_single_my_model("bicycle.jpg")
    #
    # predict_single_yolo("bicycle_wbg.jpg")
    # predict_single_my_model("bicycle_wbg.jpg")
    #
    # predict_single_yolo("bicycle_with_girl.jpg")
    # predict_single_my_model("bicycle_with_girl.jpg")

