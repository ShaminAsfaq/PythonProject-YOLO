import os
import shutil

import torch

from ultralytics import YOLO

data_folder = "datasets/bicycles-2000/data.yaml"
custom_model = "ultralytics/runs/detect/my_custom_model_2000/weights/best.pt"

def train():
    # Initialize a YOLOv10 model from scratch (no pretrained weights)
    model = YOLO("yolov10n.yaml")  # just defines architecture, not weights

    # Train on your dataset from scratch
    model.train(
        data=data_folder,  # your dataset YAML
        epochs=100,
        imgsz=320,
        batch=8,
        name="my_custom_model_2000",
        project="ultralytics/runs/detect",  # where to save results
        pretrained=False  # ensures no weights are loaded
    )

def predict_single_yolo(image_path):
    model = YOLO("yolov10n.pt")
    results = model(image_path)  # replace with your image path
    results[0].show()

def predict_single_my_model(image_path):
    model = YOLO(custom_model)
    results = model(image_path)  # replace with your image path
    results[0].show()

def predict(input_folder="datasets/bicycles/test", output_folder="./results"):
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

    print(f"Processed video saved in: {os.path.join(output_folder, 'bicycle_detection_video')}")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    # Optional CLI: python script.py <input_folder> <output_root> <results_name> <weights_path>

    # detect_video("short.mp4")
    detect_video("bike.mp4")

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
