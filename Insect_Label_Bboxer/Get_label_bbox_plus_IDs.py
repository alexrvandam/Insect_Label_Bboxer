import cv2
import detectron2
import easydict
import googlemaps
import json
import mmcv
import numpy as np
import os
import pandas as pd
import pytesseract
import random
import scipy
import spacy
import sys
import torch
import torch, torchvision
from detectron2 import model_zoo
from detectron2.config import CfgNode
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from mmocr.apis import MMOCRInferencer
from mmocr.apis import TextRecInferencer

# Define the classes
classes = ["insect_label_val"]

# Add the metadata to the MetadataCatalog
MetadataCatalog.get("insect_label_val").set(thing_classes=classes)

#### from google.colab.patches import cv2_imshow

def ensure_bgr(image):
    if image.shape[2] == 3 and np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# Define path to custom Detectron2 config file and dataset metadata
# Load the custom Detectron2 model
cfg = get_cfg()
cfg_path = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/Updated_yml/insect_label_config.yaml"
cfg.merge_from_file("/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/Updated_yml/insect_label_config.yaml")
cfg.MODEL.WEIGHTS = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

# Get image paths
image_folder = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/test"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

###

# Define paths to your custom dataset's image and annotation files
train_image_dir = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/train"
train_annotation_path = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/train/via_project_18Apr2023_15h30m_json.json"
val_image_dir = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/val"
val_annotation_path = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/val/via_project_18Apr2023_15h30m_json.json"

# Register your custom dataset
register_coco_instances("my_dataset_train", {}, train_annotation_path, train_image_dir)
register_coco_instances("my_dataset_val", {}, val_annotation_path, val_image_dir)

  
###### need to fix
#metadata_path = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/metadata.json"

# Define path to input image and video folders, and output data folder
image_folder = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/test"
video_folder = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/test/video"
output_folder = "/Users/alexvandam/Mask_RCNN/Dataset/Insect_label/test/output"


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"/Users/alexvandam/opt/miniconda3/envs/detectron2/bin/pytesseract"

# Load SpaCy NLP model and Google Maps API client
nlp = spacy.load("en_core_web_lg")
gmaps = googlemaps.Client(key="AIzaSyBFuYjvJorWM6D8n144y8rdVhAGNmLzRe4")


###############################################################################
def save_bounding_box_images(image, boxes, output_folder, filename_prefix):
    box_data = []
    for idx, box in enumerate(boxes):
        # Convert box tensor to numpy array and round to integer
        box = box.cpu().numpy().astype(int)

        # Extract and save the bounding box image
        x1, y1, x2, y2 = box
        cropped_image = image[y1:y2, x1:x2]
        cropped_filename = f"{filename_prefix}_bbox_{idx}.jpg"
        cropped_path = os.path.join(output_folder, cropped_filename)
        cv2.imwrite(cropped_path, cropped_image)
        print(f"Saved bounding box image {idx} to {cropped_path}.")
        
        # Append the unique ID and box coordinates to the box_data list
        box_data.append((filename_prefix, idx, x1, y1, x2, y2))
        
    return box_data

def process_images(image_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a list to store box data for all images
    all_box_data = []

    for filename in os.listdir(image_folder):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            # Ensure the image is in BGR format
            image = ensure_bgr(image)
            
            # Perform instance segmentation using the custom Detectron2 model
            outputs = predictor(image)

            # Extract bounding boxes and assign unique IDs
            instances = outputs["instances"]
            boxes = instances.pred_boxes
            unique_ids = range(len(boxes))

            # Save the image with bounding boxes and unique IDs
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = v.get_image()[:, :, ::-1]

            # Save the result image
            result_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result.jpg")
            cv2.imwrite(result_path, result_image)

            # Save images of individual bounding boxes and get box data
            filename_prefix = os.path.splitext(filename)[0]
            box_data = save_bounding_box_images(image, boxes, output_folder, filename_prefix)
            all_box_data.extend(box_data)

            print(f"Processed {image_path}. Saved result to {result_path}.")

    # Save all box data to a .tsv file
    tsv_file = os.path.join(output_folder, "box_data.tsv")
    with open(tsv_file, "w") as f:
        f.write("image_id\tunique_id\tx1\ty1\tx2\ty2\n")
        for data in all_box_data:
            f.write(f"{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\t{data[4]}\t{data[5]}\n")

    print(f"Saved box data to {tsv_file}")

process_images(image_folder, output_folder)
