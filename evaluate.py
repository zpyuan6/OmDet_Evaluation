
import cv2
import torch
import yaml
from tqdm import tqdm
import os
import time
import json
from sklearn.metrics import average_precision_score
import numpy as np
from torch import Tensor

import requests
from PIL import Image
import torch

from transformers import AutoProcessor, OmDetTurboForObjectDetection

def assign_id(label:str, categories:list):
    """Assign a category ID to a category name."""
    return categories.index(label)

def load_dataset_yaml(dataset_yaml_path: str = "coco8.yaml") -> list:
    """Load categories names from a dataset YAML file in YOLO farmat."""
    with open(dataset_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def compute_iou(pred_box1, anno_box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = pred_box1
    x1g, y1g, x2g, y2g = anno_box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def load_image_and_labels(image_path: str, label_path: str):
    """Load an image and its corresponding labels."""
    image = Image.open(image_path).convert("RGB")

    pixel_width, pixel_height = image.size

    with open(label_path, 'r') as file:
        raw_labels = [line.strip().split() for line in file.readlines()]

        labels = []

        for raw_label in raw_labels:
            if len(raw_label) == 5:
                category, cx, cy, w, h = raw_label
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                labels.append({
                    'category_id': int(category),
                    'bbox': [(cx - 0.5*w) * pixel_width, (cy - 0.5*h) * pixel_height, (cx + 0.5*w) * pixel_width, (cy + 0.5*h) * pixel_height]
                    })


    return image, labels

def model_validation(
    data: dict,
    model: torch.nn.Module,
    save_path: str = None):

    processor = AutoProcessor.from_pretrained("omlab/OmDet-Turbo_tiny_SWIN_T")

    if save_path is None:
        save_path = "results"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    categories = []

    for i, category in data.get("names").items():
        categories.append(category)

    image_dir = data['val']
    label_dir = image_dir.replace('images', 'labels')

    image_files = [f for f in os.listdir(os.path.join(data['path'], image_dir)) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG') or f.endswith('.jpeg') or f.endswith('.JPEG')]

    all_results = []
    all_labels = []
    all_scores = []
    all_ious = []

    t = 0
    for image_file in tqdm(image_files):
        image_path = os.path.join(data['path'], image_dir, image_file)
        file_name = ".".join(image_file.split(".")[:-1])
        label_path = os.path.join(data['path'], label_dir, f"{file_name}.txt")

        image, annos = load_image_and_labels(image_path, label_path)
        
        inputs = processor(image, text=categories, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, Tensor) }
        
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()
        t += (end_time - start_time)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs,
            classes=categories,
            target_sizes=[image.size[::-1]],
            score_threshold=0.3,
            nms_threshold=0.3,)[0]

        boxes, scores, labels = results["boxes"].cpu().detach().numpy().tolist(), results["scores"].cpu().detach().numpy().tolist(), results["classes"]

        for box, score, label in zip(boxes, scores, labels):
            all_results.append({
                    'image_id': image_file,
                    'category_id': assign_id(label, categories),
                    'bbox': box,
                    'score': score
                    })
            all_scores.append(score)
            all_labels.append(label)
            ious = [compute_iou(box, l['bbox']) for l in annos if l['category_id'] == assign_id(label, categories)]

            all_ious.append(max(ious) if ious else 0)


        # Save results to file
        with open(os.path.join(save_path, 'results.json'), 'w') as f:
            json.dump(all_results, f)

    detection_speed = t / len(image_files)

    # Print detection speed
    print(f"Detection speed: {detection_speed} seconds per image")

    iou_thresholds = list(np.arange(0.5, 0.95, 0.05))

    aps = []
    for iou_thresh in iou_thresholds:
        y_true = [1 if iou >= iou_thresh else 0 for iou in all_ious]
        ap = average_precision_score(y_true, all_scores)
        print(f"AP at IoU {iou_thresh}: {ap}")
        aps.append(ap)

    print(f"mAP: {np.mean(np.array(aps))}")
    


if __name__ == "__main__":

    data_yaml_file = "ppn_zero_shot_0_1_val.yaml"
    # Load the dataset YAML file
    data = load_dataset_yaml(data_yaml_file)

    model = OmDetTurboForObjectDetection.from_pretrained("omlab/OmDet-Turbo_tiny_SWIN_T").cuda()

    model_validation(data, model)