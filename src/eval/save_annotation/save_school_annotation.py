from glob import glob
import json
from tqdm import tqdm

import cv2
from src.eval import blank_annotations


gt_annotations = blank_annotations.copy()
gt_id = 0

for i, p in tqdm(enumerate(sorted(glob('../dataset/School/annotation/*.json')))):
    with open(p, 'r') as f:
        ann = json.load(f)

    file_name = list(ann.keys())[0]
    img = cv2.imread('../dataset/School/raw/' + file_name)
    h, w = img.shape[:2]
    faces_ann = list(ann.values())[0]

    # Ground Truth annotations
    for face_ann in faces_ann:
        xmin = int(face_ann['x_min'] * w)
        ymin = int(face_ann['y_min'] * h)
        xmax = int(face_ann['x_max'] * w)
        ymax = int(face_ann['y_max'] * h)
        gt_annotations['annotations'].append({'id': gt_id,
                                              'image_id': i,
                                              'category_id': int(face_ann['mask']),
                                              'bbox': [xmin, ymin, xmax, ymax]})
        gt_id += 1

    gt_annotations['images'].append({
        "width": w,
        "height": h,
        "flickr_url": "",
        "coco_url": "",
        "file_name": file_name,
        "date_captured": 0,
        "license": 0,
        "id": i
    })

with open('annotation_school/annotation.json', 'w') as f:
    json.dump(gt_annotations, f, indent=4)
