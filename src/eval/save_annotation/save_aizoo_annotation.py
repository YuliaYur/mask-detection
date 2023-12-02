from glob import glob
import xmltodict
import json
from tqdm import tqdm

import numpy as np

from src.eval import blank_annotations


gt_annotations = blank_annotations.copy()
gt_id = 0

for i, p in tqdm(enumerate(sorted(glob('../dataset/AIZOO/val/*.xml')))):
    try:
        with open(p, 'r') as f:
            xml_annotation = xmltodict.parse(f.read())['annotation']
            file_name = xml_annotation['filename']
            img_size = xml_annotation['size']
            ann = np.atleast_1d(xml_annotation['object'])
    except Exception as e:
        a = 1
        print('#', e, file_name)
        continue

    assert np.all([t['name'] == 'face' or t['name'] == 'face_mask' for t in ann])
    assert '.jpg' in file_name.lower() or '.png' in file_name.lower()

    # Ground Truth annotations
    for t in ann:
        bbox = t['bndbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])
        gt_annotations['annotations'].append({'id': gt_id,
                                              'image_id': i,
                                              'category_id': int(t['name'] == 'face_mask'),
                                              'bbox': [xmin, ymin, xmax, ymax]})
        gt_id += 1

    gt_annotations['images'].append({
        "width": img_size['width'],
        "height": img_size['height'],
        "flickr_url": "",
        "coco_url": "",
        "file_name": file_name,
        "date_captured": 0,
        "license": 0,
        "id": i
    })

with open('annotation_aizoo/annotation.json', 'w') as f:
    json.dump(gt_annotations, f, indent=4)
