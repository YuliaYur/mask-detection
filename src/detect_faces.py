from pathlib import Path
from typing import List, Tuple, Optional
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

from yolo7_face.models.experimental import attempt_load
from yolo7_face.utils.datasets import LoadImages
from yolo7_face.utils.general import check_img_size, non_max_suppression, scale_coords, save_one_box
from yolo7_face.utils.plots import plot_one_box
from yolo7_face.utils.torch_utils import select_device


def detect(source: str = '../dataset/upd',
           weights: str = '../models/yolov7-w6-face.pt',
           img_size: int = 640,
           conf_threshold: float = 0.25,
           iou_threshold: float = 0.45) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path], List[Tuple[int]]]:

    device = select_device('cpu')

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)

    dataset = LoadImages(source, img_size=img_size, stride=stride)

    bounding_boxes = []
    confidences = []
    paths = []
    shapes = []
    for path, img, im0s, _ in dataset:
        print()

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, kpt_label=5)

        detection = pred[0]

        if len(detection):
            scale_coords(img.shape[2:], detection[:, :4], im0s.shape, kpt_label=False)
            detection[:, :4] /= np.array([im0s.shape[1], im0s.shape[0]] * 2)

        bounding_boxes.append(np.array(detection[:, :4]))
        confidences.append(np.array(detection[:, 4]))
        paths.append(Path(path))
        shapes.append(im0s.shape)

    return bounding_boxes, confidences, paths, shapes


def save_bounding_boxes(bounding_boxes: List[np.ndarray],
                        paths: List[Path],
                        save_dir: Path,
                        confidences: Optional[List[np.ndarray]] = None) -> None:
    save_dir.mkdir()

    if confidences is None:
        confidences = [None] * len(paths)

    for bb, conf, path in zip(bounding_boxes, confidences, paths):
        detections_json = []
        for xyxy in bb:
            detections_json.append({
                'x_min': float(xyxy[0]),
                'y_min': float(xyxy[1]),
                'x_max': float(xyxy[2]),
                'y_max': float(xyxy[3])
            })

        if conf is not None:
            for i, c in enumerate(conf):
                detections_json[i].update({'conf': float(c)})

        txt_path = str(save_dir / path.stem) + '.json'
        with open(txt_path, 'w') as f:
            json.dump({Path(path).name: detections_json}, f, indent=4)


def save_labeled_image(bounding_boxes: List[np.ndarray],
                       paths: List[Path],
                       save_dir: Path,
                       confidences: Optional[List[np.ndarray]] = None,
                       masks: Optional[List[np.ndarray]] = None) -> None:
    save_dir.mkdir()

    if confidences is None:
        confidences = [None] * len(paths)

    if masks is None:
        masks = [None] * len(paths)

    for bb, conf, mask, path in zip(bounding_boxes, confidences, masks, paths):
        im0s = cv2.imread(str(path))  # BGR

        for i, xyxy in enumerate(bb):

            xyxy = xyxy * np.array([im0s.shape[1], im0s.shape[0]] * 2)
            label = None if conf is None else f'face {conf[i]:.2f}'
            if mask is None:
                color = (255, 0, 0)
            elif mask[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            plot_one_box(xyxy, im0s, label=label, color=color, line_thickness=0, steps=3, orig_shape=im0s.shape[:2])

        cv2.imwrite(str(save_dir / path.name), im0s)


def save_crops(bounding_boxes: List[np.ndarray], paths: List[Path], save_dir: Path) -> None:
    save_dir.mkdir()

    for bb, path in tqdm(zip(bounding_boxes, paths)):
        im0s = cv2.imread(str(path))  # BGR

        for i, xyxy in enumerate(bb):
            xyxy = xyxy * np.array([im0s.shape[1], im0s.shape[0]] * 2)
            save_one_box(xyxy, im0s, file=save_dir / f'{path.stem}_crop{i}.png', BGR=True)
            # VGG-Face2 -- file=save_dir / f'{str(path)[-19:-12]}/{path.name}'
