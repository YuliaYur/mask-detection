"""Convert a mask dataset's labels into a COCO-format annotation dict.

Two datasets are supported:
  * school  -- one JSON per image (normalised boxes + a `mask` flag)
  * kaggle  -- PASCAL VOC XML (object names `with_mask` / `without_mask`;
               images containing `mask_weared_incorrect` are skipped)

This is a library used by the `python -m src.eval` endpoint.
"""

import copy
import json
from glob import glob
from pathlib import Path

from . import blank_annotations, category_id


def _coco_skeleton() -> dict:
    return copy.deepcopy(blank_annotations)


def _image_entry(file_name: str, width, height, image_id: int) -> dict:
    return {
        "width": width,
        "height": height,
        "flickr_url": "",
        "coco_url": "",
        "file_name": file_name,
        "date_captured": 0,
        "license": 0,
        "id": image_id,
    }


def convert_school(annotations_dir: Path, images_dir: Path) -> dict:
    import cv2

    coco = _coco_skeleton()
    gt_id = 0
    for image_id, path in enumerate(sorted(glob(str(annotations_dir / "*.json")))):
        with open(path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        file_name = list(ann.keys())[0]
        img = cv2.imread(str(images_dir / file_name))
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {images_dir / file_name}")
        height, width = img.shape[:2]

        for face in list(ann.values())[0]:
            coco["annotations"].append(
                {
                    "id": gt_id,
                    "image_id": image_id,
                    "category_id": int(face["mask"]),
                    "bbox": [
                        int(face["x_min"] * width),
                        int(face["y_min"] * height),
                        int(face["x_max"] * width),
                        int(face["y_max"] * height),
                    ],
                }
            )
            gt_id += 1
        coco["images"].append(_image_entry(file_name, width, height, image_id))
    return coco


def _convert_voc(annotations_dir: Path, name_to_category: dict, skip_names=()) -> dict:
    import numpy as np
    import xmltodict

    coco = _coco_skeleton()
    gt_id = 0
    for image_id, path in enumerate(sorted(glob(str(annotations_dir / "*.xml")))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                xml = xmltodict.parse(f.read())["annotation"]
            file_name = xml["filename"]
            size = xml["size"]
            objects = np.atleast_1d(xml["object"])
        except Exception as e:  # malformed XML -> log and skip
            print("#", e, path)
            continue

        if any(obj["name"] in skip_names for obj in objects):
            continue

        for obj in objects:
            box = obj["bndbox"]
            coco["annotations"].append(
                {
                    "id": gt_id,
                    "image_id": image_id,
                    "category_id": name_to_category[obj["name"]],
                    "bbox": [
                        int(box["xmin"]),
                        int(box["ymin"]),
                        int(box["xmax"]),
                        int(box["ymax"]),
                    ],
                }
            )
            gt_id += 1
        coco["images"].append(_image_entry(file_name, size["width"], size["height"], image_id))
    return coco


def convert_kaggle(annotations_dir: Path) -> dict:
    return _convert_voc(annotations_dir, category_id, skip_names=("mask_weared_incorrect",))


def convert(dataset: str, annotations_dir: Path, images_dir: Path = None) -> dict:
    """Convert one of the supported datasets to a COCO annotation dict."""
    if dataset == "school":
        if images_dir is None:
            raise ValueError("images_dir is required for the school dataset.")
        return convert_school(annotations_dir, images_dir)
    if dataset == "kaggle":
        return convert_kaggle(annotations_dir)
    raise ValueError(f"Unknown dataset: {dataset}")
