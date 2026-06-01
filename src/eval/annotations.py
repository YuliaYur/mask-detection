"""Convert a mask dataset's labels into a COCO-format annotation dict.

Each supported dataset has a converter class sharing the ``DatasetConverter`` base:
  * SchoolConverter -- one JSON per image (normalised boxes + a `mask` flag)
  * VocConverter    -- PASCAL VOC XML (object names `with_mask` / `without_mask`; images
                       containing `mask_weared_incorrect` are skipped for kaggle)

``convert()`` is the entry point used by the ``python -m src.eval`` endpoint.
"""

import copy
import json
from glob import glob
from pathlib import Path

from . import blank_annotations, category_id


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


class DatasetConverter:
    """Base class: accumulates COCO images + annotations with running ids.

    Subclasses implement :meth:`convert`, calling :meth:`_add_box` / :meth:`_add_image` as they
    parse each label file.
    """

    def __init__(self):
        self.coco = copy.deepcopy(blank_annotations)
        self._gt_id = 0

    def convert(self) -> dict:
        raise NotImplementedError

    def _add_box(self, image_id: int, category: int, bbox: list) -> None:
        self.coco["annotations"].append(
            {"id": self._gt_id, "image_id": image_id, "category_id": category, "bbox": bbox}
        )
        self._gt_id += 1

    def _add_image(self, file_name: str, width, height, image_id: int) -> None:
        self.coco["images"].append(_image_entry(file_name, width, height, image_id))


class SchoolConverter(DatasetConverter):
    """School dataset: one JSON per image with normalised boxes and a `mask` flag."""

    def __init__(self, annotations_dir, images_dir):
        super().__init__()
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)

    def convert(self) -> dict:
        import cv2

        for image_id, path in enumerate(sorted(glob(str(self.annotations_dir / "*.json")))):
            with open(path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            file_name = list(ann.keys())[0]
            img = cv2.imread(str(self.images_dir / file_name))
            if img is None:
                raise FileNotFoundError(f"Unable to read image: {self.images_dir / file_name}")
            height, width = img.shape[:2]

            for face in list(ann.values())[0]:
                self._add_box(
                    image_id,
                    int(face["mask"]),
                    [
                        int(face["x_min"] * width),
                        int(face["y_min"] * height),
                        int(face["x_max"] * width),
                        int(face["y_max"] * height),
                    ],
                )
            self._add_image(file_name, width, height, image_id)
        return self.coco


class VocConverter(DatasetConverter):
    """PASCAL VOC XML: map object names to category ids; optionally skip whole images."""

    def __init__(self, annotations_dir, name_to_category: dict, skip_names=()):
        super().__init__()
        self.annotations_dir = Path(annotations_dir)
        self.name_to_category = name_to_category
        self.skip_names = skip_names

    def convert(self) -> dict:
        import numpy as np
        import xmltodict

        for image_id, path in enumerate(sorted(glob(str(self.annotations_dir / "*.xml")))):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    xml = xmltodict.parse(f.read())["annotation"]
                file_name = xml["filename"]
                size = xml["size"]
                objects = np.atleast_1d(xml["object"])
            except Exception as e:  # malformed XML -> log and skip
                print("#", e, path)
                continue

            if any(obj["name"] in self.skip_names for obj in objects):
                continue

            for obj in objects:
                box = obj["bndbox"]
                self._add_box(
                    image_id,
                    self.name_to_category[obj["name"]],
                    [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])],
                )
            self._add_image(file_name, size["width"], size["height"], image_id)
        return self.coco


def convert_school(annotations_dir: Path, images_dir: Path) -> dict:
    return SchoolConverter(annotations_dir, images_dir).convert()


def convert_kaggle(annotations_dir: Path) -> dict:
    return VocConverter(
        annotations_dir, category_id, skip_names=("mask_weared_incorrect",)
    ).convert()


def convert(dataset: str, annotations_dir: Path, images_dir: Path = None) -> dict:
    """Convert one of the supported datasets to a COCO annotation dict."""
    if dataset == "school":
        if images_dir is None:
            raise ValueError("images_dir is required for the school dataset.")
        return convert_school(annotations_dir, images_dir)
    if dataset == "kaggle":
        return convert_kaggle(annotations_dir)
    raise ValueError(f"Unknown dataset: {dataset}")
