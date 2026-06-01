"""Tests for the mAP computation (src.eval.evaluate.compute_map)."""

import copy

import pytest

pytest.importorskip("podm")
pytest.importorskip("shapely")

from podm.metrics import MetricPerClass  # noqa: E402

from src.eval import blank_annotations  # noqa: E402
from src.eval.annotations import _image_entry  # noqa: E402
from src.eval.evaluate import compute_map  # noqa: E402


def _ground_truth():
    """One image with a single with_mask box, using the project's COCO skeleton."""
    gt = copy.deepcopy(blank_annotations)
    gt["images"].append(_image_entry("a.png", 100, 100, 0))
    gt["annotations"].append({"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50]})
    return gt


def test_perfect_prediction_scores_one():
    predictions = [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9}
    ]
    results = compute_map(_ground_truth(), predictions, iou=0.5)
    assert MetricPerClass.mAP(results) == pytest.approx(1.0)


def test_wrong_class_hurts_map_but_detector_only_recovers():
    # right box, wrong mask class
    predictions = [
        {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 50, 50], "score": 0.9}
    ]

    full = MetricPerClass.mAP(compute_map(_ground_truth(), predictions, iou=0.5))
    assert full < 1.0

    detector_only = MetricPerClass.mAP(
        compute_map(_ground_truth(), predictions, iou=0.5, detector_only=True)
    )
    assert detector_only == pytest.approx(1.0)
