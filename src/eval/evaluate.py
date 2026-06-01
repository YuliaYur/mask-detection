"""Compute Pascal-VOC mAP of COCO predictions against ground truth, in memory.

This is a library used by the `python -m src.eval` endpoint.
"""

import io
import json
from typing import List


def compute_map(
    ground_truth: dict, predictions: List[dict], iou: float = 0.5, detector_only: bool = False
):
    """Pascal-VOC metrics per class for the given ground truth and predictions.

    With ``detector_only`` the mask classes are collapsed so only face *detection* is
    scored (ignoring whether the predicted mask class is right).
    """
    from podm import coco_decoder
    from podm.metrics import get_bounding_boxes, get_pascal_voc_metrics

    gold = coco_decoder.load_true_object_detection_dataset(io.StringIO(json.dumps(ground_truth)))
    pred = coco_decoder.load_pred_object_detection_dataset(
        io.StringIO(json.dumps(predictions)), gold
    )

    gt_boxes = get_bounding_boxes(gold)
    pred_boxes = get_bounding_boxes(pred)
    if detector_only:
        for box in gt_boxes + pred_boxes:
            box.category = "face"

    return get_pascal_voc_metrics(gt_boxes, pred_boxes, iou)


def print_metrics(results, iou: float = 0.5) -> float:
    """Print per-class metrics and the mAP; return the mAP."""
    from podm.metrics import MetricPerClass

    for cls, metric in results.items():
        print("Class", cls)
        print("  ap             ", metric.ap)
        print("  tp             ", metric.tp)
        print("  fp             ", metric.fp)
        print("  num_groundtruth", metric.num_groundtruth)
        print("  num_detection  ", metric.num_detection)
        print()

    mean_ap = MetricPerClass.mAP(results)
    print(f"mAP@{iou}: {mean_ap}")
    return mean_ap
