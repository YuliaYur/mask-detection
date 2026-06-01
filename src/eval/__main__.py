"""End-to-end mask-detection evaluation in one command.

Converts a dataset's labels to COCO ground truth, runs the detector + classifier to get
predictions, and computes mAP -- all under the hood:

    python -m src.eval --dataset school \
        --annotations-dir dataset/School/annotation --images-dir dataset/School/raw \
        --detector models/yolov7-lite-s.pt \
        --classifier models/efficientnet_v2_b3/combined_data/eff_b3.h5

Any stage can be skipped by passing a precomputed artifact, which also lets you re-score
existing predictions without re-running the models:

    python -m src.eval --annotations gt.json --predictions preds.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .annotations import convert
from .evaluate import compute_map, print_metrics
from .predict import run_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="Evaluate the mask detector end to end: convert ground truth, run the "
        "detector + classifier, and compute mAP. Pass --annotations / --predictions to reuse "
        "precomputed artifacts and skip the corresponding stage.",
    )

    ground = parser.add_argument_group("ground truth (use --annotations OR --dataset)")
    ground.add_argument(
        "--annotations", type=Path, help="Existing ground-truth COCO JSON (skip conversion)."
    )
    ground.add_argument(
        "--dataset",
        choices=["school", "kaggle"],
        help="Convert this dataset's labels to COCO ground truth.",
    )
    ground.add_argument(
        "--annotations-dir",
        type=Path,
        help="Directory of per-image label files (used with --dataset).",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Directory with the images (needed to predict, and for --dataset school).",
    )

    pred = parser.add_argument_group("predictions (use --predictions OR --classifier)")
    pred.add_argument(
        "--predictions", type=Path, help="Existing predictions COCO JSON (skip running the models)."
    )
    pred.add_argument("--classifier", type=Path, help="Keras .h5 mask classifier.")
    pred.add_argument(
        "--detector", type=Path, help="YOLOv7-Face .pt weights (omit with --use-gt-boxes)."
    )
    pred.add_argument(
        "--use-gt-boxes",
        action="store_true",
        help="Classify the ground-truth boxes instead of detecting faces.",
    )
    pred.add_argument(
        "--codeformer",
        type=Path,
        help="CodeFormer .pth weights; restore faces before classifying (three-stage).",
    )
    pred.add_argument("--codeformer-weight", type=float, default=0.5, help="CodeFormer fidelity w.")
    pred.add_argument("--img-size", type=int, default=640, help="Detector input size.")
    pred.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    pred.add_argument("--nms-iou", type=float, default=0.45, help="Detector NMS IoU threshold.")
    pred.add_argument(
        "--classifier-size", type=int, default=64, help="Face crop size fed to the classifier."
    )
    pred.add_argument("--device", default="cpu", help="Torch device for detector / CodeFormer.")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--map-iou", type=float, default=0.5, help="IoU threshold for mAP (default 0.5)."
    )
    evaluation.add_argument(
        "--detector-only",
        action="store_true",
        help="Score face detection only (ignore the predicted mask class).",
    )

    outputs = parser.add_argument_group("optional outputs")
    outputs.add_argument(
        "--save-annotations", type=Path, help="Write the converted ground truth to this path."
    )
    outputs.add_argument(
        "--save-predictions", type=Path, help="Write the predictions to this path."
    )
    return parser


def _load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Ground truth: reuse a COCO file or convert a dataset.
    if args.annotations is not None:
        ground_truth = _load_json(args.annotations)
    elif args.dataset is not None:
        if args.annotations_dir is None:
            parser.error("--annotations-dir is required with --dataset.")
        if args.dataset == "school" and args.images_dir is None:
            parser.error("--images-dir is required with --dataset school.")
        print(f"Converting {args.dataset} labels to COCO ground truth ...")
        ground_truth = convert(args.dataset, args.annotations_dir, args.images_dir)
        if args.save_annotations is not None:
            _dump_json(ground_truth, args.save_annotations)
    else:
        parser.error("provide ground truth via --annotations, or --dataset with --annotations-dir.")

    # 2. Predictions: reuse a COCO file or run the pipeline.
    if args.predictions is not None:
        predictions = _load_json(args.predictions)
    else:
        if args.classifier is None:
            parser.error("--classifier is required to run predictions (or pass --predictions).")
        if args.images_dir is None:
            parser.error("--images-dir is required to run predictions.")
        if not args.use_gt_boxes and args.detector is None:
            parser.error("--detector is required unless --use-gt-boxes (or pass --predictions).")
        print("Running predictions ...")
        predictions = run_predictions(
            ground_truth,
            args.images_dir,
            args.classifier,
            detector_path=args.detector,
            use_gt_boxes=args.use_gt_boxes,
            codeformer_path=args.codeformer,
            codeformer_weight=args.codeformer_weight,
            img_size=args.img_size,
            conf=args.conf,
            nms_iou=args.nms_iou,
            classifier_size=args.classifier_size,
            device=args.device,
        )
        if args.save_predictions is not None:
            _dump_json(predictions, args.save_predictions)

    # 3. mAP.
    print("Computing mAP ...")
    results = compute_map(
        ground_truth, predictions, iou=args.map_iou, detector_only=args.detector_only
    )
    print_metrics(results, args.map_iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
