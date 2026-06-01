"""Run the mask detector on image(s) and save them with the predictions drawn on top.

Two-stage by default (YOLOv7-Face detects faces, EfficientNetV2-B3 classifies each crop);
pass ``--codeformer`` for the three-stage variant. Every face is boxed **green** when a mask
is predicted and **red** when it is not, annotated with the confidence score.

    python -m src.detect --source photo.jpg \
        --detector models/yolov7-lite-s.pt \
        --classifier models/efficientnet_v2_b3/combined_data/eff_b3.h5

``--source`` may be a single image or a directory of images; annotated copies (same file
names) are written to ``--out-dir``.
"""

import argparse
from pathlib import Path
from typing import List, Optional

MASK_COLOR = (0, 255, 0)  # BGR -- green
NO_MASK_COLOR = (0, 0, 255)  # BGR -- red
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather_sources(source: Path) -> List[Path]:
    """A single image path, or every image directly inside a directory."""
    if not source.exists():
        raise FileNotFoundError(f"--source does not exist: {source}")
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def annotate(img0, boxes, results):
    """Draw a green/red box and score for each face on a copy of the BGR image."""
    import cv2

    out = img0.copy()
    thickness = max(2, round(0.002 * max(out.shape[:2])))
    for (x_min, y_min, x_max, y_max), result in zip(boxes, results):
        # class 0 is with_mask (see src.eval categories), so argmax == 0 means "mask".
        is_mask = int(result.argmax()) == 0
        color = MASK_COLOR if is_mask else NO_MASK_COLOR
        label = f"{'mask' if is_mask else 'no-mask'} {result.max():.2f}"
        top_left = (int(x_min), int(y_min))
        bottom_right = (int(x_max), int(y_max))
        cv2.rectangle(out, top_left, bottom_right, color, thickness)
        cv2.putText(
            out,
            label,
            (top_left[0], max(top_left[1] - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.detect",
        description="Run the mask detector on image(s) and save them with boxes drawn "
        "(green = mask, red = no mask).",
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Image file or directory of images."
    )
    parser.add_argument("--detector", type=Path, required=True, help="YOLOv7-Face .pt weights.")
    parser.add_argument("--classifier", type=Path, required=True, help="Keras .h5 mask classifier.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/detect"),
        help="Directory the annotated images are written to (default: runs/detect).",
    )
    parser.add_argument(
        "--codeformer",
        type=Path,
        help="CodeFormer .pth weights; restore faces before classifying (three-stage).",
    )
    parser.add_argument(
        "--codeformer-weight", type=float, default=0.5, help="CodeFormer fidelity w."
    )
    parser.add_argument("--img-size", type=int, default=640, help="Detector input size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Detector NMS IoU threshold.")
    parser.add_argument(
        "--classifier-size", type=int, default=64, help="Face crop size fed to the classifier."
    )
    parser.add_argument("--device", default="cpu", help="Torch device for detector / CodeFormer.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import cv2
    from tqdm import tqdm

    from .eval.predict import MaskDetector

    sources = _gather_sources(args.source)
    if not sources:
        raise FileNotFoundError(f"No images found at {args.source}")

    detector = MaskDetector(
        args.classifier,
        detector_path=args.detector,
        codeformer_path=args.codeformer,
        img_size=args.img_size,
        conf=args.conf,
        nms_iou=args.nms_iou,
        classifier_size=args.classifier_size,
        codeformer_weight=args.codeformer_weight,
        device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in tqdm(sources):
        img0 = cv2.imread(str(path))
        if img0 is None:
            print(f"# skipping unreadable image: {path}")
            continue

        boxes, _ = detector.detect(img0)
        results = detector.classify(img0, boxes)
        cv2.imwrite(str(args.out_dir / path.name), annotate(img0, boxes, results))
        written += 1

    print(f"Wrote {written} annotated image(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
