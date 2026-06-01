"""Draw synthetic medical masks on faces using SPIGA facial landmarks.

For every unmasked face image, SPIGA predicts landmarks, the jaw + lower-nose
landmarks are filled with a random mask colour, and the result is written out as a
synthetic *with-mask* example.

    python -m src.synthetic.draw_masks \
        --in-dir dataset/VGG-Face2/data/for_vgg/no_mask \
        --out-dir dataset/VGG-Face2/data/for_vgg/mask
"""

import argparse
from glob import glob
from pathlib import Path
from typing import List, Optional

# Landmark indices (68-point model) that outline the area a mask covers:
# the jawline (1..15) plus the lower nose (29).
JAW_LANDMARKS = list(range(1, 16))
NOSE_LANDMARK = 29

# Mask colour palette and the probability of each: mostly white, some black,
# the rest a random colour.
MASK_COLORS = [(245, 245, 245), (0, 0, 0)]
COLOR_PROBABILITIES = [0.5, 0.2, 0.3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw synthetic masks on faces using SPIGA landmarks."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Directory of unmasked face images (*.jpg).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory the masked images are written to (same file names).",
    )
    parser.add_argument(
        "--spiga-dataset",
        default="merlrav",
        help="SPIGA landmark model config (e.g. merlrav, wflw).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for mask colours.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import cv2
    import numpy as np
    from tqdm import tqdm

    from spiga_project.spiga.inference.config import ModelConfig
    from spiga_project.spiga.inference.framework import SPIGAFramework

    np.random.seed(args.seed)
    processor = SPIGAFramework(ModelConfig(args.spiga_dataset))

    keep = np.zeros(68, dtype=bool)
    keep[JAW_LANDMARKS] = True
    keep[NOSE_LANDMARK] = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(glob(str(args.in_dir / "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {args.in_dir}")

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        bbox = [0, 0, image.shape[1], image.shape[0]]

        features = processor.inference(image, [bbox])
        landmarks = np.array(features["landmarks"][0])[keep].astype(int)

        random_color = [np.random.randint(0, 256) for _ in range(3)]
        palette = np.array(MASK_COLORS + [random_color])
        color = palette[np.random.choice(len(palette), p=COLOR_PROBABILITIES)]
        cv2.fillPoly(image, pts=[landmarks], color=(int(color[0]), int(color[1]), int(color[2])))

        cv2.imwrite(str(args.out_dir / Path(image_path).name), image)

    print(f"Wrote {len(image_paths)} masked images to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
