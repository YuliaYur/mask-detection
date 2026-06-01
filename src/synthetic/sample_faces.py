"""Randomly sample face images into a flat directory.

Used to pick a subset of unmasked faces (e.g. from VGG-Face2 crops) that
``draw_masks`` then turns into synthetic *with-mask* training examples.

    python -m src.synthetic.sample_faces \
        --src-glob "dataset/VGG-Face2/data/crops/**/*.jpg" \
        --out-dir dataset/VGG-Face2/data/for_vgg/no_mask --count 18000
"""

import argparse
import shutil
from glob import glob
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly sample face images into a flat directory."
    )
    parser.add_argument(
        "--src-glob",
        required=True,
        help='Glob for the source images, e.g. "dataset/.../crops/**/*.jpg".',
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory the sampled images are copied into (named 0.jpg, 1.jpg, ...).",
    )
    parser.add_argument("--count", type=int, default=18000, help="How many images to sample.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducible sampling."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np
    from tqdm import tqdm

    paths = sorted(glob(args.src_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No images matched: {args.src_glob}")

    np.random.seed(args.seed)
    chosen = np.random.choice(paths, size=args.count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(tqdm(chosen)):
        shutil.copyfile(path, args.out_dir / f"{i}.jpg")

    print(f"Copied {len(chosen)} images to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
