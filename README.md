# Real-Time Face Mask Detection

[![CI](https://github.com/YuliaYur/mask-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YuliaYur/mask-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9-3776ab?logo=python&logoColor=white)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A real-time detector of **medical face masks**. A two-stage pipeline locates faces with
**YOLOv7-Face** and then classifies each face as *mask / no-mask* with an
**EfficientNetV2-B3** network. An optional three-stage variant inserts **CodeFormer** face
restoration before classification to improve accuracy on low-quality faces (at a speed
cost). To raise accuracy without labelling more data, the project also **synthesises masked
faces** by drawing masks onto **SPIGA** facial landmarks and trains on them.

Reaches **87.2% mAP** on a custom dataset of students and teachers ("School"). Based on the
author's MSc thesis, *Development of software for real-time detection of medical face masks*.

## Pipeline

```
Two-stage:    image ─► YOLOv7-Face ─► face crops ──────────────────► EfficientNetV2-B3 ─► mask / no-mask
Three-stage:  image ─► YOLOv7-Face ─► face crops ─► CodeFormer ─► restored crops ─► EfficientNetV2-B3 ─► mask / no-mask
```

The three building blocks are vendored as git subtrees and prepared independently:

- `yolo7_face/` — YOLOv7-Face detector (trained on WiderFace)
- `code_former/` — CodeFormer face restoration (optional middle stage)
- `spiga_project/` — SPIGA facial-landmark model (used to draw synthetic masks)

## Setup

### Option A — Docker (recommended)

The compose file builds a CPU image and a Jupyter Lab service that mounts the repo and
presets `PYTHONPATH`, so the vendored packages and `src/` import cleanly:

```bash
docker compose up mask-detection-lab
# then open http://localhost:8888
# point your dataset elsewhere with:  DATA_PATH=/path/to/dataset docker compose up mask-detection-lab
```

### Option B — Local Python

```bash
pip install -r docker/requirements.txt            # core
pip install -r docker/requirements_jupyter.txt    # add for the training notebooks
```

## Models & data

Trained weights and datasets are not committed; pass their paths to the CLIs. A typical
layout:

```
models/
  yolov7-lite-s.pt                                        # YOLOv7-Face detector
  efficientnet_v2_b3/{original_data,combined_data}/*.h5   # EfficientNetV2-B3 mask classifier (Keras)
code_former/weights/CodeFormer/codeformer.pth            # optional restoration stage
dataset/                                                  # see Datasets below
```

**Datasets**

- **Classifier training** — the [Face Mask ~12k Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
  (`Train` / `Validation` / `Test`, each with `WithMask` / `WithoutMask` subfolders). Training was run on
  Kaggle GPUs — see [`research/`](research/).
- **Synthetic masks** — generated from unmasked VGG-Face2 faces with `src.synthetic` (masks drawn on SPIGA landmarks).
- **Evaluation** — the custom *School* set (students and teachers) and the Kaggle
  [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) set ("Mask Dataset").

## Usage

Every pipeline is a CLI — run it from the **repo root** with `python -m src.…`. Datasets and
model weights are passed as arguments (nothing is hardcoded), and running from the repo root
puts the vendored packages on the import path automatically. Inside the Docker lab a terminal
is already set up this way.

The exploratory **notebooks under [`research/`](research/)** (training runs, component demos and
figure scripts — see [`research/README.md`](research/README.md)) document how each piece was built;
the CLIs below are the reproducible entry points distilled from them.

### Train the classifier

```bash
# EfficientNetV2-B3 on the original data (6 epochs)
python -m src.train --model efficientnet \
    --train-dir "dataset/Face Mask Dataset/Train" \
    --val-dir "dataset/Face Mask Dataset/Validation" \
    --epochs 6 --output models/eff_b3_original.h5
```

Synthetic → original transfer learning (the paper's combined model) chains two runs with
`--init-weights`:

```bash
python -m src.train --model efficientnet --train-dir data/synthetic/train --val-dir data/synthetic/val \
    --epochs 6 --output models/eff_b3_synthetic.h5
python -m src.train --model efficientnet --train-dir "dataset/Face Mask Dataset/Train" \
    --val-dir "dataset/Face Mask Dataset/Validation" --epochs 3 \
    --init-weights models/eff_b3_synthetic.h5 --output models/eff_b3_combined.h5
```

`--model vgg` trains the VGG19 baseline (128×128). Train/val directories hold two class
subfolders; the mask class must sort first alphabetically (e.g. `with_mask/`, `without_mask/`).
Each run also writes a `*.history.json` next to the model.

### Generate synthetic masked faces

Draw masks onto unmasked faces' SPIGA landmarks to create extra *with-mask* examples:

```bash
python -m src.synthetic.sample_faces \
    --src-glob "dataset/VGG-Face2/data/crops/**/*.jpg" \
    --out-dir data/synthetic/no_mask --count 18000
python -m src.synthetic.draw_masks --in-dir data/synthetic/no_mask --out-dir data/synthetic/mask
```

### Evaluate

One command — `python -m src.eval` — converts the dataset's labels to COCO ground truth,
runs the detector + classifier, and computes mAP:

```bash
python -m src.eval --dataset school \
    --annotations-dir dataset/School/annotation --images-dir dataset/School/raw \
    --detector models/yolov7-lite-s.pt \
    --classifier models/efficientnet_v2_b3/combined_data/efficientnetv2_b3_combined_data_epoch_6_3.h5
```

Useful flags: `--codeformer code_former/weights/CodeFormer/codeformer.pth` for the three-stage
variant; `--use-gt-boxes` (omit `--detector`) to score the classifier on its own;
`--detector-only` to score detection regardless of mask class; `--dataset kaggle` for the
VOC-XML "Mask Dataset".

Any stage can be skipped by passing a precomputed artifact (`--annotations` / `--predictions`).
Cache a run with `--save-annotations` / `--save-predictions`, then re-score it instantly — handy
for the slow three-stage (CodeFormer) runs:

```bash
# run once, caching the ground truth and the predictions
python -m src.eval --dataset school \
    --annotations-dir dataset/School/annotation --images-dir dataset/School/raw \
    --detector models/yolov7-lite-s.pt \
    --classifier models/efficientnet_v2_b3/combined_data/efficientnetv2_b3_combined_data_epoch_6_3.h5 \
    --save-annotations gt.json --save-predictions preds.json

# re-score instantly, without re-running the models
python -m src.eval --annotations gt.json --predictions preds.json
```

The thesis's headline result (Detector-Aug on the School set) is `mAP@0.5: 0.872`.

### Model FLOPs

```bash
python -m src.flops \
    --classifier models/efficientnet_v2_b3/original_data/efficientnetv2_b3_original_data_epoch_6.h5 \
    --detector models/yolov7-lite-s.pt
```

Reports GFLOPs (= 2 × multiply-accumulates): **~0.27** for EfficientNetV2-B3 @ 64×64 and
**~2.96** for yolov7-lite-s @ 640×640 — the thesis figures. The Keras classifier is measured with
a layer-by-layer estimator, the PyTorch detector with [ptflops](https://github.com/sovrasov/flops-counter.pytorch).
Pass `--img-size` to measure the detector at another resolution.

## Updating vendored components (git subtrees)

The three external models are tracked as subtrees. Pull upstream updates with:

```bash
git subtree pull --prefix=<subdirectory> <remote-name> <branch> --squash
```

| subdirectory | remote name | branch | upstream |
|---|---|---|---|
| `yolo7_face` | `yolo7_face` | `main` | `git@github.com:derronqi/yolov7-face.git` |
| `spiga_project` | `spiga_project` | `main` | `git@github.com:andresprados/SPIGA.git` |
| `code_former` | `code_former` | `master` | `git@github.com:sczhou/CodeFormer.git` |

## Development

Formatting, linting and tests cover **`src/` and `tests/`** — the vendored subtrees and the
research notebooks are excluded.

```bash
pip install -r requirements-dev.txt -r requirements-test.txt   # tools + light test deps
make format        # auto-format with black
make test          # run the unit tests
make check         # black --check + flake8 + pylint + pytest
```

On Windows use `make.bat` instead of `make` (e.g. `make.bat check`).

The unit tests need only **light dependencies** (`requirements-test.txt` — numpy, OpenCV,
shapely, …, no PyTorch/TensorFlow), so they run fast and the heavy model code is exercised
via lazy imports. They cover the CLI argument parsers, the dataset→COCO converters, the
synthetic-data sampling, and the mAP computation. GitHub Actions runs the black format check,
flake8, and the tests on every push and pull request (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## License

[MIT](LICENSE) © Yuliana Yurchenko
