# Research notebooks

The exploratory work behind the project: the classifier **training runs**, **component demos**,
and the scripts that produced the **thesis figures**. The reproducible CLIs in the main repo
(`src.train`, `src.synthetic`, `src.eval`) were distilled from these notebooks — use the CLIs to
re-run things; the notebooks are the record of how each piece was developed.

The two training notebooks were run on **Kaggle GPUs**, so they read `/kaggle/input/...` paths and
clone the vendored model repos at the top. The classifier dataset is the
[Face Mask ~12k Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
(`Train` / `Validation` / `Test`, each with `WithMask` / `WithoutMask` subfolders).

## Training — EfficientNetV2-B3 mask classifier

| Notebook | What it trains | CLI equivalent |
|---|---|---|
| `train-efficientnetv2-orig-mask-classification.ipynb` | **Classifier-Orig-Mask** — B3 on the Face Mask 12k data (64×64, batch 128, blur augmentation, frozen ImageNet backbone + `Dense(2, softmax)` head, 6 epochs) | `python -m src.train --model efficientnet …` |
| `train-efficientnetv2-syn-orig-mask-classification.ipynb` | **Classifier-Aug-Mask** — pre-train 6 epochs on synthetic masked faces, then fine-tune 3 epochs on the original data (transfer learning) | the same, chained with `--init-weights` |

These are the two classifiers behind the final results (Detector-Orig and Detector-Aug); see the
main README for how they score.

## Component demos

| Notebook | Shows |
|---|---|
| `yolo7_face_inference.ipynb` | YOLOv7-Face detecting faces on a sample image (the detection stage) |
| `spiga_inference.ipynb` | Drawing a synthetic medical mask from SPIGA facial landmarks — the project's synthetic-data method |
| `code_former_inference.ipynb` | CodeFormer restoring a low-quality face crop — the optional three-stage variant |
| `face_mask_detector.ipynb` | The full two-stage pipeline on one image: detect → classify → draw labelled boxes |

## Figures & analysis

| File | Produces |
|---|---|
| `plot_accuracy.ipynb` | Train/validation accuracy charts for the two EfficientNet models (the thesis accuracy figures) |

Model complexity (FLOPs) is computed by the CLI — `python -m src.flops` (a layer-by-layer
estimator for the Keras classifier, ptflops for the PyTorch detector).
