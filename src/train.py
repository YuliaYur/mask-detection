"""Train the mask classifier on face crops split into two class folders.

Two backbones are supported, matching the thesis:
  * efficientnet -- EfficientNetV2-B3, 64x64 input (the main model)
  * vgg          -- VGG19, 128x128 input (baseline)

The frozen ImageNet backbone is reused and only a small head is trained. Pass
--init-weights to continue training an existing model on another dataset, which is how
the paper's "synthetic -> original" transfer learning was done (train on synthetic,
then fine-tune on original).

    # train on synthetic data, then fine-tune on the original data
    python -m src.train --model efficientnet --train-dir data/synthetic/train \
        --val-dir data/synthetic/val --epochs 6 --output models/eff_b3_synthetic.h5
    python -m src.train --model efficientnet --train-dir data/original/train \
        --val-dir data/original/val --epochs 3 --init-weights models/eff_b3_synthetic.h5 \
        --output models/eff_b3_combined.h5
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

# Per-backbone input size and default batch size (from the thesis notebooks).
IMAGE_SIZE = {"efficientnet": 64, "vgg": 128}
DEFAULT_BATCH_SIZE = {"efficientnet": 128, "vgg": 32}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the mask classifier (EfficientNetV2-B3 or VGG19)."
    )
    parser.add_argument("--model", choices=["efficientnet", "vgg"], default="efficientnet")
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Training directory with two class subfolders (mask / no-mask).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Validation directory with the same two class subfolders.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to save the .h5 model.")
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Existing .h5 to continue training (synthetic -> original transfer learning).",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to 128 (efficientnet) or 32 (vgg).",
    )
    return parser


def _random_blur(image):
    """Augmentation: blur with a random odd kernel in {1, 3, 5, 7}."""
    import cv2
    import numpy as np

    kernel = np.random.randint(0, 4) * 2 + 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def build_model(model_name: str):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    size = IMAGE_SIZE[model_name]
    if model_name == "efficientnet":
        from tensorflow.keras.applications import EfficientNetV2B3

        base = EfficientNetV2B3(
            weights="imagenet",
            include_top=False,
            input_shape=(size, size, 3),
            include_preprocessing=True,
        )
        head = Dense(2, activation="softmax")
    else:
        from tensorflow.keras.applications.vgg19 import VGG19

        base = VGG19(weights="imagenet", include_top=False, input_shape=(size, size, 3))
        head = Dense(2, activation="sigmoid")

    for layer in base.layers:
        layer.trainable = False
    return Sequential([base, Flatten(), head])


def data_generators(model_name: str, train_dir: Path, val_dir: Path, batch_size: int):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if model_name == "efficientnet":
        # include_preprocessing handles normalisation, so no rescale here.
        train_aug = ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.6, 1.0],
            preprocessing_function=_random_blur,
        )
        val_aug = ImageDataGenerator()
    else:
        train_aug = ImageDataGenerator(
            rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2
        )
        val_aug = ImageDataGenerator(rescale=1.0 / 255)

    size = IMAGE_SIZE[model_name]
    options = dict(target_size=(size, size), class_mode="categorical", batch_size=batch_size)
    train_generator = train_aug.flow_from_directory(str(train_dir), **options)
    val_generator = val_aug.flow_from_directory(str(val_dir), **options)
    return train_generator, val_generator


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]

    from tensorflow import keras

    train_generator, val_generator = data_generators(
        args.model, args.train_dir, args.val_dir, batch_size
    )

    if args.init_weights is not None:
        model = keras.models.load_model(str(args.init_weights))
    else:
        model = build_model(args.model)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    history = model.fit(train_generator, validation_data=val_generator, epochs=args.epochs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    history_path = args.output.with_suffix(".history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=4)
    print(f"Saved model to {args.output}; history to {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
