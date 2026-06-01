"""Estimate the forward FLOPs of the mask-detection models.

    python -m src.flops --classifier models/efficientnet_v2_b3/original_data/eff_b3.h5
    python -m src.flops --detector models/yolov7-lite-s.pt --img-size 640
    python -m src.flops --classifier <h5> --detector <pt>     # both

The Keras classifier is measured with a layer-by-layer estimator (after Christos Kyrkou's
`net_flops`, 2019); the PyTorch detector with ptflops. Both are reported as GFLOPs, where
FLOPs = 2 x multiply-accumulates (MACs) — matching the values in the thesis
(~0.27 GFLOPs for EfficientNetV2-B3 at 64x64, ~2.96 GFLOPs for yolov7-lite-s at 640x640).
"""

import argparse
from pathlib import Path
from typing import List, Optional


def _layer_flops(layer) -> float:
    """Forward FLOPs of a single Keras layer (0 for layers with no multiply-adds)."""
    s = str(layer)
    if "Conv2D " in s and "DepthwiseConv2D" not in s and "SeparableConv2D" not in s:
        i = layer.input.get_shape()[1:4].as_list()
        filters = layer.filters if layer.filters is not None else i[2]
        return (
            2
            * (filters * layer.kernel_size[0] * layer.kernel_size[1] * i[2])
            * ((i[0] / layer.strides[0]) * (i[1] / layer.strides[1]))
        )
    if "Conv2D " in s and "DepthwiseConv2D" in s:
        i = layer.input.get_shape()[1:4].as_list()
        return (
            2
            * (layer.kernel_size[0] * layer.kernel_size[1] * i[2])
            * ((i[0] / layer.strides[0]) * (i[1] / layer.strides[1]))
        )
    if "Add" in s or "Maximum" in s or "Concatenate" in s:
        i = layer.input[0].get_shape()[1:4].as_list()
        return (len(layer.input) - 1) * i[0] * i[1] * i[2]
    if "Average" in s and "pool" not in s:
        i = layer.input[0].get_shape()[1:4].as_list()
        return len(layer.input) * i[0] * i[1] * i[2]
    if "pool" in s and "Global" not in s:
        i = layer.input.get_shape()[1:4].as_list()
        return (
            (i[0] / layer.strides[0])
            * (i[1] / layer.strides[1])
            * (layer.pool_size[0] * layer.pool_size[1] * i[2])
        )
    if "Global" in s:
        i = layer.input.get_shape()[1:4].as_list()
        return i[0] * i[1] * i[2]
    if "Dense" in s:
        i = layer.input.shape[1:4].as_list()[0]
        o = layer.output.shape[1:4].as_list()
        return 2 * (o[0] * i) if i is not None else 0.0
    return 0.0


def keras_flops(model) -> float:
    """Forward FLOPs of a Keras model, recursing into nested sub-models (e.g. the
    EfficientNetV2-B3 backbone inside the Sequential classifier)."""
    total = 0.0
    for layer in getattr(model, "layers", []):
        if getattr(layer, "layers", None):  # a nested model
            total += keras_flops(layer)
        else:
            total += _layer_flops(layer)
    return total


def torch_flops(model, img_size: int) -> float:
    """Forward FLOPs of a PyTorch model at a square input size, via ptflops (= 2 x MACs)."""
    import torch
    from ptflops import get_model_complexity_info

    model.eval()
    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model,
            (3, img_size, img_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    return 2.0 * macs


def _report(name: str, resolution: str, flops: float, params: int) -> None:
    print(f"{name}")
    print(f"  input     {resolution}")
    print(f"  GFLOPs    {flops / 1e9:.3f}")
    print(f"  params(M) {params / 1e6:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate model FLOPs (reported as GFLOPs).")
    parser.add_argument("--classifier", type=Path, help="Keras .h5 mask classifier.")
    parser.add_argument("--detector", type=Path, help="YOLOv7-Face .pt detector.")
    parser.add_argument(
        "--img-size", type=int, default=640, help="Detector input size (square; default 640)."
    )
    parser.add_argument("--device", default="cpu", help="Torch device for the detector.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.classifier is None and args.detector is None:
        parser.error("provide --classifier and/or --detector")

    if args.classifier is not None:
        import keras

        model = keras.models.load_model(str(args.classifier))
        height, width = model.input_shape[1:3]
        _report(
            f"classifier  {args.classifier.name}",
            f"{width}x{height}",
            keras_flops(model),
            model.count_params(),
        )

    if args.detector is not None:
        from yolo7_face.models.experimental import attempt_load

        model = attempt_load(str(args.detector), map_location=args.device)
        params = sum(p.numel() for p in model.parameters())
        _report(
            f"detector    {args.detector.name}",
            f"{args.img_size}x{args.img_size}",
            torch_flops(model, args.img_size),
            params,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
