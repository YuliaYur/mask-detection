"""Argument-parser tests for every CLI. These import only the (light) module top
levels, so they run without TensorFlow / PyTorch / the vendored model packages.
"""

import pytest


def test_train_parser_defaults_and_model_choice():
    from src.train import build_parser

    parser = build_parser()
    args = parser.parse_args(["--train-dir", "t", "--val-dir", "v", "--output", "m.h5"])
    assert args.model == "efficientnet"
    assert args.epochs == 6
    assert args.init_weights is None

    vgg = parser.parse_args(
        ["--train-dir", "t", "--val-dir", "v", "--output", "m.h5", "--model", "vgg"]
    )
    assert vgg.model == "vgg"

    with pytest.raises(SystemExit):  # --model only allows efficientnet / vgg
        parser.parse_args(
            ["--train-dir", "t", "--val-dir", "v", "--output", "m.h5", "--model", "resnet"]
        )


def test_eval_parser_dataset_choices():
    from src.eval.__main__ import build_parser

    parser = build_parser()
    reuse = parser.parse_args(["--annotations", "gt.json", "--predictions", "p.json"])
    assert str(reuse.annotations) == "gt.json"
    assert reuse.map_iou == 0.5
    assert reuse.detector_only is False

    kaggle = parser.parse_args(["--dataset", "kaggle", "--annotations-dir", "d"])
    assert kaggle.dataset == "kaggle"

    with pytest.raises(SystemExit):  # aizoo was removed
        parser.parse_args(["--dataset", "aizoo", "--annotations-dir", "d"])


def test_flops_parser_defaults():
    from src.flops import build_parser

    parser = build_parser()
    args = parser.parse_args(["--detector", "d.pt"])
    assert args.img_size == 640
    assert args.classifier is None
    assert args.device == "cpu"


def test_sample_faces_parser_defaults():
    from src.synthetic.sample_faces import build_parser

    parser = build_parser()
    args = parser.parse_args(["--src-glob", "x/*.jpg", "--out-dir", "out"])
    assert args.count == 18000
    assert args.seed == 0

    with pytest.raises(SystemExit):  # --src-glob is required
        parser.parse_args(["--out-dir", "out"])


def test_detect_parser_defaults():
    from src.detect import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["--source", "photo.jpg", "--detector", "d.pt", "--classifier", "c.h5"]
    )
    from pathlib import Path

    assert str(args.source) == "photo.jpg"
    assert args.out_dir == Path("runs/detect")
    assert args.conf == 0.25
    assert args.img_size == 640
    assert args.codeformer is None

    with pytest.raises(SystemExit):  # --source is required
        parser.parse_args(["--detector", "d.pt", "--classifier", "c.h5"])


def test_draw_masks_parser_defaults():
    from src.synthetic.draw_masks import build_parser

    parser = build_parser()
    args = parser.parse_args(["--in-dir", "i", "--out-dir", "o"])
    assert args.spiga_dataset == "merlrav"
    assert args.seed == 0
