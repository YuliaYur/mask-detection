"""Tests for the dataset -> COCO converters (src.eval.annotations).

Use relative, ASCII paths (via monkeypatch.chdir) because OpenCV on Windows cannot open
paths with non-ASCII characters, which pytest's absolute tmp_path may contain.
"""

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("xmltodict")

from src.eval.annotations import convert, convert_kaggle, convert_school  # noqa: E402

VOC_XML = """<annotation>
  <filename>img1.png</filename>
  <size><width>100</width><height>80</height></size>
  <object><name>with_mask</name>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>
  <object><name>without_mask</name>
    <bndbox><xmin>5</xmin><ymin>6</ymin><xmax>7</xmax><ymax>8</ymax></bndbox></object>
</annotation>"""


def test_convert_kaggle_voc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ann = Path("ann")
    ann.mkdir()
    (ann / "img1.xml").write_text(VOC_XML, encoding="utf-8")

    coco = convert_kaggle(ann)
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "img1.png"
    assert (coco["images"][0]["width"], coco["images"][0]["height"]) == ("100", "80")
    assert len(coco["annotations"]) == 2
    assert {a["category_id"] for a in coco["annotations"]} == {0, 1}  # without/with mask


def test_convert_kaggle_skips_incorrect_mask(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ann = Path("ann")
    ann.mkdir()
    bad = VOC_XML.replace("without_mask", "mask_weared_incorrect")
    (ann / "bad.xml").write_text(bad, encoding="utf-8")

    coco = convert_kaggle(ann)
    assert coco["images"] == []  # the whole image is skipped
    assert coco["annotations"] == []


def test_convert_school(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("raw").mkdir()
    Path("ann").mkdir()
    cv2.imwrite("raw/img1.png", np.full((80, 100, 3), 255, np.uint8))  # H=80, W=100
    label = {"img1.png": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6, "mask": 1}]}
    Path("ann/img1.json").write_text(json.dumps(label), encoding="utf-8")

    coco = convert_school(Path("ann"), Path("raw"))
    assert len(coco["images"]) == 1
    assert (coco["images"][0]["width"], coco["images"][0]["height"]) == (100, 80)
    annotation = coco["annotations"][0]
    assert annotation["category_id"] == 1
    assert annotation["bbox"] == [10, 16, 50, 48]  # normalised coords * (W, H)


def test_convert_is_isolated(tmp_path, monkeypatch):
    """Converting twice must not accumulate (the deepcopy fix)."""
    monkeypatch.chdir(tmp_path)
    ann = Path("ann")
    ann.mkdir()
    (ann / "img1.xml").write_text(VOC_XML, encoding="utf-8")

    first = convert_kaggle(ann)
    second = convert_kaggle(ann)
    assert len(first["annotations"]) == len(second["annotations"]) == 2


def test_convert_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        convert("aizoo", Path("ann"))  # aizoo is no longer supported
