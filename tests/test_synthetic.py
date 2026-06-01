"""Tests for the synthetic-data sampling CLI (src.synthetic.sample_faces)."""

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("tqdm")

from src.synthetic.sample_faces import main as sample_main  # noqa: E402


def test_sample_faces_count_and_naming(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = Path("src_imgs")
    src.mkdir()
    for i in range(5):
        (src / f"f{i}.jpg").write_bytes(b"x")

    rc = sample_main(
        ["--src-glob", "src_imgs/*.jpg", "--out-dir", "out", "--count", "3", "--seed", "0"]
    )
    assert rc == 0
    out = sorted(Path("out").glob("*.jpg"))
    assert {p.name for p in out} == {"0.jpg", "1.jpg", "2.jpg"}


def test_sample_faces_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = Path("s")
    src.mkdir()
    for i in range(10):
        (src / f"{i}.jpg").write_bytes(bytes([i]))  # distinct content per file

    sample_main(["--src-glob", "s/*.jpg", "--out-dir", "a", "--count", "4", "--seed", "42"])
    sample_main(["--src-glob", "s/*.jpg", "--out-dir", "b", "--count", "4", "--seed", "42"])

    a = [(Path("a") / f"{i}.jpg").read_bytes() for i in range(4)]
    b = [(Path("b") / f"{i}.jpg").read_bytes() for i in range(4)]
    assert a == b  # same seed -> same selection


def test_sample_faces_errors_on_empty_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        sample_main(["--src-glob", "nope/*.jpg", "--out-dir", "out"])
