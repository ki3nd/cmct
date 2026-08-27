"""Synthetic dataset fixtures: a handful of tiny PNGs on disk, so the data tests
need no real Office-Home."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

CLASSES = ["alarm_clock", "backpack", "mouse"]
DOMAINS = {"art": 4, "clipart": 5}
"""domain -> images per class"""


def _write_image(path: Path, seed: int) -> None:
    img = Image.new("RGB", (300, 260))
    img.putdata([((seed * 7 + i) % 256, (seed * 13 + i) % 256, (seed * 29 + i) % 256)
                 for i in range(300 * 260)])
    img.save(path)


@pytest.fixture
def class_folder_root(tmp_path) -> Path:
    """<root>/<domain>/<class>/<image>.png"""
    root = tmp_path / "office_home"
    n = 0
    for domain, per_class in DOMAINS.items():
        for cls in CLASSES:
            directory = root / domain / cls
            directory.mkdir(parents=True)
            for i in range(per_class):
                n += 1
                _write_image(directory / f"{i:03d}.png", n)
    return root


@pytest.fixture
def image_list_root(tmp_path) -> Path:
    """<root>/<domain>.txt with '<relative path> <label>' lines."""
    root = tmp_path / "visda17"
    (root).mkdir(parents=True)
    (root / "classnames.txt").write_text("\n".join(CLASSES) + "\n")
    n = 0
    for domain, per_class in DOMAINS.items():
        images = root / domain
        images.mkdir()
        lines = []
        for label, cls in enumerate(CLASSES):
            for i in range(per_class):
                n += 1
                name = f"{cls}_{i:03d}.png"
                _write_image(images / name, n)
                lines.append(f"{domain}/{name} {label}")
        (root / f"{domain}.txt").write_text("\n".join(lines) + "\n")
    return root
