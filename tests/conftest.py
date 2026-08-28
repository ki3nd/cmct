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


CHECKPOINT = Path("/home/pc1175/DA-Research/old-cmct/assets/ViT-B-16.pt")


@pytest.fixture(scope="session")
def clip_fp32():
    """One real CLIP ViT-B/16, loaded once per session.

    Only for assertions that need the real architecture. Everything else uses
    `tiny_clip`, because deep-copying 150M parameters per test is minutes, not
    seconds.
    """
    from cmct.backbones.clip import load_clip

    if not CHECKPOINT.is_file():
        pytest.skip(f"no CLIP checkpoint at {CHECKPOINT}")
    return load_clip(str(CHECKPOINT), "fp32")


def build_tiny_clip():
    """A real `CLIP` module at toy dimensions.

    Same class and same code paths as the full model, but ~1.7M parameters, so a
    test can build and deep-copy one per function. vocab_size stays 49408 because
    the tokenizer emits ids in that range; context_length stays 77.
    """
    from cmct.backbones.clip.model import CLIP

    return CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=32,
        transformer_heads=2,
        transformer_layers=1,
    ).float().eval()


@pytest.fixture
def tiny_clip():
    return build_tiny_clip()
