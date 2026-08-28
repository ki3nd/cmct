"""The prompts branch 2 builds must equal the original's hardcoded list.

The list is read out of the original file rather than pasted here: a copy in a
test only proves that two copies agree.

`vlpuda_pure/models/backbone.py` is byte-identical to upstream VLP-UDA
(bf8f0494), so this compares against upstream.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from cmct.branches.vlp_model import TEMPLATE
from cmct.config import load_experiment
from cmct.config.schema import DatasetSpec

BACKBONE = Path("/home/pc1175/DA-Research/old-cmct/vlpuda_pure/models/backbone.py")
OFFICE_HOME_DIRS = Path("/home/pc1175/DA-Research/data_root/office_home/art")

pytestmark = pytest.mark.skipif(not BACKBONE.is_file(), reason=f"no {BACKBONE}")


def original_office_home_prompts() -> list[str]:
    source = BACKBONE.read_text()
    block = source.split('if args.datasets=="office_home":')[1].split("elif")[0]
    prompts = re.findall(r"'([^']+)'", block)
    assert len(prompts) == 65, f"expected 65 prompts, parsed {len(prompts)}"
    return prompts


def test_template_matches_the_original():
    prompts = original_office_home_prompts()
    prefix = TEMPLATE.format("")
    assert all(p.startswith(prefix) for p in prompts), (
        f"TEMPLATE {TEMPLATE!r} does not produce the original's prefix; "
        f"first prompt is {prompts[0]!r}"
    )


def test_original_list_is_just_normalized_directory_names():
    """Not an arbitrary list: it is exactly the sorted directory names with
    underscores turned into spaces and lowercased. That is why the normalization
    in build_split has to be unconditional."""
    if not OFFICE_HOME_DIRS.is_dir():
        pytest.skip(f"no Office-Home directories at {OFFICE_HOME_DIRS}")
    names = sorted(p.name for p in OFFICE_HOME_DIRS.iterdir() if p.is_dir())
    expected = [TEMPLATE.format(n.replace("_", " ").lower()) for n in names]
    assert expected == original_office_home_prompts()


def test_built_prompts_equal_the_original_list():
    """End to end: what the reader plus the model actually tokenize."""
    if not OFFICE_HOME_DIRS.is_dir():
        pytest.skip(f"no Office-Home directories at {OFFICE_HOME_DIRS}")

    from cmct.data import build_split

    root = Path(__file__).resolve().parents[1] / "configs"
    cfg = load_experiment(root / "experiment" / "cmct_officehome_a2c.yaml")
    cfg.dataset = dataclasses.replace(cfg.dataset, dir="office_home")
    cfg.data = dataclasses.replace(
        cfg.data, root=str(OFFICE_HOME_DIRS.parents[1]),
        source_domains=["art"], target_domains=["clipart"],
        clarify_classnames=False,
    )
    assert isinstance(cfg.dataset, DatasetSpec)
    split = build_split(cfg.dataset, cfg.data)
    built = [TEMPLATE.format(c) for c in split.classnames]
    assert built == original_office_home_prompts()


def test_clarify_classnames_deviates_from_the_original():
    """Stated as a test so the deviation cannot be mistaken for a cleanup: with
    the flag on, the prompts are NOT the original's."""
    if not OFFICE_HOME_DIRS.is_dir():
        pytest.skip(f"no Office-Home directories at {OFFICE_HOME_DIRS}")

    from cmct.data import build_split

    root = Path(__file__).resolve().parents[1] / "configs"
    cfg = load_experiment(root / "experiment" / "cmct_officehome_a2c.yaml")
    cfg.dataset = dataclasses.replace(cfg.dataset, dir="office_home")
    cfg.data = dataclasses.replace(
        cfg.data, root=str(OFFICE_HOME_DIRS.parents[1]),
        source_domains=["art"], target_domains=["clipart"],
        clarify_classnames=True,
    )
    split = build_split(cfg.dataset, cfg.data)
    built = [TEMPLATE.format(c) for c in split.classnames]
    assert built != original_office_home_prompts()
    differing = sum(a != b for a, b in
                    zip(built, original_office_home_prompts(), strict=True))
    assert differing == 21, f"expected 21 renamed classes, got {differing}"
