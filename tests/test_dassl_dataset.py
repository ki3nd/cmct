"""dassl still loads a class-subfolder dataset, and classnames keep the order
the CMKD branch's hardcoded prompt list depends on."""
from pathlib import Path
from typing import ClassVar

from vendor.dassl.data.datasets import OfficeHome


class _Cfg:
    """Minimal stand-in for the yacs cfg DataManager reads. The attribute
    shape mirrors dassl's, so it is annotated rather than restructured."""

    class DATASET:
        ROOT = ""
        SOURCE_DOMAINS: ClassVar[list[str]] = ["art"]
        TARGET_DOMAINS: ClassVar[list[str]] = ["clipart"]


def _make_tree(root: Path, classes):
    for domain in ("art", "clipart"):
        for name in classes:
            d = root / "office_home" / domain / name
            d.mkdir(parents=True)
            (d / "a.jpg").write_bytes(b"")


def test_classnames_are_sorted_folder_names_lowercased(tmp_path):
    _make_tree(tmp_path, ["Alarm_Clock", "TV", "Table", "Backpack"])
    cfg = _Cfg()
    cfg.DATASET.ROOT = str(tmp_path)
    ds = OfficeHome(cfg)
    # ASCII sort: "Alarm_Clock" < "Backpack" < "TV" < "Table"
    assert ds.classnames == ["alarm_clock", "backpack", "tv", "table"]
