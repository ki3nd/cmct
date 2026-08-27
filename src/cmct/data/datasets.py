"""Dataset readers: on-disk layout -> DomainSplit.

A reader turns one dataset's directory or split-file convention into
list[Datum]. No transforms, no loaders. Register with @register("<key>"); the
key is what DatasetSpec.name / DataConfig.dataset names.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cmct.config.schema import DataConfig, DatasetSpec
from cmct.data.datum import Datum, DomainSplit

Reader = Callable[[Path, list[str], list[str]], DomainSplit]
LAYOUTS: dict[str, Reader] = {}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DataError(Exception):
    pass


def register(key: str) -> Callable[[Reader], Reader]:
    """Add a reader to LAYOUTS. A duplicate key raises: two readers claiming one
    name is how a run ends up reading a dataset nobody intended."""

    def wrap(fn: Reader) -> Reader:
        if key in LAYOUTS:
            raise DataError(f"layout '{key}' is already registered by {LAYOUTS[key].__name__}")
        LAYOUTS[key] = fn
        return fn

    return wrap


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES and not p.name.startswith(".")
    )


def _class_names(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


@register("class_folder")
def read_class_folders(root: Path, source: list[str], target: list[str]) -> DomainSplit:
    """`<root>/<domain>/<class>/<image>`.

    Class names are read from the FIRST source domain and reused for every
    domain, so all domains of one dataset agree on the label mapping even if a
    domain is missing a class directory.
    """
    classnames = _class_names(_domain_dir(root, source[0]))
    if not classnames:
        raise DataError(f"{root / source[0]}: no class directories found")
    label_of = {c: i for i, c in enumerate(classnames)}

    def read(domains: list[str]) -> list[Datum]:
        items: list[Datum] = []
        for domain_index, domain in enumerate(domains):
            domain_dir = _domain_dir(root, domain)
            for class_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
                if class_dir.name not in label_of:
                    raise DataError(
                        f"{class_dir}: class '{class_dir.name}' is absent from domain "
                        f"'{source[0]}'; the label mapping would differ between domains"
                    )
                for path in _list_images(class_dir):
                    items.append(
                        Datum(str(path), label_of[class_dir.name], domain_index, class_dir.name)
                    )
        return items

    train_u = read(target)
    return DomainSplit(train_x=read(source), train_u=train_u, test=list(train_u),
                       classnames=classnames)


@register("image_list")
def read_image_lists(root: Path, source: list[str], target: list[str]) -> DomainSplit:
    """`<root>/<domain>.txt`, each line `<path relative to root> <label>`.

    Class names come from `<root>/classnames.txt` (one per line, in label
    order). Requiring that file rather than deriving names from paths keeps the
    label mapping explicit, since these splits carry integer labels only.
    """
    names_file = root / "classnames.txt"
    if not names_file.is_file():
        raise DataError(
            f"{names_file}: layout 'image_list' requires this file (one class name per line)"
        )
    classnames = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]
    if not classnames:
        raise DataError(f"{names_file}: is empty")

    def read(domains: list[str]) -> list[Datum]:
        items: list[Datum] = []
        for domain_index, domain in enumerate(domains):
            list_file = root / f"{domain}.txt"
            if not list_file.is_file():
                raise DataError(f"{list_file}: split file for domain '{domain}' not found")
            for lineno, line in enumerate(list_file.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    raise DataError(
                        f"{list_file}:{lineno}: expected '<path> <label>', got {line!r}"
                    )
                rel, raw_label = parts
                try:
                    label = int(raw_label)
                except ValueError:
                    raise DataError(
                        f"{list_file}:{lineno}: label is not an integer: {raw_label!r}"
                    ) from None
                if not 0 <= label < len(classnames):
                    raise DataError(
                        f"{list_file}:{lineno}: label {label} outside [0, {len(classnames)})"
                    )
                items.append(Datum(str(root / rel), label, domain_index, classnames[label]))
        return items

    train_u = read(target)
    return DomainSplit(train_x=read(source), train_u=train_u, test=list(train_u),
                       classnames=classnames)


def _domain_dir(root: Path, domain: str) -> Path:
    directory = root / domain
    if not directory.is_dir():
        raise DataError(f"{directory}: domain directory '{domain}' not found")
    return directory


def build_split(dataset: DatasetSpec, data: DataConfig) -> DomainSplit:
    """Read the dataset named by `dataset` using the layout it declares."""
    if dataset.layout not in LAYOUTS:
        raise DataError(
            f"layout '{dataset.layout}' is not registered; available: {sorted(LAYOUTS)}"
        )
    root = Path(data.root) / dataset.dir
    if not root.is_dir():
        raise DataError(f"{root}: dataset directory not found")

    split = LAYOUTS[dataset.layout](root, data.source_domains, data.target_domains)
    if split.num_classes != dataset.num_classes:
        raise DataError(
            f"{root}: read {split.num_classes} classes but "
            f"configs/dataset/{dataset.name}.yaml declares num_classes={dataset.num_classes}"
        )
    if not split.train_x:
        raise DataError(f"{root}: no images found for source domains {data.source_domains}")
    if not split.train_u:
        raise DataError(f"{root}: no images found for target domains {data.target_domains}")

    if data.clarify_classnames:
        split.classnames = [
            dataset.classname_overrides.get(c.replace("_", " ").lower(),
                                            c.replace("_", " ").lower())
            for c in split.classnames
        ]
    return split
