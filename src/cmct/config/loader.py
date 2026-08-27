"""Load a Config from YAML. There is no other layer: a value comes only from a file."""
from __future__ import annotations

import dataclasses
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from cmct.config.schema import (
    BranchConfig,
    Config,
    CoTrainConfig,
    DataConfig,
    DatasetSpec,
    RunConfig,
)

_TOP_LEVEL = {"dataset", "data", "cotrain", "branches", "run"}


class ConfigError(Exception):
    pass


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if is_dataclass(tp):
        return _build(tp, value, where)

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        if value not in args:
            raise ConfigError(
                f"{where}: {value!r} is not a valid value; choose one of {list(args)}"
            )
        return value

    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if len(non_none) == len(args):
                raise ConfigError(f"{where}: must not be null")
            return None
        return _coerce(non_none[0], value, where)

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return [_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected exactly {len(args)} items, got {len(value)}")
        pairs = enumerate(zip(args, value, strict=True))
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in pairs)

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        if args and args[1] is not Any:
            return {k: _coerce(args[1], v, f"{where}.{k}") for k, v in value.items()}
        return dict(value)

    if tp is Any:
        return value
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is bool and not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false, got {value!r}")
    if isinstance(tp, type) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {type(value).__name__}")
    return value


def _build(cls: Any, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; valid keys: {sorted(names)}")

    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in raw.items()}
    missing = sorted(
        f.name for f in fields(cls)
        if f.name not in kwargs
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"{where}: missing required field(s) {missing}")
    return cls(**kwargs)


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level content must be a mapping")
    return raw


def _assemble(dataset: DatasetSpec, raw: dict) -> Config:
    unknown = sorted(set(raw) - _TOP_LEVEL)
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {unknown}; valid keys: {sorted(_TOP_LEVEL)}")
    cfg = Config(
        dataset=dataset,
        data=_build(DataConfig, raw.get("data", {}), "data"),
        cotrain=_build(CoTrainConfig, raw.get("cotrain", {}), "cotrain"),
        branches=[
            _build(BranchConfig, b, f"branches[{i}]")
            for i, b in enumerate(raw.get("branches", []))
        ],
        run=_build(RunConfig, raw.get("run", {}), "run"),
    )
    validate(cfg)
    return cfg


def load_experiment(path: str | Path, config_root: str | Path | None = None) -> Config:
    """Read an experiment YAML, resolve its dataset by name, and validate.

    The dataset YAML is looked up at `<config_root>/dataset/<name>.yaml`;
    config_root defaults to the parent of the experiment file's directory, so
    loading works from any cwd.
    """
    path = Path(path)
    raw = _read_yaml(path)
    if "dataset" not in raw:
        raise ConfigError(f"{path}: missing key 'dataset' (the dataset name)")
    name = raw.pop("dataset")
    if not isinstance(name, str):
        raise ConfigError(
            f"{path}: 'dataset' must be a dataset name, not {type(name).__name__}"
        )

    root = Path(config_root) if config_root is not None else path.parent.parent
    ds_path = root / "dataset" / f"{name}.yaml"
    dataset = _build(DatasetSpec, _read_yaml(ds_path), str(ds_path))
    if dataset.name != name:
        raise ConfigError(f"{ds_path}: name '{dataset.name}' does not match the file name '{name}'")
    return _assemble(dataset, raw)


def load_resolved(path: str | Path) -> Config:
    """Read back a file written by dump(), where `dataset` is the full spec
    rather than a name."""
    raw = _read_yaml(Path(path))
    if not isinstance(raw.get("dataset"), dict):
        raise ConfigError(f"{path}: not a resolved config; use load_experiment instead")
    dataset = _build(DatasetSpec, raw.pop("dataset"), "dataset")
    return _assemble(dataset, raw)


def dump(cfg: Config, directory: str | Path, filename: str = "config.yaml") -> Path:
    """Write the resolved config into the run directory, so every run leaves
    behind exactly the config it ran with."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / filename
    out.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False, allow_unicode=True))
    return out


def validate(cfg: Config) -> None:
    known = set(cfg.dataset.domains)
    for role in ("source_domains", "target_domains"):
        bad = [d for d in getattr(cfg.data, role) if d not in known]
        if bad:
            raise ConfigError(
                f"data.{role}: {bad} not in dataset '{cfg.dataset.name}'; "
                f"available: {cfg.dataset.domains}"
            )
        if not getattr(cfg.data, role):
            raise ConfigError(f"data.{role}: must not be empty")

    overlap = sorted(set(cfg.data.source_domains) & set(cfg.data.target_domains))
    if overlap:
        raise ConfigError(f"source_domains and target_domains overlap: {overlap}")

    if not cfg.branches:
        raise ConfigError("branches: at least one branch is required")

    names = [b.name for b in cfg.branches]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ConfigError(f"branches: duplicate names {dup}")

    for b in cfg.branches:
        if b.steps_per_macro < 1:
            raise ConfigError(
                f"branches[{b.name}].steps_per_macro: must be >= 1, got {b.steps_per_macro}"
            )
        if b.warmup_steps < 0:
            raise ConfigError(f"branches[{b.name}].warmup_steps: must be >= 0")
        if not 0.0 <= b.pseudo_label.threshold <= 1.0:
            raise ConfigError(f"branches[{b.name}].pseudo_label.threshold: must be within [0, 1]")

    if cfg.cotrain.total_macro_steps < 1:
        raise ConfigError("cotrain.total_macro_steps: must be >= 1")
