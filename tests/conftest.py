import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(REPO_ROOT))


def _clip_weight_candidates():
    """Candidate ViT-B/16 weight paths, in priority order.

    The default is CLIP's own download cache; `CMCT_CLIP_WEIGHTS` overrides it
    for anyone who has the weights staged elsewhere. The repo's own configured
    location (./assets) is checked last, since shipped backups have lower priority.
    """
    candidates = []
    env_path = os.environ.get("CMCT_CLIP_WEIGHTS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.home() / ".cache" / "clip" / "ViT-B-16.pt")
    candidates.append(REPO_ROOT / "assets" / "ViT-B-16.pt")
    return candidates


@pytest.fixture(scope="session")
def clip_weights():
    """ViT-B/16 weights, needed by tests that load the real CLIP model."""
    candidates = _clip_weight_candidates()
    for path in candidates:
        if path.is_file():
            return str(path)
    tried = ", ".join(str(p) for p in candidates)
    pytest.skip(f"CLIP weights not found; tried: {tried} (set CMCT_CLIP_WEIGHTS to override)")


@pytest.fixture(scope="session")
def clip_model_factory(clip_weights):
    """Factory building a fresh ViT-B/16 CLIP model on each call.

    Session-scoped: resolving the weights directory (and running the
    `clip_weights` skip-guard) only needs to happen once per session. Each call
    still returns an independent model instance, since several tests (LoRA
    injection in particular) mutate a model's submodules in place -- sharing
    one instance across tests would leak state between them.

    This is the one place `load_clip_to_cpu("ViT-B/16", <weights dir>)` is
    spelled out; the directory always comes from `os.path.dirname(clip_weights)`
    so the fixture's skip-guard actually gates what gets loaded (see
    `clip_weights` above -- passing a bare filename or a wrong-priority literal
    path here has twice caused a network download instead of a skip).
    """
    from cmct.branch_lora.model import load_clip_to_cpu

    weights_dir = os.path.dirname(clip_weights)

    def _build():
        return load_clip_to_cpu("ViT-B/16", weights_dir).float()

    return _build


@pytest.fixture
def clip_model(clip_model_factory):
    """A freshly built ViT-B/16 CLIP model, for tests that need exactly one."""
    return clip_model_factory()


def load_fixture(name):
    """Read a frozen baseline fixture from tests/fixtures/."""
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


def read_text(path):
    """Read a whole text file. Used by config tests that mutate the shipped YAML."""
    with open(path) as f:
        return f.read()
