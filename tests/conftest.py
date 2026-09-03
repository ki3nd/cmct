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
    for anyone who has the weights staged elsewhere.
    """
    candidates = []
    env_path = os.environ.get("CMCT_CLIP_WEIGHTS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.home() / ".cache" / "clip" / "ViT-B-16.pt")
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


def load_fixture(name):
    """Read a frozen baseline fixture from tests/fixtures/."""
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


def read_text(path):
    """Read a whole text file. Used by config tests that mutate the shipped YAML."""
    with open(path) as f:
        return f.read()
