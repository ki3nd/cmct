"""Checkpoint resolution and SHA-256 verification. No network is used."""
from __future__ import annotations

import hashlib

import pytest

from cmct.backbones.clip import download as dl


def test_urls_carry_their_own_digest():
    """The SHA-256 is the second-to-last path segment of each URL -- that is what
    lets a download be verified without a separate manifest."""
    for name, url in dl._MODELS.items():
        digest = url.split("/")[-2]
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), name


def test_vit_b_16_url_matches_the_original():
    assert dl._MODELS["ViT-B/16"].endswith(
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
    )


def test_an_existing_path_is_used_as_given(tmp_path):
    path = tmp_path / "ViT-B-16.pt"
    path.write_bytes(b"not really a checkpoint")
    assert dl.resolve_checkpoint(str(path)) == path


def test_a_path_wins_over_a_model_name(tmp_path, monkeypatch):
    """A file literally called RN50 must be read, not treated as a download
    request."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "RN50").write_bytes(b"local file")
    called = []
    monkeypatch.setattr(dl, "download_checkpoint", lambda *a, **k: called.append(a))
    assert dl.resolve_checkpoint("RN50").name == "RN50"
    assert not called


def test_unknown_name_lists_what_is_available(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither an existing file nor a known"):
        dl.resolve_checkpoint(str(tmp_path / "ViT-Z/99"))


def test_download_is_skipped_when_the_cached_file_verifies(tmp_path, monkeypatch):
    payload = b"pretend checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        dl._MODELS, "FAKE",
        f"https://example.invalid/clip/models/{digest}/FAKE.pt",
    )
    (tmp_path / "FAKE.pt").write_bytes(payload)

    def refuse(*_a, **_k):
        raise AssertionError("should not have opened the network")

    monkeypatch.setattr(dl.urllib.request, "urlopen", refuse)
    assert dl.download_checkpoint("FAKE", tmp_path) == tmp_path / "FAKE.pt"


def test_a_corrupt_cached_file_is_re_downloaded(tmp_path, monkeypatch):
    payload = b"pretend checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        dl._MODELS, "FAKE",
        f"https://example.invalid/clip/models/{digest}/FAKE.pt",
    )
    (tmp_path / "FAKE.pt").write_bytes(b"truncated")

    monkeypatch.setattr(dl.urllib.request, "urlopen",
                        lambda url: _FakeResponse(payload))
    result = dl.download_checkpoint("FAKE", tmp_path)
    assert result.read_bytes() == payload


def test_a_bad_download_raises_and_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setitem(
        dl._MODELS, "FAKE",
        f"https://example.invalid/clip/models/{'0' * 64}/FAKE.pt",
    )
    monkeypatch.setattr(dl.urllib.request, "urlopen",
                        lambda url: _FakeResponse(b"wrong bytes"))
    with pytest.raises(RuntimeError, match="SHA-256 of the downloaded file"):
        dl.download_checkpoint("FAKE", tmp_path)
    assert not (tmp_path / "FAKE.pt").exists()
    assert not (tmp_path / "FAKE.pt.part").exists()


def test_unknown_name_in_download_raises():
    with pytest.raises(ValueError, match="unknown model"):
        dl.download_checkpoint("ViT-Z/99")


class _FakeResponse:
    """Minimal stand-in for urlopen's context manager."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def info(self):
        return {"Content-Length": str(len(self._payload))}

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk
