"""Resolve a CLIP checkpoint: a local path, or a model name to fetch and cache.

The URLs and the caching rule come from openai/CLIP's `clip.py`, which is not
vendored wholesale. The SHA-256 of each file is the second-to-last path segment
of its own URL, which is what makes verification possible without a separate
manifest.

The default cache directory is the one openai/CLIP uses, so a checkpoint already
downloaded by any other tool is reused rather than fetched again.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/clip"))
_CHUNK = 1 << 16


def available_models() -> list[str]:
    return sorted(_MODELS)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(name: str, cache_dir: Path | str = DEFAULT_CACHE) -> Path:
    """Fetch `name` into `cache_dir` unless a file already there verifies.

    A file whose digest does not match is re-downloaded rather than trusted: a
    truncated download would otherwise fail later, inside model construction,
    where the cause is far from obvious.
    """
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}; available: {available_models()}")
    url = _MODELS[name]
    expected = url.split("/")[-2]
    target = Path(cache_dir) / os.path.basename(url)

    if target.exists() and not target.is_file():
        raise RuntimeError(f"{target} exists and is not a regular file")
    if target.is_file():
        if _sha256_of(target) == expected:
            return target
        print(f"{target}: SHA-256 mismatch, re-downloading")

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} -> {target}")
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as source, partial.open("wb") as output:
        total = int(source.info().get("Content-Length") or 0)
        done = 0
        while chunk := source.read(_CHUNK):
            output.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {100 * done / total:5.1f}%  {done >> 20} / {total >> 20} MiB",
                      end="", flush=True)
    print()

    digest = _sha256_of(partial)
    if digest != expected:
        partial.unlink()
        raise RuntimeError(
            f"{name}: SHA-256 of the downloaded file is {digest}, expected {expected}"
        )
    # Rename only after verification, so an interrupted download never leaves
    # something that looks like a usable checkpoint.
    partial.replace(target)
    return target


def resolve_checkpoint(checkpoint: str, cache_dir: Path | str = DEFAULT_CACHE) -> Path:
    """A local path is used as given; a known model name is fetched and cached.

    A path wins over a name, so a file literally called "RN50" in the working
    directory is still read rather than triggering a download.
    """
    path = Path(checkpoint)
    if path.is_file():
        return path
    if checkpoint in _MODELS:
        return download_checkpoint(checkpoint, cache_dir)
    raise FileNotFoundError(
        f"{checkpoint!r} is neither an existing file nor a known model name "
        f"({available_models()})"
    )
