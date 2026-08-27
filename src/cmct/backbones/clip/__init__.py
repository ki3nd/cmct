"""Vendored CLIP.

Architecture and tokenizer are openai/CLIP @
d05afc436d78f1c48dc0dbf8e5980a9d471f35f6, unmodified apart from the vocab file
moving into `assets/`. `clip.py` is not vendored; see `tokenize.py`.
"""
from cmct.backbones.clip.factory import load_clip, load_state_dict
from cmct.backbones.clip.model import CLIP, build_model, convert_weights
from cmct.backbones.clip.tokenize import tokenize
from cmct.backbones.clip.tokenizer import SimpleTokenizer

__all__ = [
    "CLIP", "SimpleTokenizer", "build_model", "convert_weights", "load_clip",
    "load_state_dict", "tokenize",
]
