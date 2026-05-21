"""Bundled DeViL package (https://github.com/gaostar123/DeViL).

Re-exports the public DeViL API so callers can use
``from dependence.devil import disable_torch_init, model_init, mm_infer``.
"""

from pathlib import Path

from .devil import disable_torch_init, model_init, mm_infer
from .devil.mm_utils import load_video_new, load_images

BUNDLED_ROOT = Path(__file__).resolve().parent

__all__ = [
    "BUNDLED_ROOT",
    "disable_torch_init",
    "model_init",
    "mm_infer",
    "load_video_new",
    "load_images",
]
