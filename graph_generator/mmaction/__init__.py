from __future__ import annotations

import pkgutil
from pathlib import Path

__path__ = pkgutil.extend_path(__path__, __name__)

_mmaction2_path = Path(__file__).resolve().parents[1] / "mmaction2" / "mmaction"
if _mmaction2_path.exists():
    __path__.append(str(_mmaction2_path))

