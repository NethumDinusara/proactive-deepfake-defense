"""
Put `src/` and `third_party/` on sys.path.

Every script in this directory imports this first. It means the scripts run
correctly from any working directory -- previously they had to be launched
from the project root or `from utils import ...` would fail.

    import _path  # noqa: F401
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

for _p in (ROOT / "src", ROOT / "third_party"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
