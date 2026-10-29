"""
Environment shims. Import this FIRST, before torch or diffusers.

`facenet_pytorch` pulls in a dependency chain that tries to import the `aim`
experiment tracker, which does not install cleanly on Windows. Stubbing the
module out is cheaper than carrying the dependency; nothing here calls it.

This replaces a ten-line block that was copy-pasted into six separate scripts.
"""
import sys
import types
from unittest.mock import MagicMock


def install_aim_stub() -> None:
    """Register empty `aim` / `aim.sdk` modules so imports of them succeed."""
    if "aim" in sys.modules:
        return
    aim = types.ModuleType("aim")
    aim.__spec__ = MagicMock()
    aim.__path__ = []
    sys.modules["aim"] = aim

    aim_sdk = types.ModuleType("aim.sdk")
    aim_sdk.__spec__ = MagicMock()
    sys.modules["aim.sdk"] = aim_sdk


install_aim_stub()
