"""Compatibility re-export for the visibility status modules.

This allows imports like `from status import snapshot_from_dict` to work from the
repository root while keeping the real implementation in the visibility package.
"""

from pathlib import Path
import importlib.util
import sys

_module_path = Path(__file__).resolve().parent / "visibility" / "status.py"
_spec = importlib.util.spec_from_file_location("visibility.status", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_module_path}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

Stage = _module.Stage
TrainingRun = _module.TrainingRun
VisibilitySnapshot = _module.VisibilitySnapshot
utc_now = _module.utc_now
snapshot_from_dict = _module.snapshot_from_dict

__all__ = [
    "Stage",
    "TrainingRun",
    "VisibilitySnapshot",
    "utc_now",
    "snapshot_from_dict",
]
