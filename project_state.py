"""Compatibility re-export for the visibility project-state logic."""

from pathlib import Path
import importlib.util
import sys

_module_path = Path(__file__).resolve().parent / "visibility" / "project_state.py"
_spec = importlib.util.spec_from_file_location("visibility.project_state", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_module_path}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

derive_project_state = _module.derive_project_state

__all__ = ["derive_project_state"]
