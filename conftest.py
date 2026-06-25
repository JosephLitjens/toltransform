"""
Root conftest.py — pytest session bootstrap for TolTransform.

The 'io/' package name collides with Python's frozen stdlib 'io' module.
Python's FrozenImporter sits ahead of PathFinder in sys.meta_path, so normal
'from io.schema import ...' statements always resolve to the stdlib io module,
not our local package.

Fix: pre-load our local io submodules via importlib and register them in
sys.modules before test collection begins. Python's _find_and_load() checks
sys.modules first, so subsequent 'from io.schema import X' and
'from io.serializer import Y' statements will find our versions.

The stdlib io module is not touched — only io.schema and io.serializer are added
to sys.modules under their dotted names.
"""
import sys
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).parent


def _preload_local_io() -> None:
    for name in ("schema", "serializer"):
        full_name = f"io.{name}"
        if full_name in sys.modules:
            continue
        path = _ROOT / "io" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full_name, str(path))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "io"
        # Register BEFORE exec so io.serializer's 'from io.schema import ...' works.
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)


_preload_local_io()
