"""Bridge direct coursework execution to the canonical local package."""

import importlib
import runpy
import sys
from pathlib import Path


def alias_module(legacy_name, canonical_name):
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    # Alias the module itself so globals, lazy attributes, and patches stay shared.
    sys.modules[legacy_name] = importlib.import_module(canonical_name)


def forward_module(legacy_name, canonical_name):
    """Retain both direct script execution and module identity for old callers."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    if legacy_name == "__main__":
        runpy.run_module(canonical_name, run_name="__main__")
    else:
        alias_module(legacy_name, canonical_name)
