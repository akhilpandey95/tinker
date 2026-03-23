from __future__ import annotations

# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""Notebook-friendly shim for the QDoRA SFT trainer.

Usage from a notebook in this directory:

    import qdora_train
    qdora_train.main()

Or, if you want to inspect helpers:

    from qdora_train import parse_args, build_sft_examples, evaluate_split

This file keeps the real implementation in `src/sft/qdora_train.py` and
re-exports it from `notebooks/` so Colab can import it directly when the repo
root is not already on `PYTHONPATH`.
"""

import importlib.util
import os
import sys
from pathlib import Path


def _resolve_source_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path(os.environ["TINKER_REPO_ROOT"]) / "src" / "sft" / "qdora_train.py"
        if "TINKER_REPO_ROOT" in os.environ
        else None,
        here.parents[1] / "src" / "sft" / "qdora_train.py",
        Path.cwd().resolve() / "src" / "sft" / "qdora_train.py",
        Path.cwd().resolve().parent / "src" / "sft" / "qdora_train.py",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    checked = [str(candidate) for candidate in candidates if candidate is not None]
    raise FileNotFoundError(
        "Could not locate src/sft/qdora_train.py. "
        "Set TINKER_REPO_ROOT to the repo root if needed. "
        f"Checked: {checked}"
    )


def _load_impl():
    source_path = _resolve_source_path()
    spec = importlib.util.spec_from_file_location("tinker_qdora_train_impl", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_IMPL = _load_impl()


def __getattr__(name: str):
    return getattr(_IMPL, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_IMPL)))


if hasattr(_IMPL, "__all__"):
    __all__ = list(_IMPL.__all__)
else:
    __all__ = [name for name in dir(_IMPL) if not name.startswith("_")]
