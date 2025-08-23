"""
Minimal utility functions for M0 (Environment & Repro).

Functions:
- set_global_seed(seed: int) -> None
- get_env_info() -> dict[str, str]
- deterministic_sample(seed: int) -> list[int]
"""
from __future__ import annotations

import os
import sys
import platform
import random
from typing import Dict, List


def set_global_seed(seed: int) -> None:
    """
    Set deterministic seeds for stdlib randomness and record PYTHONHASHSEED.

    Raises:
        ValueError: if `seed` is not an int or is negative.
    Notes:
        - Setting PYTHONHASHSEED in-process only affects newly spawned Python processes.
        - This function seeds the `random` module for deterministic behavior in the
          current process.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


def get_env_info() -> Dict[str, str]:
    """
    Return basic environment metadata.

    Keys:
      - "python_version": e.g. "3.11.6"
      - "platform": OS/platform string from platform.platform()
      - "poetry_virtualenv": path to detected virtualenv ('.venv' or VIRTUAL_ENV) or ""
      - "cwd": current working directory
    """
    poetry_virtualenv = ""

    # Prefer an explicit VIRTUAL_ENV environment variable (common for venvs)
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env and os.path.isdir(venv_env):
        poetry_virtualenv = os.path.abspath(venv_env)
    else:
        # Check for in-project .venv
        candidate = os.path.join(os.getcwd(), ".venv")
        if os.path.isdir(candidate):
            poetry_virtualenv = os.path.abspath(candidate)
        else:
            # Best-effort: if sys.prefix resides under the project, assume it's the venv
            project_root = os.path.abspath(os.getcwd())
            try:
                prefix_abs = os.path.abspath(sys.prefix)
                if prefix_abs.startswith(project_root):
                    poetry_virtualenv = prefix_abs
            except Exception:
                poetry_virtualenv = ""

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "poetry_virtualenv": poetry_virtualenv,
        "cwd": os.getcwd(),
    }


def deterministic_sample(seed: int, length: int = 10, upper: int = 9999) -> List[int]:
    """
    Return a deterministic list of integers derived from `seed`.

    Args:
        seed: integer seed (required)
        length: number of integers to produce (default 10)
        upper: inclusive upper bound for generated integers (default 9999)

    Raises:
        ValueError: if `seed` is not an int or is negative.

    Returns:
        A list of `length` integers in range [0, upper] produced by a local RNG.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = random.Random(seed)
    return [rng.randint(0, upper) for _ in range(length)]
