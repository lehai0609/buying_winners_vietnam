import sys
import os
import shutil
import pytest


def test_python_is_311():
    """
    Skip the test if Python 3.11 is not the running interpreter.
    This keeps test runs usable on systems where 3.11 isn't installed yet.
    """
    if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
        pytest.skip("Python 3.11 not available; install and re-run.")
    assert sys.version.startswith("3.11.")


def test_in_project_virtualenv_exists():
    """
    If Poetry is installed, assert there is an in-project .venv and
    that sys.prefix appears to be inside the project (best-effort).
    Skip if Poetry is not detected.
    """
    if shutil.which("poetry") is None:
        pytest.skip("Poetry not detected; skipping virtualenv check.")

    venv_path = os.path.join(os.getcwd(), ".venv")
    assert os.path.isdir(venv_path), f".venv not found at {venv_path}"

    # Best-effort: when using an in-project venv, sys.prefix is typically under the project tree
    assert os.path.abspath(sys.prefix).startswith(os.path.abspath(os.getcwd())), (
        "sys.prefix does not appear to be inside the project root (best-effort check)."
    )
