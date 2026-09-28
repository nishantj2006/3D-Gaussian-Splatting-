"""Repository paths and module-based subprocess commands.

The editing package shares the original project's scene/renderer packages.
Invoke commands from the repository root, or install with ``pip install -e .``.
"""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def commands():
    """Return the lightweight command registry without importing GPU modules."""
    return json.loads(Path(__file__).with_name("commands.json").read_text())


def module_command(name, python=None):
    """Resolve an old stage name to its packaged CLI, preserving Python choice."""
    name = str(name)
    command = Path(name).stem
    module = commands().get(command)
    executable = str(python or sys.executable)
    if module is not None:
        return [executable, "-m", module]
    # Missing/legacy external stages retain subprocess error/report behaviour.
    return [executable, str(PROJECT_ROOT / name)]
