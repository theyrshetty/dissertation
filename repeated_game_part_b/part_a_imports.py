"""Make the sibling Part A modules importable when Part B is run directly."""

from pathlib import Path
import sys
import importlib.util

PART_B_DIR = Path(__file__).resolve().parent
PART_A_DIR = PART_B_DIR.parent / "repeated_game"


def _pin_local_module(name: str) -> None:
    """Load one of Part B's own modules and lock it into sys.modules under its
    bare name *before* Part A is added to sys.path.

    Part A ships a same-named module (e.g. scoring.py). Appending Part A's
    directory to the *end* of sys.path is supposed to make Part B's copy win,
    but that only helps for imports that haven't happened yet. If any Part A
    module is imported first and it (directly, or via its own sys.path
    tinkering) ends up importing "scoring" before Part B's own `from scoring
    import ...` line runs, Python's sys.modules cache will keep Part A's
    version for the rest of the process - silently, since no error is raised
    until something tries to use a name Part A's copy doesn't have. Pinning
    the name here removes the race entirely.
    """
    if name in sys.modules:
        return
    file_path = PART_B_DIR / f"{name}.py"
    if not file_path.exists():
        return
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


if str(PART_A_DIR) not in sys.path:
    # Keep this Part B directory first: it has its own scoring.py and runner.py.
    # Part A is only a fallback for shared modules such as generator and solver.
    sys.path.append(str(PART_A_DIR))

# Reserve Part B's own same-named modules before any Part A module gets a
# chance to import (or shadow) them first. Must happen after PART_A_DIR is on
# sys.path, since Part B's scoring.py itself needs Part A's translator_solver
# and solver to resolve (via dependency_graph).
_pin_local_module("scoring")
