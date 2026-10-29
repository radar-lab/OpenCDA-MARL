"""opencda.py imports torch before the libraries that break its DLL load.

On Windows ``import torch`` fails with WinError 127 (shm.dll) when cv2, open3d
or matplotlib.pyplot were imported first, and ``opencda``/``opencda_marl``
pull those in.
"""
import ast
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[2] / "opencda.py"
BREAKS_TORCH = {"cv2", "open3d", "matplotlib", "opencda", "opencda_marl"}


def _top_level_imports():
    """(module root, line) of every import reachable at module level,
    including inside top-level try blocks."""
    found = []

    def visit(statements):
        for node in statements:
            if isinstance(node, ast.Import):
                found.extend((a.name.split(".")[0], node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.append((node.module.split(".")[0], node.lineno))
            elif isinstance(node, ast.Try):
                visit(node.body)

    visit(ast.parse(ENTRY.read_text(encoding="utf-8")).body)
    return found


def test_torch_is_imported_before_the_libraries_that_break_it():
    imports = _top_level_imports()
    torch_line = min(line for name, line in imports if name == "torch")
    breakers = [line for name, line in imports if name in BREAKS_TORCH]
    assert breakers and torch_line < min(breakers)
