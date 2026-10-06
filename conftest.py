"""Make the project root importable so `tests/` can `import geometry`.

pytest prepends the directory holding this file to ``sys.path``; doing it
explicitly keeps a bare ``pytest`` working regardless of import mode.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
