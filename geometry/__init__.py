"""Pure-geometry layer for fisheye camera calibration.

All formulas in this package are numpy implementations with no I/O and no
randomness (SPEC.md 2.1).  Angles are radian inside every function; only
parameters whose name ends in ``_deg`` are degrees (SPEC.md 3.1).
"""

from . import equirect, render, rotation, ucm

__all__ = ["ucm", "rotation", "equirect", "render"]
