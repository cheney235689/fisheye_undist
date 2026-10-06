"""Yaw/pitch/roll to rotation matrix.

SPEC.md 3.3.  ``d_world = R @ d_cam`` with
``R = R_yaw(about y) @ R_pitch(about x) @ R_roll(about z)``.

Camera axes are OpenCV's (x right, y down, z forward) and the world frame
coincides with the camera frame at ``R = I``.  Because latitude on the
panorama is ``lambda = arcsin(-y)``, a positive pitch tilts the optical axis
to a positive latitude (camera looks up) and a positive yaw increases the
longitude ``phi = atan2(x, z)`` (camera turns right).
"""

import numpy as np

__all__ = ["rotation_from_ypr", "rot_x", "rot_y", "rot_z"]


def rot_x(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about the x axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about the y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about the z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build the 3x3 camera-to-world rotation from yaw/pitch/roll in degrees."""
    return (
        rot_y(np.radians(float(yaw_deg)))
        @ rot_x(np.radians(float(pitch_deg)))
        @ rot_z(np.radians(float(roll_deg)))
    )
