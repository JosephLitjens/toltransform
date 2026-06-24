"""
Sole point of contact with pytransform3d. All conversion functions accept and
return plain np.ndarray — no pytransform3d types leak out of this module.

Convention reference (authoritative):
  - Euler angles: intrinsic ZYX, ordered [ez, ey, ex] where ez = rotation
    applied first (around Z), ey = second (around new Y'), ex = third (around
    new X''). This is the yaw-pitch-roll decomposition.
  - Quaternion: (w, x, y, z) order — wxyz. The scalar w component comes first.
  - Screw axis: 6-vector (omega_1, omega_2, omega_3, v_1, v_2, v_3) where the
    first three components encode rotation and the last three encode translation.
  - Rotation vectors: ω = θ·u (axis scaled by angle, not a unit vector).
"""

import warnings

import numpy as np
import pytransform3d as _p3d
import pytransform3d.rotations as _pr
import pytransform3d.transformations as _pt

# ── Version guard ────────────────────────────────────────────────────────────
_TESTED_MAJOR = 3
_installed = tuple(int(x) for x in _p3d.__version__.split(".")[:2])
if _installed[0] != _TESTED_MAJOR:
    warnings.warn(
        f"pytransform3d major version changed from {_TESTED_MAJOR} to "
        f"{_installed[0]}. Euler/quaternion/screw conventions may have "
        "shifted — verify conversions.py against the new API.",
        UserWarning,
        stacklevel=1,
    )

_ZERO_ANGLE_TOL = 1e-12


# ── Euler ────────────────────────────────────────────────────────────────────

def euler_to_rotation_matrix(
    euler_angles: np.ndarray,
    convention: str = "intrinsic_zyx",
) -> np.ndarray:
    """Return 3x3 rotation matrix from Euler angles.

    Parameters
    ----------
    euler_angles : array-like, shape (3,)
        Angles [ez, ey, ex] in radians (intrinsic ZYX = yaw-pitch-roll).
    convention : str
        Must be "intrinsic_zyx" (the only supported convention).
    """
    if convention != "intrinsic_zyx":
        raise ValueError(
            f"Unsupported Euler convention '{convention}'. "
            "Only 'intrinsic_zyx' is supported."
        )
    e = np.asarray(euler_angles, dtype=float)
    # i=2(Z), j=1(Y), k=0(X), extrinsic=False → intrinsic ZYX
    return _pr.matrix_from_euler(e, i=2, j=1, k=0, extrinsic=False)


def rotation_matrix_to_euler(
    R: np.ndarray,
    convention: str = "intrinsic_zyx",
) -> np.ndarray:
    """Return Euler angles [ez, ey, ex] in radians from a 3x3 rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
    convention : str
        Must be "intrinsic_zyx".
    """
    if convention != "intrinsic_zyx":
        raise ValueError(
            f"Unsupported Euler convention '{convention}'. "
            "Only 'intrinsic_zyx' is supported."
        )
    # i=2(Z), j=1(Y), k=0(X), extrinsic=False → intrinsic ZYX
    return _pr.euler_from_matrix(np.asarray(R, dtype=float), i=2, j=1, k=0, extrinsic=False)


# ── Quaternion ───────────────────────────────────────────────────────────────

def quaternion_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix from unit quaternion (w, x, y, z).

    Parameters
    ----------
    quat_wxyz : array-like, shape (4,)
        Unit quaternion with scalar-first ordering: (w, x, y, z).
    """
    q = np.asarray(quat_wxyz, dtype=float)
    return _pr.matrix_from_quaternion(q)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return unit quaternion (w, x, y, z) from 3x3 rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)

    Returns
    -------
    np.ndarray, shape (4,)
        Quaternion with scalar-first ordering: (w, x, y, z).
    """
    return _pr.quaternion_from_matrix(np.asarray(R, dtype=float))


# ── Screw ────────────────────────────────────────────────────────────────────

def screw_to_matrix(
    axis: np.ndarray,
    angle: float,
    translation_along_axis: float,
    point_on_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Return 4x4 HTM from screw parameters.

    Parameters
    ----------
    axis : array-like, shape (3,)
        Unit direction vector of the screw axis.
    angle : float
        Rotation about the axis in radians.
    translation_along_axis : float
        Total translation distance along the axis.
    point_on_axis : array-like, shape (3,) or None
        Any point on the screw axis. Defaults to the origin, which is only
        correct when the axis passes through the origin.

    Notes
    -----
    When angle ≈ 0 the exponential-coordinate formula is numerically
    singular; that case is handled as pure translation.
    """
    s = np.asarray(axis, dtype=float)
    q = np.zeros(3) if point_on_axis is None else np.asarray(point_on_axis, dtype=float)

    if abs(angle) < _ZERO_ANGLE_TOL:
        T = np.eye(4)
        T[:3, 3] = translation_along_axis * s
        return T

    pitch = translation_along_axis / angle
    screw_axis = _pt.screw_axis_from_screw_parameters(q, s, pitch)
    Stheta = screw_axis * angle
    return _pt.transform_from_exponential_coordinates(Stheta)


def matrix_to_screw(T: np.ndarray) -> dict:
    """Return screw parameters from a 4x4 HTM.

    Returns
    -------
    dict with keys:
        axis : np.ndarray (3,) — unit direction vector
        angle : float — rotation in radians (≥ 0)
        translation_along_axis : float — translation along the axis
        point_on_axis : np.ndarray (3,) or None
    """
    T = np.asarray(T, dtype=float)
    Stheta = _pt.exponential_coordinates_from_transform(T)
    omega = Stheta[:3]
    theta = float(np.linalg.norm(omega))

    if theta < _ZERO_ANGLE_TOL:
        # Pure translation — screw formula undefined; read directly from T.
        t = T[:3, 3].copy()
        t_norm = float(np.linalg.norm(t))
        axis = t / t_norm if t_norm > _ZERO_ANGLE_TOL else np.array([0.0, 0.0, 1.0])
        return {
            "axis": axis,
            "angle": 0.0,
            "translation_along_axis": t_norm,
            "point_on_axis": None,
        }

    S = Stheta / theta  # unit screw axis
    q, s_axis, pitch = _pt.screw_parameters_from_screw_axis(S)
    return {
        "axis": s_axis,
        "angle": theta,
        "translation_along_axis": float(pitch * theta),
        "point_on_axis": q,
    }
