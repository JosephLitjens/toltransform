"""
HTM — 4x4 homogeneous transformation matrix.

Euler convention (locked): intrinsic ZYX.
  Angles are ordered [ez, ey, ex] = [yaw, pitch, roll].
  Apply Z rotation first, then Y', then X''.

All pytransform3d calls are routed through core/conversions.py.
This module never imports pytransform3d directly.
"""

from __future__ import annotations

import numpy as np

from core import conversions

# Locked convention — single definition, referenced by all methods.
_EULER_CONVENTION = "intrinsic_zyx"

_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_BOTTOM_ROW_TOL = 1e-9
_ORTHO_TOL = 1e-6


class HTM:
    """4x4 homogeneous transformation matrix wrapping a float64 NumPy array.

    Always construct via a named class method — from_xyz_euler, from_matrix,
    from_quaternion, or from_screw — not directly via HTM().
    """

    def __init__(self, matrix: np.ndarray, input_representation: dict | None) -> None:
        """Validate and store a 4x4 HTM. Called only by named constructors."""
        matrix = np.asarray(matrix, dtype=np.float64)

        if matrix.shape != (4, 4):
            raise ValueError(
                f"HTM matrix must have shape (4, 4); got {matrix.shape}."
            )

        bottom = matrix[3, :]
        if not np.allclose(bottom, _BOTTOM_ROW, atol=_BOTTOM_ROW_TOL):
            raise ValueError(
                f"HTM bottom row must be [0, 0, 0, 1]; "
                f"got [{bottom[0]:.6g}, {bottom[1]:.6g}, {bottom[2]:.6g}, {bottom[3]:.6g}]."
            )

        R = matrix[:3, :3]
        RtR = R.T @ R
        if not np.allclose(RtR, np.eye(3), atol=_ORTHO_TOL):
            max_err = float(np.max(np.abs(RtR - np.eye(3))))
            raise ValueError(
                f"HTM rotation block is not orthonormal (max |R^T R - I| = {max_err:.3e}; "
                f"tolerance {_ORTHO_TOL:.0e}). Check for non-unit axis or accumulated "
                "floating-point error."
            )

        det = float(np.linalg.det(R))
        if abs(det - 1.0) > _ORTHO_TOL:
            raise ValueError(
                f"HTM rotation block determinant is {det:.6f}; expected 1.0 "
                f"(tolerance {_ORTHO_TOL:.0e}). Matrix may represent a reflection."
            )

        self._matrix = matrix
        self._input_representation = input_representation

    # ── Named constructors ──────────────────────────────────────────────────

    @classmethod
    def from_xyz_euler(
        cls,
        xyz: np.ndarray,
        euler_angles: np.ndarray,
        convention: str = _EULER_CONVENTION,
    ) -> HTM:
        """Construct from a translation vector and intrinsic ZYX Euler angles.

        Parameters
        ----------
        xyz : array-like, shape (3,)
            Translation [x, y, z].
        euler_angles : array-like, shape (3,)
            [ez, ey, ex] in radians — yaw (Z), pitch (Y'), roll (X'').
        convention : str
            Must be "intrinsic_zyx".
        """
        xyz = np.asarray(xyz, dtype=float)
        euler_angles = np.asarray(euler_angles, dtype=float)
        R = conversions.euler_to_rotation_matrix(euler_angles, convention=convention)
        T = _build_htm(R, xyz)
        return cls(
            T,
            {
                "kind": "xyz_euler",
                "raw_params": {"xyz": xyz.copy(), "euler_angles": euler_angles.copy(), "convention": convention},
            },
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> HTM:
        """Construct from a 4x4 NumPy array (validated on entry)."""
        m = np.asarray(matrix, dtype=np.float64)
        return cls(m, {"kind": "matrix", "raw_params": {"matrix": m.copy()}})

    @classmethod
    def from_quaternion(cls, quat_wxyz: np.ndarray, xyz: np.ndarray) -> HTM:
        """Construct from a unit quaternion (w, x, y, z) and translation vector.

        Parameters
        ----------
        quat_wxyz : array-like, shape (4,)
            Unit quaternion, scalar-first: (w, x, y, z).
        xyz : array-like, shape (3,)
            Translation [x, y, z].
        """
        quat_wxyz = np.asarray(quat_wxyz, dtype=float)
        xyz = np.asarray(xyz, dtype=float)
        R = conversions.quaternion_to_rotation_matrix(quat_wxyz)
        T = _build_htm(R, xyz)
        return cls(
            T,
            {
                "kind": "quaternion",
                "raw_params": {"quat_wxyz": quat_wxyz.copy(), "xyz": xyz.copy()},
            },
        )

    @classmethod
    def from_screw(
        cls,
        axis: np.ndarray,
        angle: float,
        translation_along_axis: float,
        point_on_axis: np.ndarray | None = None,
    ) -> HTM:
        """Construct from screw parameters.

        Parameters
        ----------
        axis : array-like, shape (3,)
            Unit direction vector of the screw axis.
        angle : float
            Rotation about the axis in radians.
        translation_along_axis : float
            Total translation along the axis.
        point_on_axis : array-like, shape (3,) or None
            A point on the screw axis. None assumes the axis passes through
            the origin.
        """
        axis = np.asarray(axis, dtype=float)
        T = conversions.screw_to_matrix(axis, angle, translation_along_axis, point_on_axis)
        return cls(
            T,
            {
                "kind": "screw",
                "raw_params": {
                    "axis": axis.copy(),
                    "angle": float(angle),
                    "translation_along_axis": float(translation_along_axis),
                    "point_on_axis": None if point_on_axis is None else np.asarray(point_on_axis, dtype=float).copy(),
                },
            },
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        """The underlying (4, 4) float64 array."""
        return self._matrix

    @property
    def input_representation(self) -> dict | None:
        return self._input_representation

    # ── Converters ──────────────────────────────────────────────────────────

    def to_xyz_euler(
        self, convention: str = _EULER_CONVENTION
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyz, euler_angles) — round-trip independent of construction form.

        Returns
        -------
        xyz : np.ndarray, shape (3,)
        euler_angles : np.ndarray, shape (3,) — [ez, ey, ex] in radians
        """
        R = self._matrix[:3, :3]
        xyz = self._matrix[:3, 3].copy()
        euler = conversions.rotation_matrix_to_euler(R, convention=convention)
        return xyz, euler

    def to_quaternion(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (quat_wxyz, xyz) — round-trip independent of construction form.

        Returns
        -------
        quat_wxyz : np.ndarray, shape (4,) — (w, x, y, z)
        xyz : np.ndarray, shape (3,)
        """
        R = self._matrix[:3, :3]
        xyz = self._matrix[:3, 3].copy()
        q = conversions.rotation_matrix_to_quaternion(R)
        return q, xyz

    def to_screw(self) -> dict:
        """Return screw parameters as a dict.

        Returns
        -------
        dict with keys: axis, angle, translation_along_axis, point_on_axis
        """
        return conversions.matrix_to_screw(self._matrix)

    # ── Composition and inverse ─────────────────────────────────────────────

    def compose(self, other: HTM) -> HTM:
        """Return self @ other as a new HTM."""
        return HTM(self._matrix @ other._matrix, {"kind": "composed"})

    def inverse(self) -> HTM:
        """Return the rigid-body inverse without calling np.linalg.inv."""
        R = self._matrix[:3, :3]
        t = self._matrix[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return HTM(T_inv, {"kind": "composed"})

    # ── Equality and display ─────────────────────────────────────────────────

    def is_close(self, other: HTM, atol: float = 1e-9) -> bool:
        """Return True if matrices agree within atol element-wise."""
        return bool(np.allclose(self._matrix, other._matrix, atol=atol))

    def __repr__(self) -> str:
        xyz, euler = self.to_xyz_euler()
        euler_deg = np.degrees(euler)
        return (
            f"HTM(xyz=[{xyz[0]:.4g}, {xyz[1]:.4g}, {xyz[2]:.4g}], "
            f"euler_zyx_deg=[{euler_deg[0]:.4g}, {euler_deg[1]:.4g}, {euler_deg[2]:.4g}])"
        )


# ── Math utilities ───────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """Batched skew-symmetric matrix from a 3-vector or batch of 3-vectors.

    Parameters
    ----------
    v : np.ndarray, shape (..., 3)

    Returns
    -------
    np.ndarray, shape (..., 3, 3)
    """
    v = np.asarray(v, dtype=float)
    *batch, _ = v.shape
    S = np.zeros((*batch, 3, 3))
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] =  v[..., 1]
    S[..., 1, 0] =  v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] =  v[..., 0]
    return S


# ── Internal helpers ────────────────────────────────────────────────────────

def _build_htm(R: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T
