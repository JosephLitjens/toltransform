"""
postprocess/bounding_shapes.py — Bounding-shape fitting for error-vector point clouds.

Fits geometrically interpretable shapes to the translation and rotation error-vector
point clouds produced by postprocess/stats.py:

  Translation clouds (N,3):
    fit_bounding_box        — axis-aligned min/max box (trivial)
    fit_bounding_sphere     — centroid + max-distance sphere (conservative, always-correct)
    fit_bounding_ellipsoid  — PCA ellipsoid, worst-case (coverage=1.0) or statistical (0<coverage<1.0)

  Rotation clouds (N,3):
    fit_rotation_cone  — PRIMARY: single worst-case half-angle + mean tilt axis
    fit_rotation_box   — secondary: per-axis min/max bounds (same arithmetic as fit_bounding_box)

Rotation-vector convention (LOCKED):
  Both fit_rotation_cone and fit_rotation_box accept only (N,3) arrays whose rows are
  the rotation vector ω = θu — axis u scaled by angle θ (radians). This is exactly
  columns [3:6] of the (N,6) output of pose_error_vector_batch(). Raw (N,4,4) pose
  arrays are explicitly rejected at runtime to prevent coordinate-coupling artifacts.

Locked decisions (Section 8 / Section 6.9, design_spec.md):
  - Cone is the lead rotation representation (max_angle + mean_axis).
  - Per-axis box is secondary — available for direction-sensitive analysis.
  - fit_bounding_sphere uses the conservative centroid+max-distance approach, not Welzl's.
  - fit_bounding_ellipsoid uses scipy.stats.chi2 for statistical (coverage<1.0) scaling.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


_DEGENERATE_AXIS_TOL = 1e-10   # rotation-vector magnitude below which an axis is treated as zero
_DEGENERATE_SIGMA_TOL = 1e-15  # ellipsoid axis length below which sigma is treated as degenerate


# ── private helpers ───────────────────────────────────────────────────────────

def _check_rotvec_shape(arr: np.ndarray, fname: str) -> None:
    """Raise ValueError if arr is not shape (N,3). Blocks raw (N,4,4) pose arrays."""
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{fname}: rotvecs must be shape (N,3) with convention ω=θu "
            f"(axis scaled by angle, radians); got shape {arr.shape}. "
            "Do not pass raw (N,4,4) pose arrays — extract rotation vectors first via "
            "postprocess.stats.pose_error_vector_batch(poses, nominal)[..., 3:6]."
        )


def _check_points_shape(arr: np.ndarray, fname: str) -> None:
    """Raise ValueError if arr is not shape (N,3)."""
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{fname}: points must be shape (N,3); got shape {arr.shape}."
        )


# ── translation bounding shapes ───────────────────────────────────────────────

def fit_bounding_box(points: np.ndarray) -> dict:
    """Axis-aligned bounding box for an (N,3) point cloud.

    Parameters
    ----------
    points : np.ndarray, shape (N,3)
        3D point cloud (translation errors in mm, or rotvec components in rad).

    Returns
    -------
    dict with keys:
        "min" : np.ndarray(3) — per-axis minimum
        "max" : np.ndarray(3) — per-axis maximum
    """
    _check_points_shape(points, "fit_bounding_box")
    return {
        "min": np.min(points, axis=0),
        "max": np.max(points, axis=0),
    }


def fit_bounding_sphere(points: np.ndarray) -> dict:
    """Conservative bounding sphere for an (N,3) point cloud.

    Uses the centroid + max-distance approach: center at the sample mean,
    radius = maximum distance from any point to the center. This always encloses
    all points (conservative) but is not the minimum-enclosing sphere — the radius
    may be up to 2× larger than optimal. For an error-budgeting tool, a slightly
    loose but always-correct bound is preferred over the complexity of Welzl's
    algorithm (locked decision, Section 6.9).

    Parameters
    ----------
    points : np.ndarray, shape (N,3)

    Returns
    -------
    dict with keys:
        "center" : np.ndarray(3) — centroid of the point cloud
        "radius" : float         — maximum distance from center to any point
    """
    _check_points_shape(points, "fit_bounding_sphere")
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    return {"center": center, "radius": radius}


def fit_bounding_ellipsoid(points: np.ndarray, coverage: float = 1.0) -> dict:
    """PCA-aligned bounding ellipsoid for an (N,3) point cloud.

    Two distinct modes answer two distinct engineering questions:

    coverage=1.0 (worst-case ellipsoid):
        Projects all points onto the PCA principal axes and sets each axis length
        to enclose the furthest point along that axis. Guarantees 100% enclosure
        by construction. Use when the question is "what is the absolute worst-case
        bounding shape?"

    0 < coverage < 1.0 (statistical ellipsoid):
        Scales the PCA axes using the chi-squared distribution with 3 degrees of
        freedom: axes_lengths[i] = sqrt(eigenvalue[i]) * sqrt(chi2.ppf(coverage, df=3)).
        For example, coverage=0.997 gives the 3σ-equivalent ellipsoid (the surface
        that encloses ~99.7% of draws from a matched trivariate normal). Use when
        the question is "what shape bounds p% of all outcomes statistically?"

    Do not confuse the two modes — they are different objects. The worst-case
    ellipsoid depends on the actual sample extremes; the statistical ellipsoid
    depends only on the sample covariance and the chosen coverage level.

    Parameters
    ----------
    points   : np.ndarray, shape (N,3)
    coverage : float in (0, 1] — 1.0 for worst-case, <1.0 for statistical

    Returns
    -------
    dict with keys:
        "center"          : np.ndarray(3)   — centroid
        "axes_lengths"    : np.ndarray(3)   — half-lengths along each principal axis,
                                              sorted descending
        "axes_directions" : np.ndarray(3,3) — columns are principal axis unit vectors,
                                              matching the axes_lengths order
    """
    _check_points_shape(points, "fit_bounding_ellipsoid")
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1]; got {coverage}.")

    center = np.mean(points, axis=0)
    X = points - center                         # (N,3) centred
    C = (X.T @ X) / len(points)                # sample covariance (biased)

    eigenvalues, eigenvectors = np.linalg.eigh(C)  # eigenvalues ascending

    # Sort descending so axes_lengths[0] is the longest axis
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]         # columns = principal axes

    if coverage >= 1.0:
        projections = X @ eigenvectors          # (N,3) — coordinates in PCA basis
        sigma = np.sqrt(np.maximum(eigenvalues, 0.0))
        # Scale the covariance-shape ellipsoid uniformly so all points are inside.
        # Ellipsoidal coordinate r²_i = Σ_j (proj[i,j]/σ_j)²; we need r_max ≤ 1.
        # Use σ_safe=1 for degenerate axes (σ≈0 ↔ proj≈0, so they contribute ~0).
        sigma_safe = np.where(sigma > _DEGENERATE_SIGMA_TOL, sigma, 1.0)
        r_sq = np.sum((projections / sigma_safe[np.newaxis, :]) ** 2, axis=1)
        uniform_scale = float(np.sqrt(np.max(r_sq))) if len(points) > 1 else 0.0
        axes_lengths = sigma * uniform_scale
    else:
        scale = np.sqrt(chi2.ppf(coverage, df=3))
        axes_lengths = np.sqrt(np.maximum(eigenvalues, 0.0)) * scale

    return {
        "center": center,
        "axes_lengths": axes_lengths,
        "axes_directions": eigenvectors,
    }


# ── rotation bounding shapes ──────────────────────────────────────────────────

def fit_rotation_cone(rotvecs: np.ndarray) -> dict:
    """Bounding cone for a rotation-vector point cloud — PRIMARY rotation representation.

    Fits a single worst-case half-angle (max_angle) and a mean tilt axis to the
    rotation-error distribution. This is the number an engineer reads first when
    asking "how far off-axis could this interface tip?"

    Input convention (LOCKED — do not change):
        rotvecs must be shape (N,3) with rows ω = θu, where u is the unit rotation
        axis and θ is the small rotation angle in radians. This is exactly columns
        [3:6] of pose_error_vector_batch()'s (N,6) output. Raw (N,4,4) pose arrays
        are explicitly rejected — passing one raises ValueError.

    Parameters
    ----------
    rotvecs : np.ndarray, shape (N,3)
        Rotation-vector point cloud (ω = θu convention).

    Returns
    -------
    dict with keys:
        "max_angle"  : float          — worst-case rotation magnitude (radians);
                                        half-angle of the bounding cone
        "mean_axis"  : np.ndarray(3)  — unit vector of the mean tilt direction
                                        (informational; [0,0,1] by convention when
                                        all angles are effectively zero)
    """
    _check_rotvec_shape(rotvecs, "fit_rotation_cone")

    angles = np.linalg.norm(rotvecs, axis=1)    # θ for each trial, shape (N,)
    max_angle = float(np.max(angles))

    nonzero_mask = angles > _DEGENERATE_AXIS_TOL
    if not np.any(nonzero_mask):
        mean_axis = np.array([0.0, 0.0, 1.0])
    else:
        unit_axes = rotvecs[nonzero_mask] / angles[nonzero_mask, np.newaxis]
        mean_dir = np.mean(unit_axes, axis=0)
        norm = np.linalg.norm(mean_dir)
        mean_axis = mean_dir / norm if norm > _DEGENERATE_AXIS_TOL else np.array([0.0, 0.0, 1.0])

    return {"max_angle": max_angle, "mean_axis": mean_axis}


def fit_rotation_box(rotvecs: np.ndarray) -> dict:
    """Per-axis bounding box for a rotation-vector point cloud — secondary representation.

    Produces three independent worst-case angle bounds (one per axis of the rotvec),
    useful when the direction of angular error matters and a single isotropic cone
    half-angle would obscure asymmetry (e.g., pitch is much more sensitive than yaw
    for a given optical system).

    The cone (fit_rotation_cone) is the lead representation; use this as a
    supplementary cross-check or when per-axis budget breakdowns are needed.

    Input convention (LOCKED — identical to fit_rotation_cone):
        rotvecs must be shape (N,3), rows ω = θu. Raw (N,4,4) arrays are rejected.

    Parameters
    ----------
    rotvecs : np.ndarray, shape (N,3)

    Returns
    -------
    dict with keys:
        "min" : np.ndarray(3) — per-axis minimum angle component (radians)
        "max" : np.ndarray(3) — per-axis maximum angle component (radians)
    """
    _check_rotvec_shape(rotvecs, "fit_rotation_box")
    return fit_bounding_box(rotvecs)
