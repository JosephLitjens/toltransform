"""
Post-processing statistics for Monte Carlo FK results.

Rotation-vector convention (locked, ω = θu):
    All rotation errors are represented as rotation vectors where each row is the
    rotation axis unit vector u scaled by the rotation angle θ (in radians).
    Extracted via scipy.spatial.transform.Rotation.from_matrix(R_error).as_rotvec(),
    which computes the exact matrix logarithm — equivalent to the skew-symmetric
    extraction at small angles, and more robust at moderately larger angles.
    The columns [3:6] of pose_error_vector_batch output are exactly these rotvecs,
    directly compatible with postprocess/bounding_shapes.py's fit_rotation_cone /
    fit_rotation_box input contract (no further conversion needed).

Scope:
    This module implements all Steps 1-9 of Section 6.8.
    Steps 1-7: basic forward-stats functions (frame_envelope_box, percentiles, etc.)
    Steps 8-9: Pareto sensitivity breakdown — ParetoSensitivityReport dataclass and
               compute_tolerance_sensitivities() (added in B1-3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from core.frame_graph import FrameGraph, compute_sensitivity
from sim.monte_carlo_fk import TrialData

# DoF label order -- must match the [dx,dy,dz,rx,ry,rz] convention throughout the codebase.
DOF_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz"]


# -- Public data types --------------------------------------------------------

@dataclass
class ParetoSensitivityReport:
    """Pareto-sorted first-order tolerance contribution breakdown.

    ranked_contributions is a list of (edge_name, dof_label, percentage) tuples,
    sorted descending by percentage. Percentages across all entries sum to ≈100.

    NOTE: Contributions are first-order linear approximations computed from the
    small-angle adjoint Jacobian. For high-leverage chains (long lever arms,
    large rotational offsets), actual nonlinear contributions can differ.
    """
    ranked_contributions: list[tuple[str, str, float]]  # (edge_name, dof_label, pct)
    total_variance: float                               # sum of all raw contributions

    def to_ascii_chart(self, top_n: int = 10) -> str:
        """Text bar chart of the top-N contributors; remainder grouped as '(others)'."""
        lines = ["Pareto Sensitivity Breakdown (first-order approximation)"]
        lines.append("─" * 65)
        shown = self.ranked_contributions[:top_n]
        others_pct = sum(pct for _, _, pct in self.ranked_contributions[top_n:])
        max_pct = shown[0][2] if shown else 0.0
        bar_width = 28
        for edge_name, dof, pct in shown:
            label = f"{edge_name} ({dof})"
            filled = int(round(pct / max_pct * bar_width)) if max_pct > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  {label:<36s}  {bar}  {pct:5.1f}%")
        if others_pct > 0.0:
            lines.append(f"  {'(others)':<36s}  {'░' * bar_width}  {others_pct:5.1f}%")
        lines.append("─" * 65)
        lines.append("  NOTE: first-order approximation — see class docstring.")
        lines.append(f"  Total variance: {self.total_variance:.3e}")
        return "\n".join(lines)


# -- Private helpers ----------------------------------------------------------

def _htm_inverse_batch(poses: np.ndarray) -> np.ndarray:
    """Closed-form batch HTM inverse for (N,4,4) rigid-transform arrays.

    Exploits the known structure (R,t) rather than np.linalg.inv, matching
    the semantics of HTM.inverse() from core/transforms.py:
        R_inv = R.T,  t_inv = -R.T @ t
    """
    R = poses[:, :3, :3]            # (N,3,3)
    t = poses[:, :3, 3]             # (N,3)
    R_inv = R.transpose(0, 2, 1)    # (N,3,3)
    T_inv = np.zeros_like(poses)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = -np.einsum("nij,nj->ni", R_inv, t)
    T_inv[:, 3, 3] = 1.0
    return T_inv


def _htm_inverse_single(pose: np.ndarray) -> np.ndarray:
    """Closed-form HTM inverse for a single (4,4) rigid-transform array."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# -- Core error-extraction primitive ------------------------------------------

def pose_error_vector_batch(
    poses: np.ndarray,
    nominal: np.ndarray,
) -> np.ndarray:
    """Extract 6-DoF error vectors from a batch of pose matrices.

    Parameters
    ----------
    poses : np.ndarray, shape (N,4,4)
        Perturbed poses (from TrialData.frame_poses or relative_pose_trials).
    nominal : np.ndarray, shape (4,4)
        The unperturbed reference pose for this frame (from TrialData.nominal_poses
        or relative_pose_nominal).

    Returns
    -------
    np.ndarray, shape (N,6)
        Columns [0:3] = translation error [dx,dy,dz].
        Columns [3:6] = rotation error as rotvec omega = theta*u (axis-scaled-by-angle),
            in radians -- compatible with postprocess/bounding_shapes.py's
            fit_rotation_cone / fit_rotation_box input contract.
    """
    # Translation error: straightforward subtraction.
    dt = poses[:, :3, 3] - nominal[:3, 3]  # (N,3)

    # Rotation error: R_error[i] = R_nominal.T @ R_perturbed[i], then log map.
    R_nominal = nominal[:3, :3]                                           # (3,3)
    R_perturbed = poses[:, :3, :3]                                        # (N,3,3)
    R_error = np.einsum("ji,njk->nik", R_nominal, R_perturbed)            # (N,3,3)
    dr = Rotation.from_matrix(R_error).as_rotvec()                        # (N,3)

    return np.hstack([dt, dr])  # (N,6)


# -- Per-frame envelope / distribution functions ------------------------------

def frame_envelope_box(
    trial_data: TrialData,
    frame_name: str,
) -> dict[str, dict[str, float]]:
    """Per-DoF worst-case bounding box for a single frame's error.

    Parameters
    ----------
    trial_data : TrialData
    frame_name : str

    Returns
    -------
    dict
        Keys are DoF labels ["dx","dy","dz","rx","ry","rz"]; values are
        {"min": float, "max": float} dicts (error relative to nominal).
    """
    errors = pose_error_vector_batch(
        trial_data.frame_poses[frame_name],
        trial_data.nominal_poses[frame_name],
    )
    return {
        label: {"min": float(errors[:, i].min()), "max": float(errors[:, i].max())}
        for i, label in enumerate(DOF_LABELS)
    }


def frame_percentiles(
    trial_data: TrialData,
    frame_name: str,
    percentiles: list[float],
) -> dict[str, dict[float, float]]:
    """Per-DoF percentile table for a single frame's error distribution.

    Parameters
    ----------
    trial_data : TrialData
    frame_name : str
    percentiles : list[float]
        Percentile values in [0, 100], e.g. [0.1, 2.5, 50.0, 97.5, 99.9].

    Returns
    -------
    dict
        Keys are DoF labels; values are {percentile_value: float} dicts.
    """
    errors = pose_error_vector_batch(
        trial_data.frame_poses[frame_name],
        trial_data.nominal_poses[frame_name],
    )
    pct_values = np.percentile(errors, percentiles, axis=0)  # (len(pcts), 6)
    return {
        label: {float(p): float(pct_values[j, i]) for j, p in enumerate(percentiles)}
        for i, label in enumerate(DOF_LABELS)
    }


def frame_histogram_data(
    trial_data: TrialData,
    frame_name: str,
    dof_index: int,
    bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram data for one DoF of a single frame's error distribution.

    Parameters
    ----------
    trial_data : TrialData
    frame_name : str
    dof_index : int
        Index into [dx,dy,dz,rx,ry,rz] (0-5).
    bins : int
        Number of histogram bins (default 50).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (counts, bin_edges) -- directly from np.histogram.
        len(counts) == bins, len(bin_edges) == bins + 1.
    """
    errors = pose_error_vector_batch(
        trial_data.frame_poses[frame_name],
        trial_data.nominal_poses[frame_name],
    )
    return np.histogram(errors[:, dof_index], bins=bins)


# -- Point-pair relative-pose functions ---------------------------------------

def relative_pose_trials(
    trial_data: TrialData,
    frame_graph: FrameGraph,
    frame_a: str,
    frame_b: str,
) -> np.ndarray:
    """Relative transform T_{a->b} for every Monte Carlo trial.

    Exploits the pre-stored absolute poses in TrialData -- no re-simulation needed.
    Validates that frame_a and frame_b are in the same weakly-connected component;
    raises DisjointFramesError with the locked message (Section 2.3.1) if not.

    Parameters
    ----------
    trial_data : TrialData
    frame_graph : FrameGraph
        Used only for same-component validation.
    frame_a, frame_b : str

    Returns
    -------
    np.ndarray, shape (N,4,4)
        inv(frame_poses[frame_a][i]) @ frame_poses[frame_b][i] for each trial i.
    """
    frame_graph._assert_same_component(frame_a, frame_b)
    T_a_inv = _htm_inverse_batch(trial_data.frame_poses[frame_a])  # (N,4,4)
    T_b = trial_data.frame_poses[frame_b]                          # (N,4,4)
    return np.einsum("nij,njk->nik", T_a_inv, T_b)                # (N,4,4)


def relative_pose_nominal(
    trial_data: TrialData,
    frame_a: str,
    frame_b: str,
) -> np.ndarray:
    """Nominal (unperturbed) relative transform T_{a->b}.

    Used as the reference point for error-vector extraction in point_pair_envelope_box.

    Returns
    -------
    np.ndarray, shape (4,4)
    """
    T_a_nom_inv = _htm_inverse_single(trial_data.nominal_poses[frame_a])  # (4,4)
    return T_a_nom_inv @ trial_data.nominal_poses[frame_b]                 # (4,4)


def point_pair_envelope_box(
    trial_data: TrialData,
    frame_graph: FrameGraph,
    frame_a: str,
    frame_b: str,
) -> dict[str, dict[str, float]]:
    """Per-DoF worst-case bounding box for the relative pose between two frames.

    The primary function for cross-chain tolerance analysis (Section 3.3): e.g.,
    camera-to-target relative alignment tolerance in a shared-base optical system.
    Because TrialData stores absolute poses per trial, this requires no
    re-simulation -- the relative transform is computed directly from stored data.

    Parameters
    ----------
    trial_data : TrialData
    frame_graph : FrameGraph
        Used only for same-component validation (raises DisjointFramesError if not).
    frame_a, frame_b : str

    Returns
    -------
    dict
        Same structure as frame_envelope_box: keys are DoF labels, values are
        {"min": float, "max": float}.
    """
    rel_poses = relative_pose_trials(trial_data, frame_graph, frame_a, frame_b)
    rel_nominal = relative_pose_nominal(trial_data, frame_a, frame_b)
    errors = pose_error_vector_batch(rel_poses, rel_nominal)
    return {
        label: {"min": float(errors[:, i].min()), "max": float(errors[:, i].max())}
        for i, label in enumerate(DOF_LABELS)
    }


# -- Pareto sensitivity breakdown ---------------------------------------------

def compute_tolerance_sensitivities(
    frame_graph: FrameGraph,
    frame_a: str,
    frame_b: str,
) -> ParetoSensitivityReport:
    """First-order Pareto breakdown of tolerance contributions between two frames.

    Uses the adjoint Jacobian from compute_sensitivity() to propagate each edge's
    variance to the 6-DoF output at frame_b (relative to frame_a), then ranks
    edge/DoF pairs by their percentage of total output variance.

    Variance is computed via ToleranceSpec.variance, which handles both symmetric
    (±bound) and asymmetric (lower/upper) specs correctly, including non-zero mean
    contributions for off-centre intervals. Locked DoFs with bound=0 contribute zero.

    Raises DisjointFramesError (propagated from path_edges_between) if frame_a
    and frame_b are in different connected components.

    Parameters
    ----------
    frame_graph : FrameGraph
    frame_a, frame_b : str
        The analysis endpoints. Typically frame_a is the chain root.

    Returns
    -------
    ParetoSensitivityReport
        Entries ranked descending by percentage contribution to total output variance.
    """
    path = frame_graph.path_edges_between(frame_a, frame_b)
    edges = [edge for edge, _ in path]
    edge_names = [e.name for e in edges]

    J = compute_sensitivity(frame_graph, frame_a, frame_b, edge_names)  # (6, 6*n)

    raw_contributions: list[tuple[str, str, float]] = []
    for i, edge in enumerate(edges):
        for j, label in enumerate(DOF_LABELS):
            spec = edge.tolerance[j]
            var_j = spec.variance
            J_col = J[:, i * 6 + j]                          # shape (6,)
            raw_contributions.append(
                (edge.name, label, float(np.dot(J_col, J_col) * var_j))
            )

    total_variance = sum(c[2] for c in raw_contributions)

    if total_variance <= 0.0:
        return ParetoSensitivityReport(
            ranked_contributions=[(n, d, 0.0) for n, d, _ in raw_contributions],
            total_variance=0.0,
        )

    ranked = sorted(
        [(n, d, 100.0 * raw / total_variance) for n, d, raw in raw_contributions],
        key=lambda x: x[2],
        reverse=True,
    )
    return ParetoSensitivityReport(ranked_contributions=ranked, total_variance=total_variance)
