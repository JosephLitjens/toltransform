"""
postprocess/reporting.py — Matplotlib plotting layer for TolTransform.

Converts error-vector arrays, fitted bounding shapes, and Pareto sensitivity
reports into renderable Matplotlib objects. All functions return Axes or Figure
objects — callers (scripts, examples, GUI widgets) own the .show() / .savefig()
decision; these functions never call either.

Public API (in implementation order):
    plot_histogram               — single-DoF error histogram
    plot_translation_projection  — 2D scatter + bounding shape overlay
    plot_rotation_summary        — rx–ry scatter with cone (primary) + box (secondary)
    plot_pareto_sensitivity       — horizontal Pareto bar chart with caveat annotation
    generate_frame_report        — multi-panel figure for one frame (10 panels)
    generate_sensitivity_report  — single-panel figure wrapping plot_pareto_sensitivity
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse, Rectangle

from postprocess.bounding_shapes import fit_bounding_box, fit_rotation_box, fit_rotation_cone
from postprocess.stats import (
    DOF_LABELS,
    ParetoSensitivityReport,
    frame_histogram_data,
    pose_error_vector_batch,
)
from sim.monte_carlo_fk import TrialData

# Plane → (axis_i, axis_j) index mapping for translation projection plots.
_PLANE_AXES: dict[str, tuple[int, int]] = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_PLANE_LABELS: dict[str, tuple[str, str]] = {
    "xy": ("dx", "dy"),
    "xz": ("dx", "dz"),
    "yz": ("dy", "dz"),
}

# Points above this threshold are subsampled before scattering (legibility + speed).
_SCATTER_MAX_POINTS = 2000


def _maybe_subsample(points: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """Return points subsampled to _SCATTER_MAX_POINTS rows if needed."""
    n = len(points)
    if n <= _SCATTER_MAX_POINTS:
        return points
    idx = np.random.default_rng(rng_seed).choice(n, _SCATTER_MAX_POINTS, replace=False)
    return points[idx]


def _ensure_ax(ax: Axes | None) -> Axes:
    """Create and return a fresh Axes if none is provided."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


# ── 1. plot_histogram ─────────────────────────────────────────────────────────

def plot_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    dof_label: str,
    ax: Axes | None = None,
) -> Axes:
    """Single-DoF error histogram from frame_histogram_data() output.

    Parameters
    ----------
    counts    : np.ndarray, shape (n_bins,)
    bin_edges : np.ndarray, shape (n_bins+1,)
    dof_label : str — axis label and title suffix, e.g. "dx"
    ax        : optional Axes — created if not provided

    Returns
    -------
    Axes
    """
    ax = _ensure_ax(ax)
    ax.stairs(counts, bin_edges, fill=True, alpha=0.75)
    ax.set_xlabel(dof_label)
    ax.set_ylabel("count")
    ax.set_title(f"Error distribution: {dof_label}")
    return ax


# ── 2. plot_translation_projection ───────────────────────────────────────────

def plot_translation_projection(
    points: np.ndarray,
    bounding_shape: dict,
    plane: Literal["xy", "xz", "yz"],
    ax: Axes | None = None,
) -> Axes:
    """2D scatter of translation errors on one coordinate plane with shape overlay.

    Parameters
    ----------
    points        : np.ndarray, shape (N,3) — translation error cloud [dx,dy,dz]
    bounding_shape: dict — output of fit_bounding_box / fit_bounding_sphere /
                           fit_bounding_ellipsoid; type inferred from dict keys
    plane         : "xy", "xz", or "yz"
    ax            : optional Axes

    Returns
    -------
    Axes
    """
    ax = _ensure_ax(ax)
    ai, aj = _PLANE_AXES[plane]
    xlabel, ylabel = _PLANE_LABELS[plane]

    pts = _maybe_subsample(points)
    ax.scatter(pts[:, ai], pts[:, aj], s=2, alpha=0.3, color="steelblue", linewidths=0)

    # Detect bounding shape type and render overlay.
    if "axes_lengths" in bounding_shape and "axes_directions" in bounding_shape:
        # Ellipsoid: project 3D covariance onto the 2D plane
        center = bounding_shape["center"]
        D = np.diag(bounding_shape["axes_lengths"] ** 2)
        V = bounding_shape["axes_directions"]         # columns = principal axes
        C_3d = V @ D @ V.T
        C_2d = C_3d[np.ix_([ai, aj], [ai, aj])]
        evals, evecs = np.linalg.eigh(C_2d)           # ascending eigenvalues
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        patch = Ellipse(
            xy=(center[ai], center[aj]),
            width=2.0 * np.sqrt(max(evals[1], 0.0)),
            height=2.0 * np.sqrt(max(evals[0], 0.0)),
            angle=angle,
            fill=False,
            edgecolor="tomato",
            linewidth=1.5,
            label="bounding ellipsoid",
        )
        ax.add_patch(patch)

    elif "radius" in bounding_shape:
        # Sphere: circle at projected center with the sphere radius
        center = bounding_shape["center"]
        radius = bounding_shape["radius"]
        patch = Circle(
            (center[ai], center[aj]),
            radius=radius,
            fill=False,
            edgecolor="tomato",
            linewidth=1.5,
            label="bounding sphere",
        )
        ax.add_patch(patch)

    else:
        # Box: axis-aligned rectangle
        lo = bounding_shape["min"]
        hi = bounding_shape["max"]
        patch = Rectangle(
            xy=(lo[ai], lo[aj]),
            width=hi[ai] - lo[ai],
            height=hi[aj] - lo[aj],
            fill=False,
            edgecolor="tomato",
            linewidth=1.5,
            label="bounding box",
        )
        ax.add_patch(patch)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Translation projection: {plane}")
    ax.autoscale_view()
    ax.legend(fontsize=7, loc="upper right")
    return ax


# ── 3. plot_rotation_summary ─────────────────────────────────────────────────

def plot_rotation_summary(
    rotvecs: np.ndarray,
    cone: dict,
    box: dict,
    ax: Axes | None = None,
) -> Axes:
    """2D rx–ry scatter with cone (primary) and per-axis box (secondary) overlay.

    Visualizes the rotation-error point cloud in the rx–ry plane.
    The cone (max_angle circle + mean_axis arrow) is the lead visual element;
    the per-axis box is a thin secondary annotation. rz is not shown in 2D.

    Parameters
    ----------
    rotvecs : np.ndarray, shape (N,3), ω=θu convention
    cone    : dict with "max_angle" (float) and "mean_axis" (array(3))
    box     : dict with "min" (array(3)) and "max" (array(3))
    ax      : optional Axes

    Returns
    -------
    Axes
    """
    ax = _ensure_ax(ax)

    pts = _maybe_subsample(rotvecs)
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.3, color="steelblue", linewidths=0,
               label="trials (rx, ry)")

    max_angle = cone["max_angle"]
    mean_axis = cone["mean_axis"]

    # Cone — primary: shaded circle at max_angle radius
    cone_circle = Circle(
        (0.0, 0.0),
        radius=max_angle,
        fill=True,
        facecolor="orange",
        edgecolor="darkorange",
        alpha=0.15,
        linewidth=2,
        label=f"cone (max|ω|={max_angle:.4f} rad)",
        zorder=1,
    )
    ax.add_patch(cone_circle)

    # Mean axis arrow — only meaningful when max_angle > 0
    if max_angle > 1e-12:
        arrow_end_x = mean_axis[0] * max_angle
        arrow_end_y = mean_axis[1] * max_angle
        ax.annotate(
            "mean axis",
            xy=(arrow_end_x, arrow_end_y),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
            fontsize=7,
            color="darkorange",
        )

    # Box — secondary: thin dashed rectangle on rx–ry axes
    bx_lo = box["min"]
    bx_hi = box["max"]
    box_rect = Rectangle(
        xy=(bx_lo[0], bx_lo[1]),
        width=bx_hi[0] - bx_lo[0],
        height=bx_hi[1] - bx_lo[1],
        fill=False,
        edgecolor="gray",
        linestyle="--",
        linewidth=0.8,
        label="per-axis box (rx, ry)",
        zorder=2,
    )
    ax.add_patch(box_rect)

    ax.set_xlabel("rx (rad)")
    ax.set_ylabel("ry (rad)")
    ax.set_title("Rotation error (rx–ry plane)")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.text(0.98, 0.98, "rz not shown", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="gray")
    ax.legend(fontsize=7, loc="lower right")
    return ax


# ── 4. plot_pareto_sensitivity ────────────────────────────────────────────────

def plot_pareto_sensitivity(
    report: ParetoSensitivityReport,
    ax: Axes | None = None,
    top_n: int = 10,
) -> Axes:
    """Horizontal Pareto bar chart of tolerance contributions.

    Top-N edge/DoF pairs are shown individually; the remainder is grouped as a
    single "(others)" bar. Includes a mandatory first-order-approximation caveat
    annotation (see Section 6.8 Step 8 — this chart is likely to be shared
    standalone in sourcing discussions where the caveat must be visible).

    Parameters
    ----------
    report : ParetoSensitivityReport from compute_tolerance_sensitivities()
    ax     : optional Axes
    top_n  : int — number of individual entries to show (default 10)

    Returns
    -------
    Axes
    """
    ax = _ensure_ax(ax)

    shown = report.ranked_contributions[:top_n]
    tail = report.ranked_contributions[top_n:]
    others_pct = sum(pct for _, _, pct in tail)

    labels = [f"{edge} ({dof})" for edge, dof, _ in shown]
    values = [pct for _, _, pct in shown]

    if others_pct > 0.0:
        labels.append("(others)")
        values.append(others_pct)

    # Reverse so highest bar is at the top (barh draws bottom-to-top)
    labels = labels[::-1]
    values = values[::-1]

    bars = ax.barh(labels, values, color="steelblue", alpha=0.8)
    ax.set_xlabel("% of total output variance")
    ax.set_xlim(0, max(100, max(values) * 1.1))
    ax.set_title("Pareto Sensitivity Breakdown")

    # 80% Pareto line (visual reference)
    ax.axvline(x=80, linestyle=":", linewidth=0.8, color="gray", alpha=0.7, label="80%")
    ax.legend(fontsize=7, loc="lower right")

    # Percentage labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=7,
        )

    # Mandatory first-order-approximation caveat (locked — must be visible on chart)
    ax.annotate(
        "* First-order linear approximation via small-angle adjoint Jacobian.\n"
        "  Nonlinear contributions may differ for high-leverage chains.",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=7,
        color="gray",
        va="bottom",
    )

    ax.set_xlabel("% of total output variance")
    return ax


# ── 5. generate_frame_report ─────────────────────────────────────────────────

def generate_frame_report(trial_data: TrialData, frame_name: str) -> Figure:
    """Multi-panel figure for a complete per-frame error breakdown.

    Layout (4 rows × 3 cols):
        Row 0: hist(dx) | hist(dy) | hist(dz)
        Row 1: proj(xy) | proj(xz) | proj(yz)
        Row 2: hist(rx) | hist(ry) | hist(rz)
        Row 3: rotation summary (spans all 3 columns)

    Parameters
    ----------
    trial_data : TrialData
    frame_name : str

    Returns
    -------
    Figure
    """
    errors = pose_error_vector_batch(
        trial_data.frame_poses[frame_name],
        trial_data.nominal_poses[frame_name],
    )  # (N,6)
    trans_pts = errors[:, :3]   # (N,3) — dx,dy,dz
    rot_pts = errors[:, 3:]     # (N,3) — rx,ry,rz (rotvecs, ω=θu)

    bbox = fit_bounding_box(trans_pts)
    cone = fit_rotation_cone(rot_pts)
    rotbox = fit_rotation_box(rot_pts)

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: translation histograms
    for col, (dof_idx, label) in enumerate(zip(range(3), DOF_LABELS[:3])):
        ax = fig.add_subplot(gs[0, col])
        counts, bin_edges = frame_histogram_data(trial_data, frame_name, dof_idx)
        plot_histogram(counts, bin_edges, label, ax=ax)

    # Row 1: translation projections
    for col, plane in enumerate(("xy", "xz", "yz")):
        ax = fig.add_subplot(gs[1, col])
        plot_translation_projection(trans_pts, bbox, plane, ax=ax)

    # Row 2: rotation histograms
    for col, (dof_idx, label) in enumerate(zip(range(3, 6), DOF_LABELS[3:])):
        ax = fig.add_subplot(gs[2, col])
        counts, bin_edges = frame_histogram_data(trial_data, frame_name, dof_idx)
        plot_histogram(counts, bin_edges, label, ax=ax)

    # Row 3: rotation summary (spans all columns)
    ax_rot = fig.add_subplot(gs[3, :])
    plot_rotation_summary(rot_pts, cone, rotbox, ax=ax_rot)

    fig.suptitle(f"Frame error report: {frame_name}", fontsize=14, fontweight="bold", y=1.01)
    return fig


# ── 6. generate_sensitivity_report ───────────────────────────────────────────

def generate_sensitivity_report(report: ParetoSensitivityReport) -> Figure:
    """Standalone single-panel figure for a ParetoSensitivityReport.

    Kept separate from generate_frame_report because sensitivity is per-point-pair
    target (frame_a → frame_b), not per-frame absolute error.

    Parameters
    ----------
    report : ParetoSensitivityReport from compute_tolerance_sensitivities()

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pareto_sensitivity(report, ax=ax)
    fig.tight_layout()
    return fig
