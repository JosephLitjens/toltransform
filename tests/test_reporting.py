"""
tests/test_reporting.py — Smoke tests for postprocess/reporting.py.

These tests verify that each public function runs without raising an exception
and returns the correct Matplotlib type (Axes or Figure). They do NOT assert on
pixel content or visual layout — rendering correctness is verified by human review
of the example scripts.

matplotlib.use("Agg") is set at import time to prevent any GUI display during CI.
All tests close figures after assertion to avoid memory leaks.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede other mpl imports

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from postprocess.bounding_shapes import (
    fit_bounding_box,
    fit_bounding_ellipsoid,
    fit_bounding_sphere,
    fit_rotation_box,
    fit_rotation_cone,
)
from postprocess.reporting import (
    generate_frame_report,
    generate_sensitivity_report,
    plot_histogram,
    plot_pareto_sensitivity,
    plot_rotation_summary,
    plot_translation_projection,
)
from postprocess.stats import ParetoSensitivityReport, compute_tolerance_sensitivities
from sim.monte_carlo_fk import MonteCarloFKEngine


# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_trial_data(n: int = 500, seed: int = 0):
    """Build a minimal 2-edge chain and run MC for smoke-test use."""
    fg = FrameGraph()
    for name in ["world", "mid", "tip"]:
        fg.add_frame(name)
    zero = ToleranceSpec("uniform", bound=0.0)
    tol = ToleranceSpec6(
        dx=ToleranceSpec("uniform", bound=0.050),
        dy=ToleranceSpec("uniform", bound=0.050),
        dz=ToleranceSpec("uniform", bound=0.020),
        rx=ToleranceSpec("normal",  bound=0.001),
        ry=ToleranceSpec("normal",  bound=0.001),
        rz=ToleranceSpec("uniform", bound=0.0005),
    )
    fg.add_edge("world", "mid", HTM.from_xyz_euler([0,0,100],[0,0,0]), tol, name="e0")
    fg.add_edge("mid",  "tip", HTM.from_xyz_euler([0,0, 50],[0,0,0]), tol, name="e1")
    return MonteCarloFKEngine.run(fg, n_trials=n, seed=seed), fg


def _make_pareto_report() -> ParetoSensitivityReport:
    """Minimal ParetoSensitivityReport with 3 entries."""
    return ParetoSensitivityReport(
        ranked_contributions=[
            ("edge_A", "rx", 65.0),
            ("edge_B", "ry", 25.0),
            ("edge_C", "dz", 10.0),
        ],
        total_variance=1.23e-6,
    )


# ── TestPlotHistogram ─────────────────────────────────────────────────────────

class TestPlotHistogram:
    def test_returns_axes(self):
        counts = np.array([1, 5, 10, 5, 1], dtype=float)
        bin_edges = np.linspace(-1, 1, 6)
        ax = plot_histogram(counts, bin_edges, "dx")
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_uses_provided_ax(self):
        _, provided_ax = plt.subplots()
        counts = np.array([2, 8, 4], dtype=float)
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        returned_ax = plot_histogram(counts, bin_edges, "ry", ax=provided_ax)
        assert returned_ax is provided_ax
        plt.close("all")


# ── TestPlotTranslationProjection ─────────────────────────────────────────────

class TestPlotTranslationProjection:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.points = rng.standard_normal((200, 3)) * 0.05

    def test_returns_axes_box_all_planes(self):
        bbox = fit_bounding_box(self.points)
        for plane in ("xy", "xz", "yz"):
            ax = plot_translation_projection(self.points, bbox, plane)
            assert isinstance(ax, Axes), f"Expected Axes for plane={plane}"
            plt.close("all")

    def test_returns_axes_sphere(self):
        sphere = fit_bounding_sphere(self.points)
        ax = plot_translation_projection(self.points, sphere, "xy")
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_returns_axes_ellipsoid(self):
        ellipsoid = fit_bounding_ellipsoid(self.points, coverage=1.0)
        ax = plot_translation_projection(self.points, ellipsoid, "xy")
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_returns_axes_statistical_ellipsoid(self):
        ellipsoid = fit_bounding_ellipsoid(self.points, coverage=0.997)
        ax = plot_translation_projection(self.points, ellipsoid, "yz")
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_subsamples_large_cloud(self):
        """N=5000 should trigger subsampling without crashing."""
        rng = np.random.default_rng(1)
        large_pts = rng.standard_normal((5000, 3)) * 0.05
        bbox = fit_bounding_box(large_pts)
        ax = plot_translation_projection(large_pts, bbox, "xy")
        assert isinstance(ax, Axes)
        plt.close("all")


# ── TestPlotRotationSummary ───────────────────────────────────────────────────

class TestPlotRotationSummary:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.rotvecs = rng.standard_normal((300, 3)) * 0.001

    def test_returns_axes(self):
        cone = fit_rotation_cone(self.rotvecs)
        box = fit_rotation_box(self.rotvecs)
        ax = plot_rotation_summary(self.rotvecs, cone, box)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_zero_rotvecs_no_crash(self):
        """All-zero rotvecs: cone max_angle=0 — no crash or division errors."""
        zero_rvs = np.zeros((50, 3))
        cone = fit_rotation_cone(zero_rvs)
        box = fit_rotation_box(zero_rvs)
        ax = plot_rotation_summary(zero_rvs, cone, box)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_large_cloud_subsampled(self):
        rng = np.random.default_rng(2)
        large_rvs = rng.standard_normal((5000, 3)) * 0.001
        cone = fit_rotation_cone(large_rvs)
        box = fit_rotation_box(large_rvs)
        ax = plot_rotation_summary(large_rvs, cone, box)
        assert isinstance(ax, Axes)
        plt.close("all")


# ── TestPlotParetoSensitivity ─────────────────────────────────────────────────

class TestPlotParetoSensitivity:
    def test_returns_axes(self):
        report = _make_pareto_report()
        ax = plot_pareto_sensitivity(report)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_top_n_grouping(self):
        """top_n=1 with 3 entries — others bar must appear without crash."""
        report = _make_pareto_report()
        ax = plot_pareto_sensitivity(report, top_n=1)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_uses_provided_ax(self):
        _, provided_ax = plt.subplots()
        report = _make_pareto_report()
        returned = plot_pareto_sensitivity(report, ax=provided_ax)
        assert returned is provided_ax
        plt.close("all")

    def test_real_sensitivity_report(self):
        """Use compute_tolerance_sensitivities output directly."""
        fg = FrameGraph()
        for name in ["world", "f0", "f1"]:
            fg.add_frame(name)
        zero = ToleranceSpec("uniform", bound=0.0)
        tol = ToleranceSpec6(
            dx=ToleranceSpec("uniform", bound=0.010),
            dy=zero, dz=zero,
            rx=ToleranceSpec("normal", bound=0.002),
            ry=zero, rz=zero,
        )
        fg.add_edge("world", "f0", HTM.from_xyz_euler([0,0,50],[0,0,0]), tol, name="e0")
        fg.add_edge("f0",   "f1", HTM.from_xyz_euler([0,0,50],[0,0,0]), tol, name="e1")
        report = compute_tolerance_sensitivities(fg, "world", "f1")
        ax = plot_pareto_sensitivity(report)
        assert isinstance(ax, Axes)
        plt.close("all")


# ── TestGenerateFrameReport ───────────────────────────────────────────────────

class TestGenerateFrameReport:
    def test_returns_figure(self):
        td, _ = _make_trial_data()
        fig = generate_frame_report(td, "tip")
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_returns_figure_for_intermediate_frame(self):
        """generate_frame_report works on any frame, not just the leaf."""
        td, _ = _make_trial_data()
        fig = generate_frame_report(td, "mid")
        assert isinstance(fig, Figure)
        plt.close("all")


# ── TestGenerateSensitivityReport ─────────────────────────────────────────────

class TestGenerateSensitivityReport:
    def test_returns_figure(self):
        report = _make_pareto_report()
        fig = generate_sensitivity_report(report)
        assert isinstance(fig, Figure)
        plt.close("all")
