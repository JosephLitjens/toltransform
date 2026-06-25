"""
Tests for postprocess/stats.py (Steps 1-9 of Section 6.8).

All test data is either constructed directly via synthetic TrialData (for unit
checks where exact values are known) or via MonteCarloFKEngine.run (for
behavioral/integration checks like shared-ancestor cancellation).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.frame_graph import DisjointFramesError, FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from postprocess.stats import (
    DOF_LABELS,
    ParetoSensitivityReport,
    compute_tolerance_sensitivities,
    frame_envelope_box,
    frame_histogram_data,
    frame_percentiles,
    point_pair_envelope_box,
    pose_error_vector_batch,
    relative_pose_nominal,
    relative_pose_trials,
)
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData


# -- Helpers ------------------------------------------------------------------

def _make_trial_data(frame_poses: dict, nominal_poses: dict, seed: int = 0) -> TrialData:
    """Construct a TrialData directly without running the engine."""
    n = next(iter(frame_poses.values())).shape[0]
    return TrialData(
        n_trials=n, seed=seed,
        frame_poses=frame_poses,
        nominal_poses=nominal_poses,
        edge_seed_log={},
    )


def _tol(bound: float) -> ToleranceSpec6:
    s = ToleranceSpec("uniform", bound=bound)
    return ToleranceSpec6(s, s, s, s, s, s)


def _zero_tol() -> ToleranceSpec6:
    return _tol(0.0)


def _htm(x=0.0, y=0.0, z=0.0, ez=0.0, ey=0.0, ex=0.0) -> HTM:
    return HTM.from_xyz_euler([x, y, z], [ez, ey, ex])


def _identity_poses(n: int) -> np.ndarray:
    return np.tile(np.eye(4), (n, 1, 1))


# -- 1. pose_error_vector_batch -----------------------------------------------

class TestPoseErrorVectorBatch:
    def test_zero_error_when_poses_equal_nominal(self):
        nominal = _htm(x=1.0, y=2.0, ez=0.1).matrix
        poses = np.tile(nominal, (10, 1, 1))
        errors = pose_error_vector_batch(poses, nominal)
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)

    def test_translation_error_only(self):
        nominal = np.eye(4)
        poses = np.tile(np.eye(4), (3, 1, 1))
        dx_vals = [0.003, -0.001, 0.002]
        for i, dx in enumerate(dx_vals):
            poses[i, 0, 3] = dx
        errors = pose_error_vector_batch(poses, nominal)
        np.testing.assert_allclose(errors[:, 0], dx_vals, atol=1e-12)
        np.testing.assert_allclose(errors[:, 1:], 0.0, atol=1e-12)

    def test_output_shape(self):
        poses = _identity_poses(50)
        nominal = np.eye(4)
        errors = pose_error_vector_batch(poses, nominal)
        assert errors.shape == (50, 6)

    def test_rotation_error_small_angle(self):
        # A small rotation about Z should appear in rz column (index 5).
        rz = 1e-3  # 1 mrad
        nominal = np.eye(4)
        poses = np.tile(np.eye(4), (1, 1, 1))
        c, s = np.cos(rz), np.sin(rz)
        poses[0, :3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        errors = pose_error_vector_batch(poses, nominal)
        # The rotvec for a pure Z rotation of angle rz is [0, 0, rz].
        np.testing.assert_allclose(errors[0, 3:6], [0.0, 0.0, rz], atol=1e-9)


# -- 2. frame_envelope_box ----------------------------------------------------

class TestFrameEnvelopeBox:
    def _make_td(self):
        # Three trials, only x-translation varies.
        poses = _identity_poses(3)
        dx_vals = [0.003, -0.001, 0.002]
        for i, dx in enumerate(dx_vals):
            poses[i, 0, 3] = dx
        return _make_trial_data(
            frame_poses={"root": _identity_poses(3), "B": poses},
            nominal_poses={"root": np.eye(4), "B": np.eye(4)},
        )

    def test_correct_dx_bounds(self):
        td = self._make_td()
        box = frame_envelope_box(td, "B")
        assert box["dx"]["min"] == pytest.approx(-0.001, abs=1e-12)
        assert box["dx"]["max"] == pytest.approx(0.003, abs=1e-12)

    def test_other_dofs_are_zero(self):
        td = self._make_td()
        box = frame_envelope_box(td, "B")
        for label in ["dy", "dz", "rx", "ry", "rz"]:
            assert box[label]["min"] == pytest.approx(0.0, abs=1e-12)
            assert box[label]["max"] == pytest.approx(0.0, abs=1e-12)

    def test_has_all_six_dof_keys(self):
        td = self._make_td()
        box = frame_envelope_box(td, "B")
        assert set(box.keys()) == set(DOF_LABELS)

    def test_root_frame_is_all_zeros(self):
        td = self._make_td()
        box = frame_envelope_box(td, "root")
        for label in DOF_LABELS:
            assert box[label]["min"] == pytest.approx(0.0, abs=1e-12)
            assert box[label]["max"] == pytest.approx(0.0, abs=1e-12)


# -- 3. frame_percentiles -----------------------------------------------------

class TestFramePercentiles:
    def _make_td(self, n: int = 1000):
        # Symmetric uniform dx in [-0.01, +0.01].
        rng = np.random.default_rng(0)
        dx = rng.uniform(-0.01, 0.01, size=n)
        poses = _identity_poses(n)
        poses[:, 0, 3] = dx
        return _make_trial_data(
            frame_poses={"root": _identity_poses(n), "B": poses},
            nominal_poses={"root": np.eye(4), "B": np.eye(4)},
        )

    def test_p0_and_p100_match_min_max(self):
        td = self._make_td()
        pct = frame_percentiles(td, "B", [0.0, 100.0])
        box = frame_envelope_box(td, "B")
        assert pct["dx"][0.0] == pytest.approx(box["dx"]["min"], abs=1e-12)
        assert pct["dx"][100.0] == pytest.approx(box["dx"]["max"], abs=1e-12)

    def test_median_near_zero_for_symmetric_data(self):
        td = self._make_td(n=100_000)
        pct = frame_percentiles(td, "B", [50.0])
        assert pct["dx"][50.0] == pytest.approx(0.0, abs=1e-3)

    def test_has_all_six_dof_keys(self):
        td = self._make_td()
        pct = frame_percentiles(td, "B", [25.0, 75.0])
        assert set(pct.keys()) == set(DOF_LABELS)


# -- 4. frame_histogram_data --------------------------------------------------

class TestFrameHistogramData:
    def _make_td(self, n: int = 500):
        rng = np.random.default_rng(1)
        poses = _identity_poses(n)
        poses[:, 0, 3] = rng.uniform(-0.005, 0.005, size=n)
        return _make_trial_data(
            frame_poses={"root": _identity_poses(n), "B": poses},
            nominal_poses={"root": np.eye(4), "B": np.eye(4)},
        )

    def test_counts_sum_to_n_trials(self):
        td = self._make_td(n=500)
        counts, _ = frame_histogram_data(td, "B", dof_index=0, bins=20)
        assert counts.sum() == 500

    def test_bin_edges_length(self):
        td = self._make_td()
        _, bin_edges = frame_histogram_data(td, "B", dof_index=0, bins=30)
        assert len(bin_edges) == 31

    def test_default_50_bins(self):
        td = self._make_td()
        counts, bin_edges = frame_histogram_data(td, "B", dof_index=0)
        assert len(counts) == 50
        assert len(bin_edges) == 51


# -- 5. relative_pose_trials --------------------------------------------------

class TestRelativePoseTrials:
    def _simple_graph_and_td(self, n: int = 10):
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(x=1.0), _zero_tol())
        td = MonteCarloFKEngine.run(fg, n_trials=n, seed=0)
        return fg, td

    def test_frame_to_itself_is_identity(self):
        fg, td = self._simple_graph_and_td()
        rel = relative_pose_trials(td, fg, "B", "B")
        expected = np.tile(np.eye(4), (10, 1, 1))
        np.testing.assert_allclose(rel, expected, atol=1e-12)

    def test_root_to_itself_is_identity(self):
        fg, td = self._simple_graph_and_td()
        rel = relative_pose_trials(td, fg, "root", "root")
        expected = np.tile(np.eye(4), (10, 1, 1))
        np.testing.assert_allclose(rel, expected, atol=1e-12)

    def test_output_shape(self):
        fg, td = self._simple_graph_and_td(n=25)
        rel = relative_pose_trials(td, fg, "root", "B")
        assert rel.shape == (25, 4, 4)

    def test_disjoint_frames_raise(self):
        fg = FrameGraph()
        for name in ["A", "B", "C", "D"]:
            fg.add_frame(name)
        fg.add_edge("A", "B", _htm(), _zero_tol())
        fg.add_edge("C", "D", _htm(), _zero_tol())
        # Build a minimal TrialData covering all four frames.
        n = 5
        td = _make_trial_data(
            frame_poses={
                "A": _identity_poses(n), "B": _identity_poses(n),
                "C": _identity_poses(n), "D": _identity_poses(n),
            },
            nominal_poses={
                "A": np.eye(4), "B": np.eye(4),
                "C": np.eye(4), "D": np.eye(4),
            },
        )
        with pytest.raises(DisjointFramesError):
            relative_pose_trials(td, fg, "B", "C")


# -- 6. relative_pose_nominal -------------------------------------------------

class TestRelativePoseNominal:
    def test_nominal_relative_matches_composition(self):
        # root -> B (x=1) -> C (y=2). Nominal T_{root->C} = T_{root->B} @ T_{B->C}.
        T_root_B = _htm(x=1.0).matrix
        T_B_C = _htm(y=2.0).matrix
        T_root_C = T_root_B @ T_B_C

        td = _make_trial_data(
            frame_poses={"root": _identity_poses(1), "B": _identity_poses(1), "C": _identity_poses(1)},
            nominal_poses={"root": np.eye(4), "B": T_root_B, "C": T_root_C},
        )
        rel_nom = relative_pose_nominal(td, "root", "C")
        np.testing.assert_allclose(rel_nom, T_root_C, atol=1e-12)

    def test_frame_to_itself_is_identity(self):
        td = _make_trial_data(
            frame_poses={"A": _identity_poses(1)},
            nominal_poses={"A": np.eye(4)},
        )
        np.testing.assert_allclose(relative_pose_nominal(td, "A", "A"), np.eye(4), atol=1e-12)


# -- 7. point_pair_envelope_box -----------------------------------------------

class TestPointPairEnvelopeBox:
    def test_has_all_six_dof_keys(self):
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(), _tol(0.001))
        td = MonteCarloFKEngine.run(fg, n_trials=100, seed=0)
        box = point_pair_envelope_box(td, fg, "root", "B")
        assert set(box.keys()) == set(DOF_LABELS)

    def test_adjacent_edge_envelope_matches_frame_envelope(self):
        # root -> B with nonzero tolerance. With root as the reference frame,
        # point_pair_envelope_box(root, B) should equal frame_envelope_box(B).
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(), _tol(0.005))
        td = MonteCarloFKEngine.run(fg, n_trials=500, seed=7)
        box_pair = point_pair_envelope_box(td, fg, "root", "B")
        box_frame = frame_envelope_box(td, "B")
        for label in DOF_LABELS:
            assert box_pair[label]["min"] == pytest.approx(box_frame[label]["min"], abs=1e-12)
            assert box_pair[label]["max"] == pytest.approx(box_frame[label]["max"], abs=1e-12)

    def test_disjoint_frames_raise(self):
        fg = FrameGraph()
        for name in ["A", "B", "C", "D"]:
            fg.add_frame(name)
        fg.add_edge("A", "B", _htm(), _zero_tol())
        fg.add_edge("C", "D", _htm(), _zero_tol())
        td = MonteCarloFKEngine.run(fg, n_trials=10, seed=0)
        with pytest.raises(DisjointFramesError):
            point_pair_envelope_box(td, fg, "B", "C")


# -- 8. Shared-ancestor cancellation (Section 2.4 / Section 9.1.3 analog) ----

class TestSharedAncestorCancellation:
    """The shared upstream edge must cancel from relative measurements.

    Setup: root -> shared -> leaf1
                   shared -> leaf2
    root->shared has LARGE tolerance (10 mm); downstream edges have SMALL tolerance
    (0.01 mm). The absolute tolerance of leaf1 is dominated by the shared edge (~10 mm).
    But the relative measurement leaf1->leaf2 should only reflect the two small downstream
    tolerances (~0.02 mm total), because the shared noise source cancels.
    """

    def _build_and_run(self, n: int = 10_000):
        fg = FrameGraph()
        for name in ["root", "shared", "leaf1", "leaf2"]:
            fg.add_frame(name)
        fg.add_edge("root", "shared", _htm(), _tol(0.010), name="root->shared")
        fg.add_edge("shared", "leaf1", _htm(), _tol(0.0001), name="shared->leaf1")
        fg.add_edge("shared", "leaf2", _htm(), _tol(0.0001), name="shared->leaf2")
        td = MonteCarloFKEngine.run(fg, n_trials=n, seed=42)
        return fg, td

    def test_absolute_tolerance_dominated_by_shared_edge(self):
        fg, td = self._build_and_run()
        box = frame_envelope_box(td, "leaf1")
        # With 10 mm shared edge and 0.1 mm branch, the absolute envelope
        # in x (and y, z) should be at least 9 mm (not 0.1 mm).
        abs_max = max(abs(box["dx"]["min"]), abs(box["dx"]["max"]))
        assert abs_max > 5e-3, (
            f"Expected absolute envelope > 5 mm (shared edge dominates), got {abs_max*1000:.3f} mm"
        )

    def test_relative_tolerance_tighter_than_absolute(self):
        fg, td = self._build_and_run()
        abs_box = frame_envelope_box(td, "leaf1")
        rel_box = point_pair_envelope_box(td, fg, "leaf1", "leaf2")

        abs_max = max(abs(abs_box["dx"]["min"]), abs(abs_box["dx"]["max"]))
        rel_max = max(abs(rel_box["dx"]["min"]), abs(rel_box["dx"]["max"]))

        assert rel_max < abs_max / 10, (
            f"Relative envelope ({rel_max*1000:.4f} mm) should be <<< absolute "
            f"envelope ({abs_max*1000:.3f} mm) when a large shared-edge tolerance cancels."
        )

    def test_relative_envelope_bounded_by_branch_tolerances(self):
        fg, td = self._build_and_run(n=100_000)
        rel_box = point_pair_envelope_box(td, fg, "leaf1", "leaf2")
        rel_max = max(abs(rel_box["dx"]["min"]), abs(rel_box["dx"]["max"]))
        # With ±0.1 mm branch tolerance on each side, the relative max should be
        # well below 1 mm (the shared 10 mm edge shouldn't appear at all).
        assert rel_max < 1e-3, (
            f"Relative envelope max ({rel_max*1000:.4f} mm) exceeded 1 mm — "
            "shared-edge cancellation may be broken."
        )


# -- TestComputeToleranceSensitivities ----------------------------------------

def _tol_rx(rx_bound: float, distribution: str = "uniform") -> ToleranceSpec6:
    """Helper: ToleranceSpec6 with only rx non-zero."""
    zero = ToleranceSpec("uniform", bound=0.0)
    return ToleranceSpec6(
        dx=zero, dy=zero, dz=zero,
        rx=ToleranceSpec(distribution, bound=rx_bound),
        ry=zero, rz=zero,
    )


def _build_serial_chain(
    rx_bounds: list[float],
    distribution: str = "uniform",
    dz_offset: float = 0.0,
) -> FrameGraph:
    """Build a serial chain world->f0->f1->... where each edge has only rx tolerance."""
    fg = FrameGraph()
    n = len(rx_bounds)
    frame_names = ["world"] + [f"f{i}" for i in range(n)]
    for name in frame_names:
        fg.add_frame(name)
    for i, bound in enumerate(rx_bounds):
        T = HTM.from_xyz_euler([0.0, 0.0, dz_offset], [0.0, 0.0, 0.0])
        fg.add_edge(
            frame_names[i], frame_names[i + 1],
            T,
            _tol_rx(bound, distribution),
            name=f"e{i}",
        )
    return fg


class TestComputeToleranceSensitivities:

    def test_dominant_edge_ranked_first(self):
        """Edge with 10× larger tolerance is ranked first with ~98%+ contribution."""
        # 3 edges: e0 rx=0.010 rad, e1 rx=0.001 rad, e2 rx=0.001 rad (all uniform)
        # Variance ratio: 100 : 1 : 1 → e0 contributes ≈ 98.04%
        fg = _build_serial_chain([0.010, 0.001, 0.001])
        report = compute_tolerance_sensitivities(fg, "world", "f2")
        assert report.ranked_contributions[0][0] == "e0"
        assert report.ranked_contributions[0][1] == "rx"
        assert report.ranked_contributions[0][2] == pytest.approx(100 * 100 / 102, rel=1e-6)

    def test_percentages_sum_to_100(self):
        """Sum of all ranked_contributions percentages must equal 100%."""
        fg = _build_serial_chain([0.005, 0.003, 0.002, 0.001])
        report = compute_tolerance_sensitivities(fg, "world", "f3")
        total = sum(pct for _, _, pct in report.ranked_contributions)
        assert total == pytest.approx(100.0, abs=1e-9)

    def test_uniform_normal_equivalent_variance_same_ranking(self):
        """Uniform(b) and Normal(b/sqrt(3), sigma_level=1) yield equal contributions.

        Both have variance = b²/3, so rankings and percentages must match exactly.
        """
        b = 0.004
        b_normal = b  # sigma = b_normal/sigma_level = b_normal/1; var = b²
        # To match uniform variance b²/3: use b_normal = b/sqrt(3), sigma_level=1
        b_normal_equiv = b / np.sqrt(3)

        zero = ToleranceSpec("uniform", bound=0.0)
        fg_u = FrameGraph()
        fg_n = FrameGraph()
        for fg in [fg_u, fg_n]:
            for name in ["world", "f0", "f1"]:
                fg.add_frame(name)

        def _make_tol6(rx_spec: ToleranceSpec) -> ToleranceSpec6:
            return ToleranceSpec6(dx=zero, dy=zero, dz=zero, rx=rx_spec, ry=zero, rz=zero)

        # Edge 0: dominant, edge 1: weaker — both in uniform and normal variants
        fg_u.add_edge("world", "f0", _htm(), _make_tol6(ToleranceSpec("uniform", bound=0.010)), name="e0")
        fg_u.add_edge("f0", "f1", _htm(), _make_tol6(ToleranceSpec("uniform", bound=0.002)), name="e1")

        fg_n.add_edge("world", "f0", _htm(), _make_tol6(ToleranceSpec("normal", bound=0.010 * np.sqrt(3), sigma_level=1.0)), name="e0")
        fg_n.add_edge("f0", "f1", _htm(), _make_tol6(ToleranceSpec("normal", bound=0.002 * np.sqrt(3), sigma_level=1.0)), name="e1")

        report_u = compute_tolerance_sensitivities(fg_u, "world", "f1")
        report_n = compute_tolerance_sensitivities(fg_n, "world", "f1")

        # Rankings (edge/dof order) must match
        assert [r[:2] for r in report_u.ranked_contributions] == [r[:2] for r in report_n.ranked_contributions]
        # Percentages must match to floating-point precision
        for (_, _, pct_u), (_, _, pct_n) in zip(report_u.ranked_contributions, report_n.ranked_contributions):
            assert pct_u == pytest.approx(pct_n, rel=1e-9)

    def test_zero_tolerance_zero_contribution(self):
        """An edge with all bounds=0 contributes 0% — appears at the bottom ranked at 0.0."""
        fg = FrameGraph()
        for name in ["world", "f0", "f1"]:
            fg.add_frame(name)
        zero = ToleranceSpec("uniform", bound=0.0)
        fg.add_edge("world", "f0", _htm(), _tol(0.005), name="e0")
        fg.add_edge("f0", "f1", _htm(), ToleranceSpec6(
            dx=zero, dy=zero, dz=zero, rx=zero, ry=zero, rz=zero
        ), name="e1_zero")

        report = compute_tolerance_sensitivities(fg, "world", "f1")
        zero_entries = [(n, d, pct) for n, d, pct in report.ranked_contributions if n == "e1_zero"]
        assert all(pct == pytest.approx(0.0, abs=1e-12) for _, _, pct in zero_entries)

    def test_all_zero_tolerances_returns_zero_variance(self):
        """If all bounds are zero, total_variance=0 and all percentages are 0."""
        fg = FrameGraph()
        for name in ["world", "f0"]:
            fg.add_frame(name)
        zero = ToleranceSpec("uniform", bound=0.0)
        fg.add_edge("world", "f0", _htm(), ToleranceSpec6(
            dx=zero, dy=zero, dz=zero, rx=zero, ry=zero, rz=zero
        ), name="e0")
        report = compute_tolerance_sensitivities(fg, "world", "f0")
        assert report.total_variance == 0.0
        assert all(pct == 0.0 for _, _, pct in report.ranked_contributions)

    def test_ascii_chart_returns_nonempty_string(self):
        """to_ascii_chart() smoke test — returns a non-empty string."""
        fg = _build_serial_chain([0.005, 0.003, 0.001])
        report = compute_tolerance_sensitivities(fg, "world", "f2")
        chart = report.to_ascii_chart()
        assert isinstance(chart, str)
        assert len(chart) > 0
        assert "%" in chart
        assert "first-order" in chart.lower()

    def test_ascii_chart_top_n_groups_others(self):
        """top_n=1 shows the single top entry and groups the rest as '(others)'."""
        fg = _build_serial_chain([0.010, 0.005, 0.001])
        report = compute_tolerance_sensitivities(fg, "world", "f2")
        # Only one rx entry is non-zero per edge; with top_n=1 the other two rx
        # contributions (and all zero-bound DoFs) collapse into others.
        chart = report.to_ascii_chart(top_n=1)
        assert "(others)" in chart

    def test_disjoint_frames_raises(self):
        """DisjointFramesError propagates when frame_a and frame_b are disconnected."""
        fg = FrameGraph()
        fg.add_frame("island_a")
        fg.add_frame("island_b")
        with pytest.raises(DisjointFramesError):
            compute_tolerance_sensitivities(fg, "island_a", "island_b")
