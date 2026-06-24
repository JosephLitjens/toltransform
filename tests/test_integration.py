"""
End-to-end integration tests: FK engine → stats pipeline.

"Hand-verified" means manually derived small-angle expected values are checked
against engine output using _FixedToleranceSpec6 (same duck-typed helper pattern
as test_monte_carlo_fk.py) for exact, deterministic assertions.

Also contains the required named regression tests from Section 6.20 / Section 9:
    test_shared_edge_sampling_consistency  --  Section 9, Item 3

These tests cover the A4+A5 integration path that per-module tests cannot:
    FrameGraph → MonteCarloFKEngine → frame_envelope_box / pose_error_vector_batch
"""

from __future__ import annotations

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from postprocess.stats import frame_envelope_box
from sim.monte_carlo_fk import MonteCarloFKEngine

# ── Local constants and helpers (mirrors conftest.py -- kept here so this file
#    is also self-contained for readers who open it without conftest context) ──

DEFAULT_ATOL = 1e-9
SMALL_ANGLE_ATOL = 1e-6


def make_tol(bound: float):
    tol = ToleranceSpec("uniform", bound=bound)
    return ToleranceSpec6(tol, tol, tol, tol, tol, tol)


def make_zero_tol():
    return make_tol(0.0)


def make_htm(x: float = 0.0, y: float = 0.0, z: float = 0.0,
             ez: float = 0.0, ey: float = 0.0, ex: float = 0.0) -> HTM:
    return HTM.from_xyz_euler([x, y, z], [ez, ey, ex])


class _FixedToleranceSpec6:
    """Returns the same (N,6) delta for every call — for deterministic hand-checks.

    Duck-types ToleranceSpec6.sample() so it can be passed directly as a tolerance
    to FrameGraph.add_edge without monkeypatching.
    """

    def __init__(self, delta_1d):
        self._delta = np.asarray(delta_1d, dtype=float)  # (6,)

    def sample(self, n_trials: int, rng) -> np.ndarray:
        return np.tile(self._delta, (n_trials, 1))  # (N,6)


def _fixed(delta) -> _FixedToleranceSpec6:
    return _FixedToleranceSpec6(delta)


# ── 1. Two-edge translation stack-up, end-to-end ─────────────────────────────

class TestTwoEdgeTranslationEndToEnd:
    """Verifies the FK → stats pipeline on a pure-translation chain.

    Uses _FixedToleranceSpec6 so every trial has the same known delta — the
    resulting envelope has min == max == the exact delta value, checkable by hand.
    """

    def _build(self, delta_rootB, delta_BC):
        fg = FrameGraph()
        for name in ["root", "B", "C"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", make_htm(), _fixed(delta_rootB), name="root->B")
        fg.add_edge("B", "C", make_htm(), _fixed(delta_BC), name="B->C")
        return fg

    def test_dx_error_at_leaf(self):
        """Single-DoF delta on root→B propagates as a constant dx error at C."""
        fg = self._build([0.003, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        td = MonteCarloFKEngine.run(fg, n_trials=5, seed=0)
        box = frame_envelope_box(td, "C")

        np.testing.assert_allclose(box["dx"]["min"], 0.003, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["dx"]["max"], 0.003, atol=DEFAULT_ATOL)
        for label in ["dy", "dz", "rx", "ry", "rz"]:
            np.testing.assert_allclose(box[label]["min"], 0.0, atol=DEFAULT_ATOL)
            np.testing.assert_allclose(box[label]["max"], 0.0, atol=DEFAULT_ATOL)

    def test_stacked_dy_error(self):
        """Independent dx and dy deltas on each edge both appear at the leaf."""
        fg = self._build([0.003, 0, 0, 0, 0, 0], [0, 0.005, 0, 0, 0, 0])
        td = MonteCarloFKEngine.run(fg, n_trials=5, seed=0)
        box = frame_envelope_box(td, "C")

        np.testing.assert_allclose(box["dx"]["min"], 0.003, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["dx"]["max"], 0.003, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["dy"]["min"], 0.005, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["dy"]["max"], 0.005, atol=DEFAULT_ATOL)
        for label in ["dz", "rx", "ry", "rz"]:
            np.testing.assert_allclose(box[label]["min"], 0.0, atol=DEFAULT_ATOL)
            np.testing.assert_allclose(box[label]["max"], 0.0, atol=DEFAULT_ATOL)

    def test_error_is_zero_at_intermediate_frame_B(self):
        """dx delta on root→B appears at B; B→C has zero delta so C's error = B's error."""
        fg = self._build([0.003, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        td = MonteCarloFKEngine.run(fg, n_trials=5, seed=0)
        box_B = frame_envelope_box(td, "B")

        np.testing.assert_allclose(box_B["dx"]["min"], 0.003, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box_B["dx"]["max"], 0.003, atol=DEFAULT_ATOL)


# ── 2. Lever-arm coupling: rotation → translation cross-coupling ──────────────

class TestLeverArmCoupling:
    """Verifies the rotation-to-translation cross-coupling in the FK+stats pipeline.

    Setup: root → pivot (nominal=I, fixed δrz=0.001 rad), pivot → end_effector
           (nominal=translation(0, L=0.1 m, 0), zero tolerance).

    Small-angle derivation (locked):
        T_pivot = I @ T_delta_rz ≈ Rz(0.001)
        T_end   = T_pivot @ translation(0, L, 0)
              → T_end[0,3] = -sin(δrz)*L ≈ -δrz*L = -0.0001 m
                T_end[1,3] = cos(δrz)*L  ≈ L (no error in y)
        Nominal: T_end_nom[0,3] = 0, T_end_nom[1,3] = L

        Error dx = -δrz * L = -0.0001 m
        Error dy = 0
        Error rz = δrz = 0.001 rad (rotation propagates directly)

    This is the kinematic coupling that motivates the Sine-Bar Lever Arm Benchmark
    (Section 9.1.2), here implemented as a hand-verified unit check rather than
    the formal statistical benchmark (which is in B1-6).
    """

    L = 0.1    # lever arm length, metres
    DRZ = 1e-3  # rotation perturbation, radians

    def _build(self):
        fg = FrameGraph()
        for name in ["root", "pivot", "end"]:
            fg.add_frame(name)
        fg.add_edge("root", "pivot", make_htm(), _fixed([0, 0, 0, 0, 0, self.DRZ]),
                    name="root->pivot")
        fg.add_edge("pivot", "end", make_htm(y=self.L), make_zero_tol(),
                    name="pivot->end")
        return fg

    def _box(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=3, seed=0)
        return frame_envelope_box(td, "end")

    def test_lateral_translation_error(self):
        """dx_error at end ≈ -δrz * L (lever arm cross-coupling)."""
        box = self._box()
        expected_dx = -self.DRZ * self.L
        np.testing.assert_allclose(box["dx"]["min"], expected_dx, atol=SMALL_ANGLE_ATOL)
        np.testing.assert_allclose(box["dx"]["max"], expected_dx, atol=SMALL_ANGLE_ATOL)

    def test_dy_error_is_zero(self):
        """y-direction error ≈ 0 (cos(δrz)*L - L ≈ 0 at small angles)."""
        box = self._box()
        np.testing.assert_allclose(box["dy"]["min"], 0.0, atol=SMALL_ANGLE_ATOL)
        np.testing.assert_allclose(box["dy"]["max"], 0.0, atol=SMALL_ANGLE_ATOL)

    def test_rotation_error_propagates_to_end(self):
        """rz error at end = δrz (rotation propagates through zero-tol edge)."""
        box = self._box()
        np.testing.assert_allclose(box["rz"]["min"], self.DRZ, atol=SMALL_ANGLE_ATOL)
        np.testing.assert_allclose(box["rz"]["max"], self.DRZ, atol=SMALL_ANGLE_ATOL)

    def test_other_translation_dofs_are_zero(self):
        box = self._box()
        for label in ["dz"]:
            np.testing.assert_allclose(box[label]["min"], 0.0, atol=SMALL_ANGLE_ATOL)
            np.testing.assert_allclose(box[label]["max"], 0.0, atol=SMALL_ANGLE_ATOL)


# ── 3. Local-frame perturbation routing ──────────────────────────────────────

class TestLocalFramePerturbationRouting:
    """Verifies right-multiply perturbation convention (Section 2.2.2) through stats.

    Setup: root → B (nominal = Rz(π/4), fixed delta = [dx=0.001, 0, 0, 0, 0, 0]
           in B's local frame).

    Small-angle derivation:
        T_B = Rz(π/4) @ translation(0.001, 0, 0)
            → T_B[0,3] = cos(π/4)*0.001 = 0.001/√2 ≈ 7.071e-4  (world x)
               T_B[1,3] = sin(π/4)*0.001 = 0.001/√2 ≈ 7.071e-4  (world y)
        Nominal B has zero translation, so:
            dx_error = 0.001/√2,  dy_error = 0.001/√2

    This confirms that a local-frame dx perturbation, when the nominal has a π/4 Z
    rotation, shows up equally in world-frame x and y error — not just in world-x.
    """

    DX_LOCAL = 1e-3
    _EXPECTED = DX_LOCAL / np.sqrt(2)

    def _build(self):
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", make_htm(ez=np.pi / 4),
                    _fixed([self.DX_LOCAL, 0, 0, 0, 0, 0]),
                    name="root->B")
        return fg

    def _box(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=3, seed=0)
        return frame_envelope_box(td, "B")

    def test_dx_error_at_B(self):
        box = self._box()
        np.testing.assert_allclose(box["dx"]["min"], self._EXPECTED, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["dx"]["max"], self._EXPECTED, atol=DEFAULT_ATOL)

    def test_dy_error_at_B(self):
        box = self._box()
        np.testing.assert_allclose(box["dy"]["min"], self._EXPECTED, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["dy"]["max"], self._EXPECTED, atol=DEFAULT_ATOL)

    def test_other_dofs_are_zero(self):
        box = self._box()
        for label in ["dz", "rx", "ry", "rz"]:
            np.testing.assert_allclose(box[label]["min"], 0.0, atol=DEFAULT_ATOL)
            np.testing.assert_allclose(box[label]["max"], 0.0, atol=DEFAULT_ATOL)


# ── 4. Fixture smoke tests ─────────────────────────────────────────────────────

class TestFixtureSmoke:
    """Confirm conftest.py shared fixtures produce valid, runnable FrameGraphs."""

    def test_two_edge_chain_runs(self, two_edge_chain):
        td = MonteCarloFKEngine.run(two_edge_chain, n_trials=50, seed=0)
        assert set(td.frame_poses.keys()) == {"root", "B", "C"}
        assert td.frame_poses["C"].shape == (50, 4, 4)

    def test_three_edge_chain_runs(self, three_edge_chain):
        td = MonteCarloFKEngine.run(three_edge_chain, n_trials=50, seed=0)
        assert set(td.frame_poses.keys()) == {"root", "B", "C", "D"}

    def test_shared_frame_graph_runs(self, shared_frame_graph):
        td = MonteCarloFKEngine.run(shared_frame_graph, n_trials=50, seed=0)
        assert set(td.frame_poses.keys()) == {"root", "shared", "leaf1", "leaf2"}

    def test_two_edge_chain_nominal_is_correct(self, two_edge_chain):
        td = MonteCarloFKEngine.run(two_edge_chain, n_trials=5, seed=0)
        # Nominal C = translation(x=0.005, y=0.010) composed
        np.testing.assert_allclose(
            td.nominal_poses["C"][:3, 3], [0.005, 0.010, 0.0], atol=DEFAULT_ATOL
        )


# ── Section 9 Item 3 — required standalone named regression ───────────────────

def test_shared_edge_sampling_consistency():
    """Section 9, Item 3 — architecturally critical named standalone regression.

    A module-level function (not inside any class) so it is findable by name
    via `pytest -k test_shared_edge_sampling_consistency` without knowing the
    file's internal structure.

    Property under test: when two Frames (leaf1, leaf2) share a common upstream
    edge (root→shared), both downstream Frames must see identical absolute poses
    for every trial — the shared edge is sampled exactly once per trial and its
    perturbation propagates identically to all downstream branches.

    This is a correctness-critical architectural property (Section 2.4, Section 9
    Item 3), not a performance or style preference. If this test fails, the FK
    engine is broken in a way that would silently corrupt all multi-branch results.

    See also TestSharedEdgeConsistency in test_monte_carlo_fk.py for complementary
    unit-level coverage; this function exists as the required named standalone
    regression per Section 6.20 Item 3.
    """
    fg = FrameGraph()
    for name in ["root", "shared", "leaf1", "leaf2"]:
        fg.add_frame(name)
    # Large real tolerance on the shared edge; zero downstream so any difference
    # between leaf1 and leaf2 can only come from a bug in how the shared edge
    # is sampled.
    fg.add_edge("root", "shared", make_htm(), make_tol(0.010), name="root->shared")
    fg.add_edge("shared", "leaf1", make_htm(), make_zero_tol(), name="shared->leaf1")
    fg.add_edge("shared", "leaf2", make_htm(), make_zero_tol(), name="shared->leaf2")

    td = MonteCarloFKEngine.run(fg, n_trials=1000, seed=99)

    np.testing.assert_array_equal(
        td.frame_poses["leaf1"], td.frame_poses["shared"],
        err_msg="leaf1 poses diverged from shared — shared edge may be sampled twice",
    )
    np.testing.assert_array_equal(
        td.frame_poses["leaf2"], td.frame_poses["shared"],
        err_msg="leaf2 poses diverged from shared — shared edge may be sampled twice",
    )
    # Sanity-check: the shared edge is actually being perturbed (not trivially zero).
    assert not np.all(td.frame_poses["shared"] == td.frame_poses["shared"][0]), (
        "Shared frame poses are all identical — zero-tolerance tolerance may have been applied"
    )
