"""
Tests for sim/monte_carlo_fk.py.

Covers: 2-edge and 3-edge chain hand-checks (via fixed-delta helper), shared-edge
consistency (Section 9 Item 3), reproducibility and RNG independence, root-anchor,
and miscellaneous field checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6, apply_perturbation_batch
from core.transforms import HTM
from helpers import _FixedToleranceSpec6
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zero_tol() -> ToleranceSpec6:
    z = ToleranceSpec("uniform", bound=0.0)
    return ToleranceSpec6(z, z, z, z, z, z)


def _uniform_tol(bound: float = 0.001) -> ToleranceSpec6:
    s = ToleranceSpec("uniform", bound=bound)
    return ToleranceSpec6(s, s, s, s, s, s)


def _htm(x=0.0, y=0.0, z=0.0, ez=0.0, ey=0.0, ex=0.0) -> HTM:
    return HTM.from_xyz_euler([x, y, z], [ez, ey, ex])


# ── 1. 2-edge chain hand-check ────────────────────────────────────────────────

class TestTwoEdgeChainHandCheck:
    """Pure-translation 2-edge chain with fixed known deltas."""

    def _build(self):
        fg = FrameGraph()
        for name in ["root", "B", "C"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(), _FixedToleranceSpec6([0.01, 0, 0, 0, 0, 0]))
        fg.add_edge("B", "C", _htm(), _FixedToleranceSpec6([0, 0.02, 0, 0, 0, 0]))
        return fg

    def test_frame_b_translation(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=5, seed=0)
        for i in range(5):
            np.testing.assert_allclose(
                td.frame_poses["B"][i, :3, 3], [0.01, 0.0, 0.0], atol=1e-12
            )

    def test_frame_c_translation(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=5, seed=0)
        for i in range(5):
            np.testing.assert_allclose(
                td.frame_poses["C"][i, :3, 3], [0.01, 0.02, 0.0], atol=1e-12
            )

    def test_output_shapes(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=10, seed=0)
        assert td.frame_poses["root"].shape == (10, 4, 4)
        assert td.frame_poses["B"].shape == (10, 4, 4)
        assert td.frame_poses["C"].shape == (10, 4, 4)


# ── 2. 3-edge chain hand-check ────────────────────────────────────────────────

class TestThreeEdgeChainHandCheck:
    """3-edge chain with a nonzero nominal rotation on the middle edge."""

    def test_composition_through_nominal_rotation(self):
        fg = FrameGraph()
        for name in ["root", "B", "C", "D"]:
            fg.add_frame(name)

        # Edge 0: root → B, identity nominal, delta = pure dx=0.01
        fg.add_edge("root", "B", _htm(), _FixedToleranceSpec6([0.01, 0, 0, 0, 0, 0]))
        # Edge 1: B → C, nominal = pure Z rotation π/4, delta = pure dy=0.005
        nominal_bc = _htm(ez=np.pi / 4)
        fg.add_edge("B", "C", nominal_bc, _FixedToleranceSpec6([0, 0.005, 0, 0, 0, 0]))
        # Edge 2: C → D, identity nominal, zero delta
        fg.add_edge("C", "D", _htm(), _zero_tol())

        td = MonteCarloFKEngine.run(fg, n_trials=3, seed=0)

        # Hand-compute expected D pose for trial 0:
        # T_root_B = T_nominal_rootB @ T_delta_rootB
        delta_rootB = np.eye(4)
        delta_rootB[:3, 3] = [0.01, 0, 0]
        T_rootB = np.eye(4) @ delta_rootB  # nominal=I

        # T_B_C = T_nominal_BC @ T_delta_BC
        delta_BC = np.eye(4)
        delta_BC[:3, 3] = [0, 0.005, 0]
        T_BC = nominal_bc.matrix @ delta_BC

        # T_C_D = T_nominal_CD @ T_delta_CD = I @ I = I
        T_CD = np.eye(4)

        T_expected = T_rootB @ T_BC @ T_CD
        np.testing.assert_allclose(td.frame_poses["D"][0], T_expected, atol=1e-12)

    def test_nominal_poses_chain(self):
        """nominal_poses should be the product of nominal transforms, ignoring deltas."""
        fg = FrameGraph()
        for name in ["root", "B", "C"]:
            fg.add_frame(name)
        nom1 = _htm(x=1.0)
        nom2 = _htm(y=2.0)
        fg.add_edge("root", "B", nom1, _zero_tol())
        fg.add_edge("B", "C", nom2, _zero_tol())

        td = MonteCarloFKEngine.run(fg, n_trials=5, seed=0)

        expected_C_nominal = nom1.matrix @ nom2.matrix
        np.testing.assert_allclose(td.nominal_poses["C"], expected_C_nominal, atol=1e-12)


# ── 3. Shared-edge consistency (Section 9, Item 3) ────────────────────────────

class TestSharedEdgeConsistency:
    """The shared upstream edge must be sampled exactly once per trial.

    Setup: root → shared → leaf1
                shared → leaf2
    The root→shared edge has real tolerance; downstream edges have zero tolerance.
    With zero-bound downstream deltas, leaf1 and leaf2 must have the same
    pose as shared for every trial — proving the shared edge's perturbation
    propagated identically to both branches.
    """

    def _build(self):
        fg = FrameGraph()
        for name in ["root", "shared", "leaf1", "leaf2"]:
            fg.add_frame(name)
        fg.add_edge("root", "shared", _htm(), _uniform_tol(0.01), name="root->shared")
        fg.add_edge("shared", "leaf1", _htm(), _zero_tol(), name="shared->leaf1")
        fg.add_edge("shared", "leaf2", _htm(), _zero_tol(), name="shared->leaf2")
        return fg

    def test_leaf1_matches_shared_for_every_trial(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=500, seed=7)
        np.testing.assert_array_equal(
            td.frame_poses["leaf1"], td.frame_poses["shared"]
        )

    def test_leaf2_matches_shared_for_every_trial(self):
        td = MonteCarloFKEngine.run(self._build(), n_trials=500, seed=7)
        np.testing.assert_array_equal(
            td.frame_poses["leaf2"], td.frame_poses["shared"]
        )

    def test_shared_poses_differ_across_trials(self):
        """Sanity-check: the shared edge is actually being perturbed, not trivially zero."""
        td = MonteCarloFKEngine.run(self._build(), n_trials=500, seed=7)
        # With nonzero tolerance the poses should not all be identical.
        assert not np.all(td.frame_poses["shared"] == td.frame_poses["shared"][0])


# ── 4. Reproducibility and RNG independence ───────────────────────────────────

class TestReproducibility:
    def _simple_chain(self) -> FrameGraph:
        fg = FrameGraph()
        for name in ["root", "B", "C"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(), _uniform_tol(0.005))
        fg.add_edge("B", "C", _htm(), _uniform_tol(0.01))
        return fg

    def test_same_seed_same_output(self):
        fg = self._simple_chain()
        td1 = MonteCarloFKEngine.run(fg, n_trials=200, seed=42)
        td2 = MonteCarloFKEngine.run(fg, n_trials=200, seed=42)
        for frame in ["root", "B", "C"]:
            assert np.array_equal(td1.frame_poses[frame], td2.frame_poses[frame]), (
                f"frame_poses['{frame}'] was not bit-for-bit identical across two runs "
                "with the same seed."
            )

    def test_unrelated_edge_does_not_change_existing_samples(self):
        """Adding a disconnected component must not alter other edges' samples."""
        fg_original = self._simple_chain()
        td_original = MonteCarloFKEngine.run(fg_original, n_trials=200, seed=42)

        # Add a completely disconnected component (D → E).
        fg_extended = self._simple_chain()
        fg_extended.add_frame("D")
        fg_extended.add_frame("E")
        fg_extended.add_edge("D", "E", _htm(), _uniform_tol(0.02))
        td_extended = MonteCarloFKEngine.run(fg_extended, n_trials=200, seed=42)

        for frame in ["root", "B", "C"]:
            assert np.array_equal(
                td_original.frame_poses[frame],
                td_extended.frame_poses[frame],
            ), (
                f"Adding an unrelated edge changed frame_poses['{frame}'] — "
                "per-edge RNG keying is broken."
            )

    def test_different_seeds_give_different_output(self):
        fg = self._simple_chain()
        td1 = MonteCarloFKEngine.run(fg, n_trials=200, seed=42)
        td2 = MonteCarloFKEngine.run(fg, n_trials=200, seed=99)
        # Vanishingly unlikely to be equal across 200 trials.
        assert not np.array_equal(td1.frame_poses["C"], td2.frame_poses["C"])


# ── 5. Root-anchor test ───────────────────────────────────────────────────────

class TestRootAnchor:
    def test_root_frame_is_identity_every_trial(self):
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(), _uniform_tol(0.01))
        td = MonteCarloFKEngine.run(fg, n_trials=100, seed=0)

        expected = np.tile(np.eye(4), (100, 1, 1))
        np.testing.assert_array_equal(td.frame_poses["root"], expected)

    def test_multi_component_both_roots_are_identity(self):
        fg = FrameGraph()
        for name in ["R1", "A", "R2", "B"]:
            fg.add_frame(name)
        fg.add_edge("R1", "A", _htm(), _uniform_tol(0.01))
        fg.add_edge("R2", "B", _htm(), _uniform_tol(0.01))
        td = MonteCarloFKEngine.run(fg, n_trials=50, seed=0)

        expected = np.tile(np.eye(4), (50, 1, 1))
        np.testing.assert_array_equal(td.frame_poses["R1"], expected)
        np.testing.assert_array_equal(td.frame_poses["R2"], expected)


# ── 6. Miscellaneous field checks ─────────────────────────────────────────────

class TestTrialDataFields:
    def _run(self, n=20, seed=5) -> TrialData:
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        fg.add_edge("root", "B", _htm(), _uniform_tol(), name="root->B")
        return MonteCarloFKEngine.run(fg, n_trials=n, seed=seed)

    def test_n_trials_matches(self):
        td = self._run(n=33, seed=0)
        assert td.n_trials == 33

    def test_seed_matches(self):
        td = self._run(n=10, seed=17)
        assert td.seed == 17

    def test_edge_seed_log_populated(self):
        td = self._run()
        assert "root->B" in td.edge_seed_log
        assert isinstance(td.edge_seed_log["root->B"], int)

    def test_zero_delta_child_equals_nominal(self):
        fg = FrameGraph()
        for name in ["root", "B"]:
            fg.add_frame(name)
        nominal = _htm(x=1.0, y=2.0, ez=0.3)
        fg.add_edge("root", "B", nominal, _FixedToleranceSpec6([0, 0, 0, 0, 0, 0]))
        td = MonteCarloFKEngine.run(fg, n_trials=5, seed=0)
        for i in range(5):
            np.testing.assert_allclose(td.frame_poses["B"][i], nominal.matrix, atol=1e-12)


# ── 7. Validation precondition ────────────────────────────────────────────────

class TestValidationPrecondition:
    def test_cyclic_graph_raises_before_sampling(self):
        """validate_dag() must be called before any sampling occurs."""
        fg = FrameGraph()
        for name in ["A", "B", "C"]:
            fg.add_frame(name)
        fg.add_edge("A", "B", _htm(), _zero_tol())
        fg.add_edge("B", "C", _htm(), _zero_tol())
        # Force a cycle directly on the NetworkX graph (bypassing add_edge checks).
        fg._g.add_edge("C", "A")
        with pytest.raises(ValueError, match="Cycle"):
            MonteCarloFKEngine.run(fg, n_trials=10, seed=0)
