"""
Tests for core/frame_graph.py.

Covers: DAG validation, topology queries, path composition, DisjointFramesError
(exact locked message), common-base pattern, and sensitivity cross-check.
"""

import numpy as np
import pytest

from core.frame_graph import (
    DisjointFramesError,
    Frame,
    FrameGraph,
    HTMEdge,
    adjoint,
    compute_sensitivity,
)
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM


# ── Shared helpers ────────────────────────────────────────────────────────────

def _zero_tol() -> ToleranceSpec6:
    z = ToleranceSpec("uniform", bound=0.0)
    return ToleranceSpec6(z, z, z, z, z, z)


def _htm(x=0.0, y=0.0, z=0.0, ez=0.0, ey=0.0, ex=0.0) -> HTM:
    return HTM.from_xyz_euler([x, y, z], [ez, ey, ex])


def _simple_chain(n_edges: int) -> FrameGraph:
    """Build root -> F1 -> ... -> Fn with pure X translations of 1.0 each."""
    fg = FrameGraph()
    names = ["root"] + [f"F{i}" for i in range(1, n_edges + 1)]
    for name in names:
        fg.add_frame(name)
    for parent, child in zip(names[:-1], names[1:]):
        fg.add_edge(parent, child, _htm(x=1.0), _zero_tol())
    return fg


# ── 1. Cycle detection ────────────────────────────────────────────────────────

class TestCycleDetection:
    def test_cycle_raises(self):
        fg = FrameGraph()
        for n in ["A", "B", "C"]:
            fg.add_frame(n)
        fg.add_edge("A", "B", _htm(), _zero_tol())
        fg.add_edge("B", "C", _htm(), _zero_tol())
        # Force a cycle — bypass add_edge validation by adding directly to the nx graph.
        fg._g.add_edge("C", "A")
        with pytest.raises(ValueError, match="Cycle"):
            fg.validate_dag()

    def test_clean_graph_does_not_raise(self):
        fg = _simple_chain(3)
        fg.validate_dag()  # must not raise


# ── 2. Multiple-incoming-edge detection ──────────────────────────────────────

class TestMultipleIncomingEdges:
    def test_two_parents_raises(self):
        fg = FrameGraph()
        for n in ["P1", "P2", "C"]:
            fg.add_frame(n)
        fg.add_edge("P1", "C", _htm(), _zero_tol(), name="P1->C")
        fg.add_edge("P2", "C", _htm(), _zero_tol(), name="P2->C")
        with pytest.raises(ValueError, match="'C'"):
            fg.validate_dag()

    def test_error_names_the_frame(self):
        fg = FrameGraph()
        for n in ["A", "B", "BottleneckFrame"]:
            fg.add_frame(n)
        fg.add_edge("A", "BottleneckFrame", _htm(), _zero_tol(), name="A->BF")
        fg.add_edge("B", "BottleneckFrame", _htm(), _zero_tol(), name="B->BF")
        with pytest.raises(ValueError, match="BottleneckFrame"):
            fg.validate_dag()


# ── 3. Root identification ────────────────────────────────────────────────────

class TestRootFrames:
    def test_single_root(self):
        fg = _simple_chain(3)
        assert fg.root_frames() == ["root"]

    def test_two_disjoint_chains_have_two_roots(self):
        fg = FrameGraph()
        for n in ["R1", "A", "R2", "B"]:
            fg.add_frame(n)
        fg.add_edge("R1", "A", _htm(), _zero_tol())
        fg.add_edge("R2", "B", _htm(), _zero_tol())
        roots = set(fg.root_frames())
        assert roots == {"R1", "R2"}


# ── 4. Shared-frame (junction) ────────────────────────────────────────────────

class TestSharedFrame:
    def test_nominal_transform_through_shared_ancestor(self):
        # root --(T1=1m X)--> mid --(T2=0.5m Y)--> tip_A
        #                          --(T3=0.5m Z)--> tip_B  (NOT ALLOWED: multi-parent on mid)
        # Instead build a true junction where mid is a shared PARENT:
        # root -> mid -> tip_A
        #             -> tip_B   <-- mid has TWO outgoing edges (fine)
        fg = FrameGraph()
        for n in ["root", "mid", "tip_A", "tip_B"]:
            fg.add_frame(n)
        fg.add_edge("root", "mid", _htm(x=1.0), _zero_tol(), name="root->mid")
        fg.add_edge("mid", "tip_A", _htm(y=0.5), _zero_tol(), name="mid->tip_A")
        fg.add_edge("mid", "tip_B", _htm(z=0.5), _zero_tol(), name="mid->tip_B")
        fg.validate_dag()

        # root -> tip_A should compose root->mid then mid->tip_A
        T = fg.nominal_transform_between("root", "tip_A")
        np.testing.assert_allclose(T.matrix[:3, 3], [1.0, 0.5, 0.0], atol=1e-9)

        # root -> tip_B similarly
        T2 = fg.nominal_transform_between("root", "tip_B")
        np.testing.assert_allclose(T2.matrix[:3, 3], [1.0, 0.0, 0.5], atol=1e-9)

    def test_cross_branch_path_through_ancestor(self):
        # tip_A -> root -> tip_B (path goes backwards then forwards)
        fg = FrameGraph()
        for n in ["root", "mid", "tip_A", "tip_B"]:
            fg.add_frame(n)
        fg.add_edge("root", "mid", _htm(x=1.0), _zero_tol(), name="root->mid")
        fg.add_edge("mid", "tip_A", _htm(y=0.5), _zero_tol(), name="mid->tip_A")
        fg.add_edge("mid", "tip_B", _htm(z=0.5), _zero_tol(), name="mid->tip_B")
        # tip_A -> tip_B: go backwards through mid->tip_A, then forwards through mid->tip_B
        T = fg.nominal_transform_between("tip_A", "tip_B")
        # = inv(T_mid->tip_A) @ T_mid->tip_B = inv(0.5Y) @ 0.5Z
        T_A = _htm(y=0.5)
        T_B = _htm(z=0.5)
        expected = T_A.inverse().compose(T_B)
        assert T.is_close(expected, atol=1e-9)


# ── 5. Disjoint-component error with locked message ──────────────────────────

class TestDisjointFrames:
    def _two_disjoint_chains(self):
        fg = FrameGraph()
        for n in ["R1", "A", "R2", "B"]:
            fg.add_frame(n)
        fg.add_edge("R1", "A", _htm(), _zero_tol())
        fg.add_edge("R2", "B", _htm(), _zero_tol())
        return fg

    def test_raises_disjoint_error(self):
        fg = self._two_disjoint_chains()
        with pytest.raises(DisjointFramesError):
            fg.nominal_transform_between("A", "B")

    def test_error_message_names_both_frames(self):
        fg = self._two_disjoint_chains()
        with pytest.raises(DisjointFramesError, match="'A'"):
            fg.nominal_transform_between("A", "B")
        with pytest.raises(DisjointFramesError, match="'B'"):
            fg.nominal_transform_between("A", "B")

    def test_exact_locked_message_text(self):
        fg = self._two_disjoint_chains()
        with pytest.raises(DisjointFramesError) as exc_info:
            fg.nominal_transform_between("A", "B")
        msg = str(exc_info.value)
        assert "no connected path" in msg
        assert "machine" in msg or "optical table" in msg
        assert "explicit joint Frame" in msg


# ── 6. Common-physical-base pattern (Section 2.3.1) ──────────────────────────

class TestCommonPhysicalBase:
    def test_two_chains_connected_through_base(self):
        # Chain 1: base --(1m X)--> cam
        # Chain 2: base --(2m Y)--> sample
        # Before connecting both to base, cam and sample would be disjoint.
        fg = FrameGraph()
        for n in ["base", "cam", "sample"]:
            fg.add_frame(n)
        fg.add_edge("base", "cam", _htm(x=1.0), _zero_tol(), name="base->cam")
        fg.add_edge("base", "sample", _htm(y=2.0), _zero_tol(), name="base->sample")
        fg.validate_dag()

        # Now cam and sample share a component via base — path should work.
        T = fg.nominal_transform_between("cam", "sample")
        # = inv(T_base->cam) @ T_base->sample = inv(1m X) @ (2m Y)
        expected = _htm(x=1.0).inverse().compose(_htm(y=2.0))
        assert T.is_close(expected, atol=1e-9)

    def test_base_tolerance_present(self):
        # A nonzero tolerance on a base edge should remain accessible (not silently dropped).
        z = ToleranceSpec("uniform", bound=0.0)
        big = ToleranceSpec("uniform", bound=0.005)
        tol_with_base = ToleranceSpec6(big, z, z, z, z, z)
        fg = FrameGraph()
        for n in ["base", "cam", "sample"]:
            fg.add_frame(n)
        fg.add_edge("base", "cam", _htm(x=1.0), tol_with_base, name="base->cam")
        fg.add_edge("base", "sample", _htm(y=2.0), _zero_tol(), name="base->sample")
        edge = fg._edges["base->cam"]
        assert edge.tolerance.dx.bound == 0.005


# ── 7. Adjoint and sensitivity cross-check ───────────────────────────────────

class TestSensitivity:
    def _make_two_edge_chain(self):
        """A -> B -> C with known non-trivial nominals."""
        fg = FrameGraph()
        for n in ["A", "B", "C"]:
            fg.add_frame(n)
        # T1: 0.1 rad around Z, 0.5 m along X
        fg.add_edge("A", "B", _htm(x=0.5, ez=0.1), _zero_tol(), name="A->B")
        # T2: 0.05 rad around Y, 0.3 m along Z
        fg.add_edge("B", "C", _htm(z=0.3, ey=0.05), _zero_tol(), name="B->C")
        return fg

    def test_adjoint_identity_is_identity(self):
        Ad = adjoint(HTM.from_matrix(np.eye(4)))
        np.testing.assert_allclose(Ad, np.eye(6), atol=1e-12)

    def test_adjoint_pure_translation(self):
        # T = pure translation by t = [tx, 0, 0], R = I.
        # [v,ω] convention: Ad_T = [[R, skew(t)@R], [0, R]] = [[I, skew(t)], [0, I]]
        # skew([tx,0,0]) = [[0,0,0],[0,0,-tx],[0,tx,0]]
        tx = 2.0
        T = HTM.from_xyz_euler([tx, 0.0, 0.0], [0.0, 0.0, 0.0])
        Ad = adjoint(T)
        expected_top_right = np.array([[0.,  0., 0.],
                                        [0.,  0., -tx],
                                        [0.,  tx,  0.]])
        np.testing.assert_allclose(Ad[:3, :3], np.eye(3), atol=1e-12)   # R top-left
        np.testing.assert_allclose(Ad[3:, 3:], np.eye(3), atol=1e-12)   # R bottom-right
        np.testing.assert_allclose(Ad[:3, 3:], expected_top_right, atol=1e-12)  # skew(t)@R
        np.testing.assert_allclose(Ad[3:, :3], np.zeros((3, 3)), atol=1e-12)   # 0

    def test_sensitivity_shape(self):
        fg = self._make_two_edge_chain()
        J = compute_sensitivity(fg, "A", "C", ["A->B", "B->C"])
        assert J.shape == (6, 12)

    def test_sensitivity_finite_difference(self):
        """Cross-check compute_sensitivity against numerical perturbation."""
        fg = self._make_two_edge_chain()
        J = compute_sensitivity(fg, "A", "C", ["A->B", "B->C"])

        eps = 1e-6
        T_nominal = fg.nominal_transform_between("A", "C")

        # For each edge × each DoF, perturb and finite-difference.
        edge_names = ["A->B", "B->C"]
        for edge_idx, ename in enumerate(edge_names):
            edge = fg._edges[ename]
            for dof in range(6):
                delta = np.zeros(6)
                delta[dof] = eps

                # Build perturbed edge nominal.
                from core.tolerance import delta_to_htm_batch
                T_delta = delta_to_htm_batch(delta[np.newaxis])[0]
                T_perturbed_nominal = HTM.from_matrix(edge.nominal.matrix @ T_delta)

                # Rebuild the chain with one perturbed edge.
                fg2 = FrameGraph()
                for n in ["A", "B", "C"]:
                    fg2.add_frame(n)
                fg2.add_edge("A", "B",
                             T_perturbed_nominal if ename == "A->B" else fg._edges["A->B"].nominal,
                             _zero_tol(), name="A->B")
                fg2.add_edge("B", "C",
                             T_perturbed_nominal if ename == "B->C" else fg._edges["B->C"].nominal,
                             _zero_tol(), name="B->C")

                T_perturbed = fg2.nominal_transform_between("A", "C")

                # Extract 6-vector difference: delta_T = T_perturbed @ T_nominal^{-1}
                delta_T = T_perturbed.compose(T_nominal.inverse()).matrix
                # Output in [v, ω] order to match our adjoint convention.
                d_v = delta_T[:3, 3]
                d_omega = np.array([
                    (delta_T[2, 1] - delta_T[1, 2]) / 2,
                    (delta_T[0, 2] - delta_T[2, 0]) / 2,
                    (delta_T[1, 0] - delta_T[0, 1]) / 2,
                ])
                fd_col = np.concatenate([d_v, d_omega]) / eps

                # The Jacobian column for this (edge, dof) pair.
                j_col = J[:, edge_idx * 6 + dof]

                np.testing.assert_allclose(j_col, fd_col, atol=1e-5,
                    err_msg=f"Sensitivity mismatch at edge={ename}, dof={dof}")


# ── Topological order ─────────────────────────────────────────────────────────

class TestTopologicalOrder:
    def test_order_is_topological(self):
        fg = _simple_chain(4)
        order = fg.topological_edge_order()
        # Edge i must appear before edge i+1 in the result.
        edge_names = [f"root->F1", "F1->F2", "F2->F3", "F3->F4"]
        indices = [order.index(n) for n in edge_names]
        assert indices == sorted(indices)

    def test_all_edges_appear_exactly_once(self):
        fg = _simple_chain(3)
        order = fg.topological_edge_order()
        assert sorted(order) == sorted(fg._edges.keys())


# ── add_frame / add_edge validation ──────────────────────────────────────────

class TestBuildValidation:
    def test_duplicate_frame_raises(self):
        fg = FrameGraph()
        fg.add_frame("A")
        with pytest.raises(ValueError, match="'A'"):
            fg.add_frame("A")

    def test_missing_parent_raises(self):
        fg = FrameGraph()
        fg.add_frame("B")
        with pytest.raises(ValueError, match="'A'"):
            fg.add_edge("A", "B", _htm(), _zero_tol())

    def test_missing_child_raises(self):
        fg = FrameGraph()
        fg.add_frame("A")
        with pytest.raises(ValueError, match="'B'"):
            fg.add_edge("A", "B", _htm(), _zero_tol())

    def test_duplicate_edge_name_raises(self):
        fg = FrameGraph()
        for n in ["A", "B", "C"]:
            fg.add_frame(n)
        fg.add_edge("A", "B", _htm(), _zero_tol(), name="myedge")
        with pytest.raises(ValueError, match="'myedge'"):
            fg.add_edge("B", "C", _htm(), _zero_tol(), name="myedge")

    def test_identity_transform_is_valid(self):
        fg = FrameGraph()
        fg.add_frame("A")
        fg.add_frame("B")
        fg.add_edge("A", "B", HTM.from_matrix(np.eye(4)), _zero_tol())
        T = fg.nominal_transform_between("A", "B")
        assert T.is_close(HTM.from_matrix(np.eye(4)), atol=1e-12)
