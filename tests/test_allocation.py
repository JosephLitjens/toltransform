"""
Tests for sim/allocation.py (Milestone B-2).
"""

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from sim.allocation import DOF_LABELS, AllocationEngine, LoosestAllocation, _mc_validate_multi


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spec(bound=1.0, locked=False):
    return ToleranceSpec("uniform", bound, locked=locked)


def _free_tol6(bound=1.0):
    return ToleranceSpec6(*[_spec(bound) for _ in range(6)])


def _locked_tol6():
    return ToleranceSpec6(*[_spec(0.0, locked=True) for _ in range(6)])


def _identity():
    return HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def make_simple_chain(n_edges=3, locked_indices=()):
    """Serial chain with identity nominal transforms.

    All edges have ±1 free bounds unless listed in locked_indices (0-based),
    in which case all 6 DoF are locked with bound=0.
    """
    fg = FrameGraph()
    frames = [f"f{i}" for i in range(n_edges + 1)]
    for name in frames:
        fg.add_frame(name)
    for i in range(n_edges):
        tol = _locked_tol6() if i in locked_indices else _free_tol6(1.0)
        fg.add_edge(frames[i], frames[i + 1], _identity(), tol, name=f"e{i}")
    return fg, frames[0], frames[-1]


def make_lever_arm_chain(L=1.0):
    """Three-frame chain designed to expose the MC / linear Jacobian discrepancy.

    Layout:
      base  → pivot : nominal = I,        only rz FREE; rest locked, bound=0
      pivot → arm   : nominal = Tx(L),    all locked, bound=0
      arm   → exit  : nominal = Ry(π/2),  all locked, bound=0  ← downstream Ry node

    Why the discrepancy:
      The linear Jacobian block for "pivot_edge" is Ad_{T_{base→pivot}} = Ad_I = I_6.
      Column 5 (rz input) of I_6 is [0,0,0,0,0,1] — zero dy coupling.

      In the MC, the rz perturbation rotates the pivot frame, then the LOCKED Tx(L) arm
      is applied in that rotated frame, sweeping the exit node through a circular arc:
          dy = L · sin(δrz) ≈ L · δrz          (first order — MISSED by Jacobian)
          dx = L · (cos(δrz) − 1) ≈ 0           (second order)

      With L=1 m and rz-bound ≈ B_rz = 0.10 rad (LoosestAllocation, uncoupled):
          dy_mc ≈ 0.10 m  >>  B_dy = 0.05 m  → validation FAILS

      Damping tightens rz by 0.9 per iteration:
          need 0.10 · 0.9^k ≤ 0.05  →  k ≥ 7   (within max_iter=10)
    """
    fg = FrameGraph()
    for name in ("base", "pivot", "arm", "exit"):
        fg.add_frame(name)

    # pivot edge: only rz is free (initial bound is just a placeholder — alloc overwrites it)
    pivot_tol = ToleranceSpec6(
        _spec(0.0, locked=True),   # dx
        _spec(0.0, locked=True),   # dy
        _spec(0.0, locked=True),   # dz
        _spec(0.0, locked=True),   # rx
        _spec(0.0, locked=True),   # ry
        _spec(0.01),               # rz FREE
    )
    fg.add_edge("base", "pivot", _identity(), pivot_tol, name="pivot_edge")

    arm_nom = HTM.from_xyz_euler([L, 0.0, 0.0], [0.0, 0.0, 0.0])
    fg.add_edge("pivot", "arm", arm_nom, _locked_tol6(), name="arm_edge")

    ry_nom = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, np.pi / 2.0, 0.0])
    fg.add_edge("arm", "exit", ry_nom, _locked_tol6(), name="ry_edge")

    return fg, "base", "exit"


def _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10):
    """Symmetric target tolerance for lever-arm tests.

    LoosestAllocation allocates rz-bound ≈ b_rz (only rz output is sensitive to rz input
    at the linear level). MC then produces dy ≈ L·b_rz; fails when L·b_rz > b_trans.
    """
    return ToleranceSpec6(
        _spec(b_trans), _spec(b_trans), _spec(b_trans),
        _spec(b_rx),    _spec(b_rx),    _spec(b_rz),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_solve_returns_all_free_edges():
    """solve() must return a dict covering exactly the free edges on the path."""
    fg, fa, fb = make_simple_chain(n_edges=3)
    target = ToleranceSpec6(*[_spec(3.0) for _ in range(6)])
    result = AllocationEngine.solve(fg, fa, fb, target)

    assert set(result.keys()) == {"e0", "e1", "e2"}
    # Symmetric chain + symmetric target → bounds should be equal across edges
    ref = result["e0"]
    for edge_name in ("e1", "e2"):
        for j in range(6):
            assert result[edge_name][j].bound == pytest.approx(ref[j].bound, rel=0.01), (
                f"edge {edge_name} DoF {j}: bound mismatch with reference edge"
            )


def test_locked_edge_excluded():
    """An entirely-locked edge must be absent from the allocation result."""
    fg, fa, fb = make_simple_chain(n_edges=3, locked_indices=(1,))
    target = _free_tol6(1.0)
    result = AllocationEngine.solve(fg, fa, fb, target)

    assert "e1" not in result, "locked edge e1 must not appear in allocation result"
    assert set(result.keys()) == {"e0", "e2"}


def test_all_edges_locked_raises():
    """All-locked chain must raise ValueError — no free edges to allocate."""
    fg, fa, fb = make_simple_chain(n_edges=3, locked_indices=(0, 1, 2))
    with pytest.raises(ValueError, match="No free edges"):
        AllocationEngine.solve(fg, fa, fb, _free_tol6(1.0))


def test_allocation_mc_validation_discrepancy():
    """Section 9 Item 4 — linear allocation fails MC validation on the lever-arm chain.

    Chain: base→pivot (rz free), pivot→arm (Tx(1m) locked), arm→exit (Ry(π/2) locked).
    Linear Jacobian at exit_node=pivot (T=I) predicts no dy coupling from rz.
    MC captures dy ≈ L·δrz: with L=1 m and rz-bound=0.10, dy_mc ≈ 0.10 > B_dy=0.05.
    This discrepancy is the diagnostic that motivates the iterative damping loop.
    """
    fg, fa, fb = make_lever_arm_chain(L=1.0)
    target = _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10)

    linear_alloc = AllocationEngine.solve(fg, fa, fb, target)
    report = AllocationEngine.validate(
        fg, linear_alloc, fa, fb, target, n_trials=1000, seed=42
    )

    assert not report.passed, "linear allocation must fail MC validation on lever-arm chain"
    assert not report.per_dof_pass["dy"], "dy is the failing DoF (lever-arm coupling)"


def test_damping_loop_convergence():
    """allocate() must converge on the lever-arm chain and tighten angular bounds.

    With L=1 m, B_dy=0.05, B_rz=0.10: need 0.9^k ≤ 0.5 → k≥7 iterations.
    Converges within max_iter=10.
    """
    fg, fa, fb = make_lever_arm_chain(L=1.0)
    target = _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10)

    result = AllocationEngine.allocate(
        fg, fa, fb, target,
        n_validate=1000, gamma=0.9, max_iter=10, seed=42,
    )

    assert result.converged, f"expected convergence; got: {result.status_message}"
    assert result.iterations_used >= 1, "at least one damping iteration must have occurred"

    for edge_name in result.corrected_allocation:
        b_tol = result.baseline_linear_allocation[edge_name]
        c_tol = result.corrected_allocation[edge_name]
        for j in (3, 4, 5):  # rx, ry, rz
            if not b_tol[j].locked:
                assert c_tol[j].bound < b_tol[j].bound, (
                    f"{edge_name} DoF {j}: corrected bound ({c_tol[j].bound:.6f}) "
                    f"must be < baseline ({b_tol[j].bound:.6f})"
                )


def test_damping_loop_nonconvergence():
    """allocate() must report non-convergence when the target is infeasible within max_iter.

    With B_dy=0.001: need 0.9^k ≤ 0.001/0.10 = 0.01 → k≥44, far beyond max_iter=10.
    """
    fg, fa, fb = make_lever_arm_chain(L=1.0)
    target = _lever_target(b_trans=0.001, b_rx=0.20, b_rz=0.10)

    result = AllocationEngine.allocate(
        fg, fa, fb, target,
        n_validate=1000, gamma=0.9, max_iter=10, seed=42,
    )

    assert not result.converged
    assert result.status_message == "Allocation could not converge to target budget"
    assert result.iterations_used == 10


def test_allocation_result_preserves_both_allocations():
    """AllocationResult must carry distinct baseline and corrected dicts.

    After damping, the baseline (looser) and corrected (tighter) allocations must be
    separate objects with baseline angular bounds strictly greater than corrected.
    """
    fg, fa, fb = make_lever_arm_chain(L=1.0)
    target = _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10)

    result = AllocationEngine.allocate(
        fg, fa, fb, target,
        n_validate=1000, gamma=0.9, max_iter=10, seed=42,
    )

    assert result.converged
    assert result.baseline_linear_allocation is not result.corrected_allocation

    for edge_name in result.corrected_allocation:
        b_tol = result.baseline_linear_allocation[edge_name]
        c_tol = result.corrected_allocation[edge_name]
        for j in (3, 4, 5):  # rx, ry, rz
            if not b_tol[j].locked:
                assert b_tol[j].bound > c_tol[j].bound, (
                    f"{edge_name} DoF {j}: baseline ({b_tol[j].bound:.6f}) "
                    f"must exceed corrected ({c_tol[j].bound:.6f})"
                )


# ── LoosestAllocation tests ───────────────────────────────────────────────────

def test_solve_fills_independent_dof_budgets():
    """solve() must fill each DoF's budget independently when constraints are decoupled.

    2-edge identity chain: tight dx target (0.01), loose rz target (10.0).

    With identity Jacobian the constraints are per-DoF independent:
        b_dx0 + b_dx1  ≤  0.01   (tight)
        b_rz0 + b_rz1  ≤  10.0   (loose)

    LoosestAllocation must fill BOTH to their respective limits — it must not
    let the tight dx constraint drag rz down to the same small value.
    """
    fg = FrameGraph()
    for name in ("f0", "f1", "f2"):
        fg.add_frame(name)
    free_tol = ToleranceSpec6(*[_spec(1.0) for _ in range(6)])
    fg.add_edge("f0", "f1", HTM.from_xyz_euler([0, 0, 0], [0, 0, 0]), free_tol, name="e0")
    fg.add_edge("f1", "f2", HTM.from_xyz_euler([0, 0, 0], [0, 0, 0]), free_tol, name="e1")

    target = ToleranceSpec6(
        _spec(0.01),   # dx tight
        _spec(5.0),    # dy loose
        _spec(5.0),    # dz loose
        _spec(5.0),    # rx loose
        _spec(5.0),    # ry loose
        _spec(10.0),   # rz very loose
    )

    result = AllocationEngine.solve(fg, "f0", "f2", target)

    def total_bound(dof_idx):
        return sum(result[e][dof_idx].bound for e in ("e0", "e1"))

    assert total_bound(0) == pytest.approx(0.01, rel=0.01), (
        f"total dx must fill target 0.01; got {total_bound(0):.6f}"
    )
    assert total_bound(5) == pytest.approx(10.0, rel=0.01), (
        f"total rz must fill target 10.0; got {total_bound(5):.4f}"
    )


def test_loosest_allocation_mc_validation_passes():
    """LoosestAllocation's output must satisfy MC validation on a simple chain.

    Uses the simple 3-edge identity chain with a 50% budget margin so that
    MC sampling noise cannot borderline-fail the test: the NLP fills each DoF
    budget to the constraint boundary (sum = B_k per output DoF), and with a
    uniform distribution the MC worst-case sum equals exactly B_k — leaving no
    room for sampling fluctuations.  Setting the MC check target to 1.5×B_k
    gives a clear pass margin while still verifying the allocation is feasible.
    """
    fg, fa, fb = make_simple_chain(n_edges=3)
    alloc_target = ToleranceSpec6(*[_spec(0.4) for _ in range(6)])
    check_target  = ToleranceSpec6(*[_spec(0.6) for _ in range(6)])  # 50% margin

    lp_alloc = AllocationEngine.solve(fg, fa, fb, alloc_target)
    report = AllocationEngine.validate(fg, lp_alloc, fa, fb, check_target, n_trials=2000, seed=7)

    assert report.passed, (
        f"LoosestAllocation must satisfy MC validation (with 50% margin); failing DoFs: "
        f"{[k for k, v in report.per_dof_pass.items() if not v]}"
    )


def test_loosest_allocation_no_zero_bounds_on_coupled_chain():
    """LoosestAllocation must never return a zero (or near-zero) bound due to LP degeneracy.

    A chain with a 90-degree nominal rotation creates cross-coupling in the Jacobian:
    both ry and rz inputs contribute to the same output DoF rows.  A naive linear-sum
    LP assigns all budget to one DoF and zeros the other (vertex degeneracy).
    The log-sum objective must give positive bounds to BOTH ry and rz.

    Regression test for the bug reported: "Ry and Rz show 0.0000000 in corrected allocation."
    """
    # 2-frame chain with Ry(π/2) nominal — couples ry and rz inputs in J rows
    fg = FrameGraph()
    fg.add_frame("a")
    fg.add_frame("b")
    ry90_nom = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, np.pi / 2.0, 0.0])
    fg.add_edge("a", "b", ry90_nom, _free_tol6(0.01), name="e0")

    target = ToleranceSpec6(*[_spec(0.001) for _ in range(6)])

    result = AllocationEngine.solve(fg, "a", "b", target)

    min_bound = 1e-6  # any finite, positive tolerance is acceptable
    for j in range(6):
        bound = result["e0"][j].bound
        assert bound > min_bound, (
            f"LoosestAllocation must not produce near-zero bounds (DoF {j}={DOF_LABELS[j]}: "
            f"got {bound:.2e}, expected > {min_bound:.0e})"
        )


def test_loosest_allocation_lever_arm_converges():
    """LoosestAllocation + damping loop must converge on the lever-arm chain.

    LP allocation is still first-order linear and misses dy≈L·δrz coupling,
    so the baseline will fail MC validation. The damping loop must correct it
    within max_iter.
    """
    fg, fa, fb = make_lever_arm_chain(L=1.0)
    target = _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10)

    result = AllocationEngine.allocate(
        fg, fa, fb, target,
        n_validate=1000, gamma=0.9, max_iter=15, seed=42,
    )

    assert result.converged, f"LoosestAllocation must converge; got: {result.status_message}"
    assert result.method == "LoosestAllocation"
    # Corrected allocation must satisfy the target
    report = result.final_validation_report
    assert report.passed, (
        f"Final validation must pass; failing DoFs: "
        f"{[k for k, v in report.per_dof_pass.items() if not v]}"
    )


# ── Multi-pair allocation tests ───────────────────────────────────────────────

def _make_branching_graph():
    """Three-branch graph: root -> base -> {m1, m2}.

    Edge e_rb  : root -> base  (free, all DoFs)
    Edge e_bm1 : base -> m1   (free, all DoFs)
    Edge e_bm2 : base -> m2   (free, all DoFs)

    e_rb is SHARED between pair (root, m1) and pair (root, m2).
    """
    fg = FrameGraph()
    for f in ("root", "base", "m1", "m2"):
        fg.add_frame(f)
    fg.add_edge("root", "base", _identity(), _free_tol6(1.0), "e_rb")
    fg.add_edge("base", "m1",  _identity(), _free_tol6(1.0), "e_bm1")
    fg.add_edge("base", "m2",  _identity(), _free_tol6(1.0), "e_bm2")
    return fg


def test_solve_multi_shared_edge_respects_tighter_constraint():
    """solve_multi: shared edge bound must satisfy the TIGHTER of two pairs' constraints.

    Graph: root -> base -> m1 (path for pair 1)
           root -> base -> m2 (path for pair 2)
    Shared edge: e_rb (root -> base)

    Pair 1: tight dx = 0.01  (2-edge path: e_rb + e_bm1)
    Pair 2: loose dx = 1.0   (2-edge path: e_rb + e_bm2)

    With identity Jacobians, the stacked constraints are:
      b_rb_dx + b_bm1_dx  <=  0.01   (pair 1, tight)
      b_rb_dx + b_bm2_dx  <=  1.0    (pair 2, loose)

    Log-sum NLP: fills pair 1's constraint to equality → b_rb_dx ≈ 0.005.
    The shared edge e_rb.dx must therefore be <= 0.005, regardless of pair 2's loose budget.
    """
    fg = _make_branching_graph()

    target1 = ToleranceSpec6(_spec(0.01), *[_spec(5.0)] * 5)
    target2 = ToleranceSpec6(_spec(1.0),  *[_spec(5.0)] * 5)
    targets = [("root", "m1", target1), ("root", "m2", target2)]

    alloc = AllocationEngine.solve_multi(fg, targets)

    # All three free edges must appear in the allocation
    assert set(alloc.keys()) == {"e_rb", "e_bm1", "e_bm2"}

    b_rb  = alloc["e_rb"].dx.bound
    b_bm1 = alloc["e_bm1"].dx.bound
    b_bm2 = alloc["e_bm2"].dx.bound

    # Pair 1 constraint must be satisfied at the linear level
    assert b_rb + b_bm1 <= 0.01 + 1e-6, (
        f"Pair-1 dx constraint violated: e_rb={b_rb:.5f} + e_bm1={b_bm1:.5f} > 0.01"
    )
    # The shared edge must be tight (~0.005 each for a 2-edge identity chain)
    assert b_rb < 0.006, (
        f"Shared edge e_rb.dx={b_rb:.5f} should be ≈0.005 (constrained by tight pair 1)"
    )
    # Non-shared pair-2 edge can be loose (pair 2's budget is 1.0)
    assert b_bm2 > 0.9, (
        f"Non-shared edge e_bm2.dx={b_bm2:.5f} should be ≈0.995 (pair 2 budget is loose)"
    )


def test_solve_multi_independent_pairs_unaffected():
    """solve_multi: pairs with disjoint paths are solved independently.

    Graph: f0 -> f1 -> f2 -> f3 (serial chain, 3 edges)
    Pair 1: f0 -> f1  (only e0)
    Pair 2: f2 -> f3  (only e2)
    e1 (f1->f2) lies on neither path and should not appear in the allocation.

    The two pairs share no edges, so their constraints are completely decoupled.
    Each pair's budget should fill its own target independently.
    """
    fg, _, _ = make_simple_chain(n_edges=3)

    target1 = ToleranceSpec6(_spec(0.02), *[_spec(5.0)] * 5)  # tight dx for pair 1
    target2 = ToleranceSpec6(_spec(0.5),  *[_spec(5.0)] * 5)  # looser dx for pair 2
    targets = [("f0", "f1", target1), ("f2", "f3", target2)]

    alloc = AllocationEngine.solve_multi(fg, targets)

    # Only edges on the respective paths appear
    assert set(alloc.keys()) == {"e0", "e2"}, f"Unexpected edges: {set(alloc.keys())}"

    # e0 is constrained by target1 (dx=0.02, 1 edge → bound ≈ 0.02)
    assert alloc["e0"].dx.bound == pytest.approx(0.02, rel=0.01)
    # e2 is constrained by target2 (dx=0.5, 1 edge → bound ≈ 0.5)
    assert alloc["e2"].dx.bound == pytest.approx(0.5, rel=0.01)


def test_allocate_multi_result_structure():
    """allocate_multi must return an AllocationResult with per_pair_validation populated.

    Validates the shape and types of the result, not the numerical optimality.
    """
    fg = _make_branching_graph()
    target1 = ToleranceSpec6(*[_spec(0.05)] * 6)
    target2 = ToleranceSpec6(*[_spec(0.05)] * 6)
    targets = [("root", "m1", target1), ("root", "m2", target2)]

    result = AllocationEngine.allocate_multi(
        fg, targets,
        n_validate=200, max_iter=30, seed=42,
    )

    # Result structure
    assert result.per_pair_validation is not None
    assert len(result.per_pair_validation) == 2

    fa0, fb0, vr0 = result.per_pair_validation[0]
    fa1, fb1, vr1 = result.per_pair_validation[1]
    assert fa0 == "root" and fb0 == "m1"
    assert fa1 == "root" and fb1 == "m2"

    # Allocations cover all three edges
    assert set(result.corrected_allocation.keys()) == {"e_rb", "e_bm1", "e_bm2"}
    assert result.method == "LoosestAllocation"


def test_allocate_multi_mc_validation_check_target():
    """allocate_multi linear allocation passes MC validation when checked at 1.5× target.

    Uses a pair of independent pairs (disjoint paths) with angular DoFs locked so there
    is no angular-to-translation coupling. The NLP fills each pair's budget to its target
    exactly, so the MC max equals the target. Validating at 1.5× gives a clear pass margin.
    """
    # Two single-edge chains with angular locked: f0->f1 and f2->f3
    fg = FrameGraph()
    for f in ("f0", "f1", "f2", "f3"):
        fg.add_frame(f)

    # All angular DoFs locked at 0; only translation is free
    def _trans_free(bound=1.0):
        return ToleranceSpec6(
            _spec(bound), _spec(bound), _spec(bound),   # dx, dy, dz free
            _spec(0.0, locked=True),                    # rx locked
            _spec(0.0, locked=True),                    # ry locked
            _spec(0.0, locked=True),                    # rz locked
        )

    fg.add_edge("f0", "f1", _identity(), _trans_free(1.0), "e0")
    fg.add_edge("f2", "f3", _identity(), _trans_free(1.0), "e2")

    alloc_target = ToleranceSpec6(_spec(0.4), _spec(0.4), _spec(0.4), _spec(1.0), _spec(1.0), _spec(1.0))
    check_target  = ToleranceSpec6(_spec(0.6), _spec(0.6), _spec(0.6), _spec(1.0), _spec(1.0), _spec(1.0))

    targets_alloc  = [("f0", "f1", alloc_target), ("f2", "f3", alloc_target)]
    targets_check  = [("f0", "f1", check_target),  ("f2", "f3", check_target)]

    alloc = AllocationEngine.solve_multi(fg, targets_alloc)

    # Each edge gets 0.4 budget (single edge per pair)
    assert alloc["e0"].dx.bound == pytest.approx(0.4, rel=0.01)
    assert alloc["e2"].dx.bound == pytest.approx(0.4, rel=0.01)

    # MC validation at 1.5× budget must pass for both pairs
    all_passed, per_pair = _mc_validate_multi(fg, alloc, targets_check, n_validate=2000, seed=7)
    assert all_passed, (
        f"Multi-pair MC validation must pass at 1.5× margin. Failing: "
        + ", ".join(f"{fa}→{fb}: {[k for k, v in vr.per_dof_pass.items() if not v]}"
                    for fa, fb, vr in per_pair if not vr.passed)
    )


def test_allocate_multi_lever_arm_two_pairs():
    """allocate_multi with two lever-arm pairs: both must converge.

    Uses the lever-arm fixture (only rz free, all other DoFs locked) for both pairs.
    The two chains are disjoint so their allocations are independent, but both
    must be corrected by the damping loop for angular-to-translation coupling.
    """
    fg = FrameGraph()
    for f in ("base", "pivot", "arm", "exit",
              "base2", "pivot2", "arm2", "exit2"):
        fg.add_frame(f)

    pivot_tol = ToleranceSpec6(
        _spec(0.0, locked=True), _spec(0.0, locked=True), _spec(0.0, locked=True),
        _spec(0.0, locked=True), _spec(0.0, locked=True), _spec(0.01),
    )
    arm_nom = HTM.from_xyz_euler([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ry_nom  = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, np.pi / 2.0, 0.0])

    fg.add_edge("base",  "pivot",  _identity(), pivot_tol,    "pivot_edge")
    fg.add_edge("pivot", "arm",    arm_nom,     _locked_tol6(), "arm_edge")
    fg.add_edge("arm",   "exit",   ry_nom,      _locked_tol6(), "ry_edge")

    fg.add_edge("base2", "pivot2", _identity(), pivot_tol,    "pivot_edge2")
    fg.add_edge("pivot2", "arm2",  arm_nom,     _locked_tol6(), "arm_edge2")
    fg.add_edge("arm2",  "exit2",  ry_nom,      _locked_tol6(), "ry_edge2")

    target = _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10)
    targets = [("base", "exit", target), ("base2", "exit2", target)]

    result = AllocationEngine.allocate_multi(
        fg, targets,
        n_validate=1000, gamma=0.9, max_iter=20, seed=42,
    )

    assert result.converged, (
        f"allocate_multi must converge on two lever-arm pairs; got: {result.status_message}"
    )
    assert result.per_pair_validation is not None
    assert len(result.per_pair_validation) == 2

    for fa, fb, vr in result.per_pair_validation:
        assert vr.passed, (
            f"Pair {fa}→{fb} must pass validation; failing DoFs: "
            f"{[k for k, v in vr.per_dof_pass.items() if not v]}"
        )

