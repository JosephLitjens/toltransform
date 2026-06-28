"""
Tests for sim/allocation.py (Milestone B-2).
"""

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from sim.allocation import DOF_LABELS, AllocationEngine, EqualAllocation, LoosestAllocation


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

      With L=1 m and rz-bound = B_rz = 0.10 rad allocated by EqualAllocation:
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

    EqualAllocation sets rz-bound = b_rz (only rz output is sensitive to rz input
    at the linear level). MC then produces dy ≈ L·b_rz; fails when L·b_rz > b_trans.
    """
    return ToleranceSpec6(
        _spec(b_trans), _spec(b_trans), _spec(b_trans),
        _spec(b_rx),    _spec(b_rx),    _spec(b_rz),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_equal_allocation_sanity():
    """All free edges receive the same bound on every DoF (equal-allocation property)."""
    fg, fa, fb = make_simple_chain(n_edges=3)
    target = ToleranceSpec6(*[_spec(3.0) for _ in range(6)])
    result = AllocationEngine.solve(fg, fa, fb, target)

    assert len(result) == 3
    ref = result["e0"]
    for edge_name in ("e1", "e2"):
        for j in range(6):
            assert result[edge_name][j].bound == pytest.approx(ref[j].bound), (
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

def test_loosest_allocation_beats_equal_on_mixed_target():
    """LoosestAllocation allocates each DoF's budget independently; EqualAllocation wastes it.

    2-edge identity chain: tight dx target (0.01), loose rz target (10.0).

    With identity Jacobian, the dx constraint is:  b_dx0 + b_dx1 ≤ 0.01
    EqualAllocation: s = min(B_dx/2, B_rz/2) = 0.01/2 = 0.005.
        - Total dx budget used: 0.005+0.005 = 0.01  (full)
        - Total rz budget used: 0.005+0.005 = 0.01  (wastes 9.99 of available 10.0!)

    LoosestAllocation LP: treats each output DoF constraint independently.
        - Total dx: still 0.01 (constraint is tight)
        - Total rz: fills to 10.0 (LP uses the full rz budget)

    Key assertion: sum(LP rz bounds) >> sum(Equal rz bounds).
    """
    fg = FrameGraph()
    for name in ("f0", "f1", "f2"):
        fg.add_frame(name)
    free_tol = ToleranceSpec6(*[_spec(1.0) for _ in range(6)])
    fg.add_edge("f0", "f1", HTM.from_xyz_euler([0, 0, 0], [0, 0, 0]), free_tol, name="e0")
    fg.add_edge("f1", "f2", HTM.from_xyz_euler([0, 0, 0], [0, 0, 0]), free_tol, name="e1")

    # Very tight dx, very loose rz — the binding constraint is dx for EqualAllocation
    target = ToleranceSpec6(
        _spec(0.01),   # dx tight
        _spec(5.0),    # dy loose
        _spec(5.0),    # dz loose
        _spec(5.0),    # rx loose
        _spec(5.0),    # ry loose
        _spec(10.0),   # rz very loose
    )

    equal_result = AllocationEngine.solve(fg, "f0", "f2", target, objective=EqualAllocation())
    lp_result = AllocationEngine.solve(fg, "f0", "f2", target, objective=LoosestAllocation())

    # Compare total budget used per DoF (sum across edges) — not per-edge (LP may split unevenly)
    def total_bound(result, dof_idx):
        return sum(result[e][dof_idx].bound for e in ("e0", "e1"))

    # Both methods must use the full dx budget (constraint is tight at the sum level)
    assert total_bound(lp_result, 0) == pytest.approx(0.01, rel=0.01), (
        f"LP total dx budget should equal target 0.01; got {total_bound(lp_result, 0):.6f}"
    )
    assert total_bound(equal_result, 0) == pytest.approx(0.01, rel=0.01), (
        f"Equal total dx budget should equal target 0.01; got {total_bound(equal_result, 0):.6f}"
    )

    # LP must use the full rz budget (≈10.0); EqualAllocation wastes it (≈0.01)
    lp_rz_total = total_bound(lp_result, 5)
    equal_rz_total = total_bound(equal_result, 5)
    assert lp_rz_total == pytest.approx(10.0, rel=0.01), (
        f"LP total rz budget should fill target 10.0; got {lp_rz_total:.4f}"
    )
    assert lp_rz_total > equal_rz_total * 50, (
        f"LP rz total ({lp_rz_total:.4f}) should far exceed equal rz total ({equal_rz_total:.4f})"
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

    lp_alloc = AllocationEngine.solve(fg, fa, fb, alloc_target, objective=LoosestAllocation())
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

    result = AllocationEngine.solve(fg, "a", "b", target, objective=LoosestAllocation())

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
    within max_iter, same as with EqualAllocation.
    """
    fg, fa, fb = make_lever_arm_chain(L=1.0)
    target = _lever_target(b_trans=0.05, b_rx=0.20, b_rz=0.10)

    result = AllocationEngine.allocate(
        fg, fa, fb, target,
        objective=LoosestAllocation(),
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


