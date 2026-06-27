"""
Tests for sim/allocation.py (Milestone B-2).
"""

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from sim.allocation import AllocationEngine


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
