"""
Inverse tolerance allocation engine (Mode 2, Section 3.2).

Builds on core/frame_graph.py's compute_sensitivity() — no Jacobian math
is re-implemented here. The allocation objective is extensible via
AllocationObjective; EqualAllocation is the only v1 implementation.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from core.frame_graph import FrameGraph, compute_sensitivity
from core.tolerance import ToleranceSpec, ToleranceSpec6
from postprocess.stats import point_pair_envelope_box
from sim.monte_carlo_fk import MonteCarloFKEngine

DOF_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz"]
_ANGULAR_INDICES = (3, 4, 5)  # rx, ry, rz within a 6-vector


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Outcome of a single MC validation pass against a target envelope."""
    achieved_envelope: dict[str, dict[str, float]]
    passed: bool
    per_dof_pass: dict[str, bool]


@dataclass
class AllocationResult:
    """Full output of AllocationEngine.allocate().

    Both allocations are always present. The difference between
    baseline_linear_allocation and corrected_allocation quantifies how much
    geometric-leverage coupling exists in the chain (Section 6.7 Step 5f).
    """
    baseline_linear_allocation: dict[str, ToleranceSpec6]
    corrected_allocation: dict[str, ToleranceSpec6]
    converged: bool
    iterations_used: int
    status_message: str
    final_validation_report: ValidationReport
    target_tolerance: ToleranceSpec6 | None = None
    method: str = ""


# ── Allocation objective interface ────────────────────────────────────────────

class AllocationObjective(ABC):
    """Strategy interface for computing per-edge tolerances from a sensitivity matrix."""

    @abstractmethod
    def solve(
        self,
        sensitivity_matrix: np.ndarray,
        target_tolerance: ToleranceSpec6,
        free_edges: list,
    ) -> dict[str, ToleranceSpec6]:
        """Propose per-edge tolerances for the free edges.

        Parameters
        ----------
        sensitivity_matrix : np.ndarray, shape (6, 6*N)
            Adjoint-based Jacobian from compute_sensitivity(), one 6-column block
            per free edge in free_edges order.
        target_tolerance : ToleranceSpec6
            Desired output-frame error envelope.
        free_edges : list[HTMEdge]
            The N free edges in the same order as sensitivity_matrix columns.

        Returns
        -------
        dict[str, ToleranceSpec6]
            Proposed tolerance keyed by edge name, covering every free edge.
        """


def _build_free_mask(free_edges: list, N: int) -> np.ndarray:
    """Boolean mask (6*N,): True where edge DoF is not locked."""
    mask = np.zeros(6 * N, dtype=bool)
    for i, edge in enumerate(free_edges):
        for j in range(6):
            if not edge.tolerance[j].locked:
                mask[6 * i + j] = True
    return mask


def _build_result(free_edges: list, s: float) -> dict[str, ToleranceSpec6]:
    """Assign uniform bound=s to every free DoF; copy locked DoF unchanged."""
    result: dict[str, ToleranceSpec6] = {}
    for edge in free_edges:
        specs = []
        for j in range(6):
            existing = edge.tolerance[j]
            if existing.locked:
                specs.append(copy.copy(existing))
            else:
                specs.append(ToleranceSpec(distribution="uniform", bound=s))
        result[edge.name] = ToleranceSpec6(*specs)
    return result


class EqualAllocation(AllocationObjective):
    """Worst-case (linear-sum) equal allocation.

    For each active output DoF k:
        s_k = B_k / Σ_{free cols} |J[k, col]|

    s = min(s_k).  Assumes all sources contribute simultaneously in the same
    direction — maximally conservative.  Appropriate when hard bounds must be
    guaranteed regardless of error direction correlation.
    """

    def solve(
        self,
        sensitivity_matrix: np.ndarray,
        target_tolerance: ToleranceSpec6,
        free_edges: list,
    ) -> dict[str, ToleranceSpec6]:
        J = sensitivity_matrix
        N = len(free_edges)
        free_mask = _build_free_mask(free_edges, N)

        s_values: list[float] = []
        for k in range(6):
            row_sum = float(np.sum(np.abs(J[k, free_mask])))
            if row_sum > 0.0:
                s_values.append(target_tolerance[k].bound / row_sum)

        s = min(s_values) if s_values else float(min(target_tolerance[k].bound for k in range(6)))
        return _build_result(free_edges, s)


class RSSAllocation(AllocationObjective):
    """Statistical (RSS) equal allocation.

    For each active output DoF k:
        s_k = B_k / sqrt(Σ_{free cols} J[k, col]²)

    s = min(s_k).  Assumes sources are statistically independent — correct for
    manufacturing processes where errors are uncorrelated.  Gives bounds
    sqrt(N_free) times less conservative than EqualAllocation when N free DoF
    contribute equally, where N_free is the number of active free columns.
    """

    def solve(
        self,
        sensitivity_matrix: np.ndarray,
        target_tolerance: ToleranceSpec6,
        free_edges: list,
    ) -> dict[str, ToleranceSpec6]:
        J = sensitivity_matrix
        N = len(free_edges)
        free_mask = _build_free_mask(free_edges, N)

        s_values: list[float] = []
        for k in range(6):
            rss = float(np.sqrt(np.sum(J[k, free_mask] ** 2)))
            if rss > 0.0:
                s_values.append(target_tolerance[k].bound / rss)

        s = min(s_values) if s_values else float(min(target_tolerance[k].bound for k in range(6)))
        return _build_result(free_edges, s)


def _build_type_masks(
    free_edges: list, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (trans_mask, ang_mask) — free translational / angular columns."""
    trans = np.zeros(6 * N, dtype=bool)
    ang = np.zeros(6 * N, dtype=bool)
    for i, edge in enumerate(free_edges):
        for j in range(6):
            if not edge.tolerance[j].locked:
                if j < 3:
                    trans[6 * i + j] = True
                else:
                    ang[6 * i + j] = True
    return trans, ang


def _build_split_result(
    free_edges: list, s_trans: float, s_ang: float
) -> dict[str, ToleranceSpec6]:
    """Assign s_trans to free translational DoF, s_ang to free angular DoF."""
    result: dict[str, ToleranceSpec6] = {}
    for edge in free_edges:
        specs = []
        for j in range(6):
            existing = edge.tolerance[j]
            if existing.locked:
                specs.append(copy.copy(existing))
            else:
                s = s_trans if j < 3 else s_ang
                specs.append(ToleranceSpec(distribution="uniform", bound=s))
        result[edge.name] = ToleranceSpec6(*specs)
    return result


class SplitAllocation(AllocationObjective):
    """Two-step allocation that assigns separate bounds to translational and angular DoF.

    Problem with equal allocation: a large lever arm forces angular DoF tight
    (bound ≈ target/L), and that same tight bound is applied to translational DoF
    even though they have no lever-arm amplification.

    Fix — two steps:

    Step 1 — s_ang from ALL output rows, angular input columns only:
        For each k: s_k = B_k / sqrt(Σ_{ang_free_cols} J[k,col]²)  (RSS)
                 or s_k = B_k / Σ_{ang_free_cols} |J[k,col]|       (worst-case)
        s_ang = min(s_k) over rows where ang denom > 0.
        This captures both direct angular output coupling AND lever-arm coupling.

    Step 2 — s_trans from translational output rows, after subtracting s_ang's
    contribution from the budget:
        For each k in {0,1,2}:
          residual² = B_k² - Σ_{ang_free_cols} J[k,col]² · s_ang²  (RSS)
          s_trans_k = sqrt(residual²) / sqrt(Σ_{trans_free_cols} J[k,col]²)
        s_trans = min(s_trans_k) over rows with positive residual.
        Falls back to s_ang when no residual remains (lever arm fully consumes budget).

    Constructor parameter mode: "rss" (default) or "worst_case".
    """

    def __init__(self, mode: str = "rss") -> None:
        if mode not in ("rss", "worst_case"):
            raise ValueError(f"mode must be 'rss' or 'worst_case', got {mode!r}")
        self._mode = mode

    def solve(
        self,
        sensitivity_matrix: np.ndarray,
        target_tolerance: ToleranceSpec6,
        free_edges: list,
    ) -> dict[str, ToleranceSpec6]:
        J = sensitivity_matrix
        N = len(free_edges)
        trans_mask, ang_mask = _build_type_masks(free_edges, N)

        # ── Step 1: s_ang ─────────────────────────────────────────────────────
        s_ang_values: list[float] = []
        for k in range(6):
            if self._mode == "rss":
                denom = float(np.sqrt(np.sum(J[k, ang_mask] ** 2)))
            else:
                denom = float(np.sum(np.abs(J[k, ang_mask])))
            if denom > 0.0:
                s_ang_values.append(target_tolerance[k].bound / denom)

        s_ang = (
            min(s_ang_values) if s_ang_values
            else float(min(target_tolerance[k].bound for k in range(6)))
        )

        # ── Step 2: s_trans from residual translational budget ────────────────
        s_trans_values: list[float] = []
        for k in range(3):  # translational output rows only
            if self._mode == "rss":
                ang_used = float(np.sum(J[k, ang_mask] ** 2)) * s_ang ** 2
                residual_sq = target_tolerance[k].bound ** 2 - ang_used
                if residual_sq <= 0.0:
                    continue
                trans_denom = float(np.sqrt(np.sum(J[k, trans_mask] ** 2)))
                if trans_denom > 0.0:
                    s_trans_values.append(float(np.sqrt(residual_sq)) / trans_denom)
            else:
                ang_used = float(np.sum(np.abs(J[k, ang_mask]))) * s_ang
                residual = target_tolerance[k].bound - ang_used
                if residual <= 0.0:
                    continue
                trans_denom = float(np.sum(np.abs(J[k, trans_mask])))
                if trans_denom > 0.0:
                    s_trans_values.append(residual / trans_denom)

        s_trans = min(s_trans_values) if s_trans_values else s_ang

        return _build_split_result(free_edges, s_trans, s_ang)


# ── Private helpers ───────────────────────────────────────────────────────────

def _copy_frame_graph_with_tolerances(
    fg: FrameGraph,
    new_tolerances: dict[str, ToleranceSpec6],
) -> FrameGraph:
    """Return a new FrameGraph with the same structure but updated edge tolerances."""
    new_fg = FrameGraph()
    for frame in fg.all_frames():
        new_fg.add_frame(frame.name, copy.deepcopy(frame.metadata))
    for edge in fg.all_edges():
        new_fg.add_edge(
            parent=edge.parent,
            child=edge.child,
            nominal=edge.nominal,
            tolerance=new_tolerances.get(edge.name, edge.tolerance),
            name=edge.name,
        )
    return new_fg


def _damp_angular(
    allocation: dict[str, ToleranceSpec6], gamma: float
) -> dict[str, ToleranceSpec6]:
    """Return a new allocation with angular DoF (rx, ry, rz) bounds scaled by gamma.

    Translation bounds (dx, dy, dz) are unchanged — the failure mode this corrects
    is angular-to-positional lever-arm coupling, not translational error (Section 6.7).
    Locked DoF are not damped.
    """
    result: dict[str, ToleranceSpec6] = {}
    for edge_name, tol6 in allocation.items():
        specs = [tol6[j] for j in range(6)]
        for j in _ANGULAR_INDICES:
            spec = specs[j]
            if not spec.locked:
                specs[j] = ToleranceSpec(
                    distribution=spec.distribution,
                    bound=spec.bound * gamma,
                    sigma_level=spec.sigma_level,
                    locked=False,
                )
        result[edge_name] = ToleranceSpec6(*specs)
    return result


def _scale_angular(
    allocation: dict[str, ToleranceSpec6], factor: float
) -> dict[str, ToleranceSpec6]:
    """Return a copy of allocation with every free angular bound multiplied by factor."""
    result: dict[str, ToleranceSpec6] = {}
    for edge_name, tol6 in allocation.items():
        specs = [tol6[j] for j in range(6)]
        for j in _ANGULAR_INDICES:
            spec = specs[j]
            if not spec.locked:
                specs[j] = ToleranceSpec(
                    distribution=spec.distribution,
                    bound=spec.bound * factor,
                    sigma_level=spec.sigma_level,
                    locked=False,
                )
        result[edge_name] = ToleranceSpec6(*specs)
    return result


def _bisect_angular(
    frame_graph,
    lo: dict[str, ToleranceSpec6],   # passing allocation (tighter)
    hi: dict[str, ToleranceSpec6],   # failing allocation (looser)
    frame_a: str,
    frame_b: str,
    target_tolerance,
    n_validate: int,
    seed: int,
    tol: float = 0.005,  # stop when hi/lo angular ratio < 0.5%
) -> tuple[dict[str, ToleranceSpec6], "ValidationReport"]:
    """Binary search between lo (passing) and hi (failing) on the angular scale factor.

    Recovers the slack introduced by the fixed gamma step in the damping loop.
    Returns the loosest allocation that still passes MC validation.
    """
    # Extract angular scale from any free angular DoF on any edge (all equal).
    lo_scale = hi_scale = 1.0
    for tol6 in lo.values():
        for j in _ANGULAR_INDICES:
            if not tol6[j].locked and tol6[j].bound > 0:
                lo_scale = tol6[j].bound
                break
        else:
            continue
        break
    for tol6 in hi.values():
        for j in _ANGULAR_INDICES:
            if not tol6[j].locked and tol6[j].bound > 0:
                hi_scale = tol6[j].bound
                break
        else:
            continue
        break

    best = lo
    best_report = None

    while (hi_scale - lo_scale) / max(hi_scale, 1e-12) > tol:
        mid_scale = (lo_scale + hi_scale) / 2.0
        ratio = mid_scale / lo_scale
        mid = _scale_angular(lo, ratio)
        report = _mc_validate(
            frame_graph, mid, frame_a, frame_b, target_tolerance, n_validate, seed
        )
        if report.passed:
            best = mid
            best_report = report
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale

    if best_report is None:
        best_report = _mc_validate(frame_graph, best, frame_a, frame_b, target_tolerance, n_validate, seed)

    return best, best_report


def _mc_validate(
    frame_graph, allocation, frame_a, frame_b, target_tolerance, n_validate, seed
) -> "ValidationReport":
    fg_copy = _copy_frame_graph_with_tolerances(frame_graph, allocation)
    trial_data = MonteCarloFKEngine.run(fg_copy, n_validate, seed)
    achieved = point_pair_envelope_box(trial_data, fg_copy, frame_a, frame_b)
    per_dof_pass = {
        label: max(abs(achieved[label]["min"]), abs(achieved[label]["max"]))
               <= target_tolerance[k].bound
        for k, label in enumerate(DOF_LABELS)
    }
    return ValidationReport(
        achieved_envelope=achieved,
        passed=all(per_dof_pass.values()),
        per_dof_pass=per_dof_pass,
    )


# ── AllocationEngine ──────────────────────────────────────────────────────────

class AllocationEngine:
    """Inverse tolerance allocator (Section 6.7).

    Entry point: allocate() — wraps solve() and validate() in a nonlinear
    damping/correction loop. solve() and validate() are also public for
    direct unit testing and inspection.
    """

    @staticmethod
    def solve(
        frame_graph: FrameGraph,
        frame_a: str,
        frame_b: str,
        target_tolerance: ToleranceSpec6,
        objective: AllocationObjective | None = None,
    ) -> dict[str, ToleranceSpec6]:
        """Closed-form linear allocation step.

        Returns
        -------
        dict[str, ToleranceSpec6]
            Proposed tolerance for each free edge on the path frame_a -> frame_b.

        Raises
        ------
        ValueError
            If the path has no free edges (all path edges are entirely locked).
        """
        if objective is None:
            objective = EqualAllocation()

        frame_graph.validate_dag()
        path = frame_graph.path_edges_between(frame_a, frame_b)

        # Free edges: those where not ALL 6 DoF are locked.
        free_edges = [
            edge for edge, _ in path
            if not all(edge.tolerance[j].locked for j in range(6))
        ]
        if not free_edges:
            raise ValueError(
                "No free edges to allocate — all path edges are locked"
            )

        free_edge_names = [e.name for e in free_edges]
        J = compute_sensitivity(frame_graph, frame_a, frame_b, free_edge_names)
        return objective.solve(J, target_tolerance, free_edges)

    @staticmethod
    def validate(
        frame_graph: FrameGraph,
        proposed_tolerances: dict[str, ToleranceSpec6],
        frame_a: str,
        frame_b: str,
        target_tolerance: ToleranceSpec6,
        n_trials: int,
        seed: int,
    ) -> ValidationReport:
        """Single MC validation pass for a proposed allocation.

        Builds a copy of frame_graph with proposed_tolerances applied, runs
        MonteCarloFKEngine, and compares the achieved envelope to target_tolerance.
        """
        return _mc_validate(
            frame_graph, proposed_tolerances, frame_a, frame_b,
            target_tolerance, n_trials, seed,
        )

    @staticmethod
    def allocate(
        frame_graph: FrameGraph,
        frame_a: str,
        frame_b: str,
        target_tolerance: ToleranceSpec6,
        objective: AllocationObjective | None = None,
        n_validate: int = 1000,
        gamma: float = 0.9,
        max_iter: int = 10,
        seed: int = 42,
    ) -> AllocationResult:
        """Iterative inverse allocation with nonlinear damping/correction.

        1. Computes the baseline linear allocation via solve().
        2. Validates it with n_validate MC trials.
        3. If validation fails (e.g., angular-to-positional lever-arm coupling),
           applies damping factor gamma to angular DoF bounds and retries,
           up to max_iter times.

        Returns
        -------
        AllocationResult
            Always carries both baseline_linear_allocation and corrected_allocation.
            If no correction was needed, both fields hold the same allocation object.
            If max_iter is reached without convergence, returns converged=False with
            status_message == "Allocation could not converge to target budget".
        """
        if objective is None:
            objective = EqualAllocation()
        method_name = type(objective).__name__

        baseline = AllocationEngine.solve(
            frame_graph, frame_a, frame_b, target_tolerance, objective
        )
        report = AllocationEngine.validate(
            frame_graph, baseline, frame_a, frame_b, target_tolerance, n_validate, seed
        )
        if report.passed:
            return AllocationResult(
                baseline_linear_allocation=baseline,
                corrected_allocation=baseline,
                converged=True,
                iterations_used=0,
                status_message="",
                final_validation_report=report,
                target_tolerance=target_tolerance,
                method=method_name,
            )

        current = copy.deepcopy(baseline)
        prev = baseline  # last failing allocation (tighter is safer to start)
        for iteration in range(max_iter):
            prev = current
            current = _damp_angular(current, gamma)
            report = AllocationEngine.validate(
                frame_graph, current, frame_a, frame_b, target_tolerance, n_validate, seed
            )
            if report.passed:
                # Binary-search between current (passing) and prev (failing) to
                # recover the slack introduced by the fixed gamma step size.
                current, report = _bisect_angular(
                    frame_graph, current, prev, frame_a, frame_b,
                    target_tolerance, n_validate, seed,
                )
                return AllocationResult(
                    baseline_linear_allocation=baseline,
                    corrected_allocation=current,
                    converged=True,
                    iterations_used=iteration + 1,
                    status_message="",
                    final_validation_report=report,
                    target_tolerance=target_tolerance,
                    method=method_name,
                )

        return AllocationResult(
            baseline_linear_allocation=baseline,
            corrected_allocation=current,
            converged=False,
            iterations_used=max_iter,
            status_message="Allocation could not converge to target budget",
            final_validation_report=report,
            target_tolerance=target_tolerance,
            method=method_name,
        )
