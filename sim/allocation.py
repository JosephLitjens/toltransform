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


class EqualAllocation(AllocationObjective):
    """Uniform scale-factor allocation: every free edge gets the same bound.

    For each active output DoF k, computes:
        s_k = B_k / Σ_{free (i,j)} |J[k, 6*i+j]|

    where the sum runs only over non-locked (edge, DoF) pairs. Uses the most
    restrictive: s = min(s_k), ensuring every active constraint is satisfied
    simultaneously at the linear-approximation level (Option A, locked 2026-06-26).

    Non-locked DoF on free edges receive ToleranceSpec("uniform", bound=s).
    Locked DoF on free edges are left unchanged.
    """

    def solve(
        self,
        sensitivity_matrix: np.ndarray,
        target_tolerance: ToleranceSpec6,
        free_edges: list,
    ) -> dict[str, ToleranceSpec6]:
        J = sensitivity_matrix  # (6, 6*N)
        N = len(free_edges)

        # Boolean mask over the 6*N columns: True = free (non-locked) DoF.
        free_mask = np.zeros(6 * N, dtype=bool)
        for i, edge in enumerate(free_edges):
            for j in range(6):
                if not edge.tolerance[j].locked:
                    free_mask[6 * i + j] = True

        # Per-output-DoF scale factors, restricted to free columns only.
        s_values: list[float] = []
        for k in range(6):
            row_sum = float(np.sum(np.abs(J[k, free_mask])))
            if row_sum > 0.0:
                s_values.append(target_tolerance[k].bound / row_sum)

        if not s_values:
            # Degenerate: all free-column sensitivities are zero.
            # Fall back to the smallest target bound as a safe conservative default.
            s = float(min(target_tolerance[k].bound for k in range(6)))
        else:
            s = min(s_values)

        # Build the proposed ToleranceSpec6 for each free edge.
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
        fg_copy = _copy_frame_graph_with_tolerances(frame_graph, proposed_tolerances)
        trial_data = MonteCarloFKEngine.run(fg_copy, n_trials, seed)
        achieved = point_pair_envelope_box(trial_data, fg_copy, frame_a, frame_b)

        per_dof_pass: dict[str, bool] = {}
        for k, label in enumerate(DOF_LABELS):
            half_width = max(
                abs(achieved[label]["min"]), abs(achieved[label]["max"])
            )
            per_dof_pass[label] = half_width <= target_tolerance[k].bound

        return ValidationReport(
            achieved_envelope=achieved,
            passed=all(per_dof_pass.values()),
            per_dof_pass=per_dof_pass,
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
            )

        current = copy.deepcopy(baseline)
        for iteration in range(max_iter):
            current = _damp_angular(current, gamma)
            report = AllocationEngine.validate(
                frame_graph, current, frame_a, frame_b, target_tolerance, n_validate, seed
            )
            if report.passed:
                return AllocationResult(
                    baseline_linear_allocation=baseline,
                    corrected_allocation=current,
                    converged=True,
                    iterations_used=iteration + 1,
                    status_message="",
                    final_validation_report=report,
                    target_tolerance=target_tolerance,
                )

        return AllocationResult(
            baseline_linear_allocation=baseline,
            corrected_allocation=current,
            converged=False,
            iterations_used=max_iter,
            status_message="Allocation could not converge to target budget",
            final_validation_report=report,
            target_tolerance=target_tolerance,
        )
