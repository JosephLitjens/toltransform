"""
Inverse tolerance allocation engine (Mode 2, Section 3.2).

Builds on core/frame_graph.py's compute_sensitivity() — no Jacobian math
is re-implemented here. The allocation objective is extensible via
AllocationObjective; EqualAllocation is the only v1 implementation.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from core.frame_graph import FrameGraph, compute_sensitivity
from core.tolerance import ToleranceSpec, ToleranceSpec6

DOF_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz"]


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
