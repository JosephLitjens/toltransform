"""
Inverse tolerance allocation engine (Mode 2, Section 3.2).

Builds on core/frame_graph.py's compute_sensitivity() — no Jacobian math
is re-implemented here. The allocation objective is extensible via
AllocationObjective; three implementations are provided:

  EqualAllocation   — worst-case linear sum, single uniform scale factor
  RSSAllocation     — statistical RSS, single uniform scale factor
  LoosestAllocation — log-sum NLP per-DoF maximization; each DoF gets the
                      loosest bound its sensitivity allows, with no zeros
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

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
    """Full output of AllocationEngine.allocate() or allocate_multi().

    Both allocations are always present. The difference between
    baseline_linear_allocation and corrected_allocation quantifies how much
    geometric-leverage coupling exists in the chain (Section 6.7 Step 5f).

    per_pair_validation is set by allocate_multi(); it is None for single-pair
    allocate() calls.  Each entry is (frame_a, frame_b, ValidationReport).
    """
    baseline_linear_allocation: dict[str, ToleranceSpec6]
    corrected_allocation: dict[str, ToleranceSpec6]
    converged: bool
    iterations_used: int
    status_message: str
    final_validation_report: ValidationReport
    target_tolerance: ToleranceSpec6 | None = None
    method: str = ""
    per_pair_validation: list[tuple[str, str, ValidationReport]] | None = None
    per_pair_targets: list[tuple[str, str, ToleranceSpec6]] | None = None


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


def _build_result(free_edges: list, bounds: np.ndarray) -> dict[str, ToleranceSpec6]:
    """Build per-edge ToleranceSpec6 from a bounds vector of shape (6*N,).

    Each entry bounds[6*i+j] is the proposed bound for edge i, DoF j.
    Locked DoFs are copied unchanged from the edge's existing spec; their
    corresponding entries in bounds are ignored.
    """
    result: dict[str, ToleranceSpec6] = {}
    for i, edge in enumerate(free_edges):
        specs = []
        for j in range(6):
            existing = edge.tolerance[j]
            if existing.locked:
                specs.append(copy.copy(existing))
            else:
                specs.append(ToleranceSpec(distribution="uniform", bound=float(bounds[6 * i + j])))
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
        bounds = np.full(6 * N, s)
        return _build_result(free_edges, bounds)


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
        bounds = np.full(6 * N, s)
        return _build_result(free_edges, bounds)


class LoosestAllocation(AllocationObjective):
    """Log-sum (geometric-mean) loosest-possible worst-case allocation.

    Solves the convex NLP:

        maximize  Σ_{i,j} log(b_{ij})
        subject to:
            Σ_{i,j} |J[k, 6i+j]| * b_{ij}  ≤  B_k   for each active output DoF k
            b_{ij} ≥ ε  (small positive floor)

    The log-sum objective is the key difference from a plain linear-sum LP.
    A linear-sum LP finds a vertex of the feasible polytope and will assign zero
    to any DoF that competes in the same constraint row as a higher-sensitivity
    DoF — producing exactly-zero bounds that are physically meaningless.  The
    log-sum drives b_{ij} → 0 with an infinite penalty (log(0) = −∞), so every
    free DoF always receives a positive, practically useful bound.

    This is solved via SLSQP (scipy.optimize.minimize).  Warm-started from the
    EqualAllocation solution, which is always feasible and close to the optimum
    for symmetric chains.

    DoFs with zero sensitivity to all output DoFs are unconstrained; their
    bounds grow until they hit CAP (1e6), which is effectively infinite for
    any realistic engineering tolerance.
    """

    CAP: float = 1e6   # upper bound for truly unconstrained (zero-sensitivity) DoFs
    EPS: float = 1e-10  # lower bound floor — prevents log(0) and negative bounds

    def solve(
        self,
        sensitivity_matrix: np.ndarray,
        target_tolerance: ToleranceSpec6,
        free_edges: list,
    ) -> dict[str, ToleranceSpec6]:
        J = sensitivity_matrix
        N = len(free_edges)
        n_vars = 6 * N
        free_mask = _build_free_mask(free_edges, N)
        free_idx = np.where(free_mask)[0]

        # Build active output-DoF constraints using only free-variable columns
        A_rows, b_vals = [], []
        for k in range(6):
            row_free = np.abs(J[k, free_idx])
            if row_free.sum() > 0.0 and target_tolerance[k].bound > 0.0:
                A_rows.append(row_free)
                b_vals.append(target_tolerance[k].bound)

        if not A_rows:
            # No active constraints — return equal allocation as a safe default
            s_vals = [target_tolerance[k].bound for k in range(6) if target_tolerance[k].bound > 0]
            s = min(s_vals) if s_vals else 1.0
            return _build_result(free_edges, np.full(n_vars, s))

        A = np.array(A_rows)   # shape (n_constraints, n_free)
        b = np.array(b_vals)

        x = self._run_nlp(A, b)
        full_bounds = np.zeros(n_vars)
        full_bounds[free_idx] = x
        return _build_result(free_edges, full_bounds)

    def _run_nlp(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Run the log-sum NLP given a pre-built constraint matrix.

        Parameters
        ----------
        A : np.ndarray, shape (n_constraints, n_free)
            Rows are active output-DoF constraint coefficients; columns are free
            input DoFs.  All entries are non-negative (|J| rows).
        b : np.ndarray, shape (n_constraints,)
            Upper-bound targets for each row.

        Returns
        -------
        np.ndarray, shape (n_free,)
            The optimal per-DoF bounds vector.
        """
        n_free = A.shape[1]

        # Objective: minimize −Σ log(x)  (maximise geometric mean of all bounds)
        def obj(x):
            return -np.sum(np.log(np.maximum(x, self.EPS)))

        def obj_jac(x):
            return -1.0 / np.maximum(x, self.EPS)

        # Per-DoF warm start: each variable gets 90% of its tightest individual budget.
        # This avoids the SLSQP linesearch instability caused by a global equal-allocation
        # warm start when targets differ by orders of magnitude (e.g., 0.001 rad vs 10 rad):
        # some DoFs start thousands of times below their optimum and the large gradient
        # mismatch breaks the linesearch.  Starting each DoF near its own optimum fixes this.
        x0 = np.zeros(n_free)
        for j in range(n_free):
            col = A[:, j]
            active = col > 0
            if active.any():
                # For each active constraint, estimate the fair share for this DoF:
                # budget / (sensitivity × number of DoFs sharing this constraint)
                n_sharing = np.maximum((A[active, :] > 0).sum(axis=1), 1)
                x0[j] = float(np.min(b[active] / (col[active] * n_sharing))) * 0.9
            else:
                x0[j] = 1.0  # unconstrained DoF — arbitrary reasonable start

        x0 = np.maximum(x0, self.EPS)

        # trust-constr handles poorly-scaled problems much better than SLSQP
        # (it uses a trust region rather than a linesearch, so large gradient
        # differences between DoFs don't cause divergence).
        lin_con = LinearConstraint(A, lb=-np.inf, ub=b)
        bounds = Bounds(lb=self.EPS, ub=self.CAP)

        res = minimize(
            obj, x0, jac=obj_jac, method="trust-constr",
            constraints=lin_con, bounds=bounds,
            options={"maxiter": 3000, "gtol": 1e-10, "verbose": 0},
        )
        if not res.success:
            raise ValueError(f"LoosestAllocation NLP failed: {res.message}")

        return res.x


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


def _first_free_angular_bound(allocation: dict[str, ToleranceSpec6]) -> float:
    """Return the bound of the first free angular DoF found in the allocation."""
    for tol6 in allocation.values():
        for j in _ANGULAR_INDICES:
            if not tol6[j].locked and tol6[j].bound > 0:
                return tol6[j].bound
    return 1.0  # fallback — no free angular DoFs found


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

    The bisection tracks absolute angular-bound values (lo_scale, hi_scale), and
    always computes the scale ratio relative to the *original* lo allocation
    (base_scale), so that _scale_angular(lo, ratio) produces exactly mid_scale
    regardless of how many iterations have elapsed.
    """
    base_scale = _first_free_angular_bound(lo)   # fixed reference — never updated
    lo_scale = base_scale
    hi_scale = _first_free_angular_bound(hi)

    best = lo
    best_report = None

    while (hi_scale - lo_scale) / max(hi_scale, 1e-12) > tol:
        mid_scale = (lo_scale + hi_scale) / 2.0
        ratio = mid_scale / base_scale            # always relative to original lo
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


_MultiTarget = list[tuple[str, str, "ToleranceSpec6"]]


def _mc_validate_multi(
    frame_graph, allocation, targets: "_MultiTarget", n_validate: int, seed: int
) -> tuple[bool, list[tuple[str, str, "ValidationReport"]]]:
    """Run ONE MC simulation and validate all (frame_a, frame_b, target) pairs.

    Returns (all_passed, per_pair_list) where per_pair_list contains one
    (frame_a, frame_b, ValidationReport) tuple per entry in targets.
    """
    fg_copy = _copy_frame_graph_with_tolerances(frame_graph, allocation)
    trial_data = MonteCarloFKEngine.run(fg_copy, n_validate, seed)

    all_passed = True
    per_pair: list[tuple[str, str, ValidationReport]] = []
    for frame_a, frame_b, target in targets:
        achieved = point_pair_envelope_box(trial_data, fg_copy, frame_a, frame_b)
        per_dof_pass = {
            label: max(abs(achieved[label]["min"]), abs(achieved[label]["max"]))
                   <= target[k].bound
            for k, label in enumerate(DOF_LABELS)
        }
        passed = all(per_dof_pass.values())
        all_passed = all_passed and passed
        per_pair.append((frame_a, frame_b, ValidationReport(
            achieved_envelope=achieved,
            passed=passed,
            per_dof_pass=per_dof_pass,
        )))

    return all_passed, per_pair


def _bisect_angular_multi(
    frame_graph,
    lo: dict[str, "ToleranceSpec6"],
    hi: dict[str, "ToleranceSpec6"],
    targets: "_MultiTarget",
    n_validate: int,
    seed: int,
    tol: float = 0.005,
) -> tuple[dict[str, "ToleranceSpec6"], list[tuple[str, str, "ValidationReport"]]]:
    """Binary search between lo (passing) and hi (failing) on the angular scale factor.

    Multi-pair variant of _bisect_angular — validates all pairs simultaneously.
    Returns (best_allocation, per_pair_reports).
    """
    base_scale = _first_free_angular_bound(lo)
    lo_scale = base_scale
    hi_scale = _first_free_angular_bound(hi)

    best = lo
    best_per_pair: list[tuple[str, str, ValidationReport]] = []

    while (hi_scale - lo_scale) / max(hi_scale, 1e-12) > tol:
        mid_scale = (lo_scale + hi_scale) / 2.0
        ratio = mid_scale / base_scale
        mid = _scale_angular(lo, ratio)
        all_passed, per_pair = _mc_validate_multi(frame_graph, mid, targets, n_validate, seed)
        if all_passed:
            best = mid
            best_per_pair = per_pair
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale

    if not best_per_pair:
        _, best_per_pair = _mc_validate_multi(frame_graph, best, targets, n_validate, seed)

    return best, best_per_pair


def _combine_validation_reports(
    per_pair: list[tuple[str, str, "ValidationReport"]],
) -> "ValidationReport":
    """Synthesise a combined ValidationReport that passes only if every pair passes.

    The achieved_envelope and per_dof_pass reflect the worst (most-exceeded)
    pair so the results viewer can show a single representative summary row.
    """
    if not per_pair:
        return ValidationReport(achieved_envelope={}, passed=True, per_dof_pass={})

    combined_envelope: dict[str, dict[str, float]] = {}
    combined_per_dof: dict[str, bool] = {}
    all_passed = True

    for dof in DOF_LABELS:
        worst_min = float("inf")
        worst_max = float("-inf")
        dof_pass = True
        for _, _, rpt in per_pair:
            d = rpt.achieved_envelope.get(dof, {})
            worst_min = min(worst_min, d.get("min", 0.0))
            worst_max = max(worst_max, d.get("max", 0.0))
            dof_pass = dof_pass and rpt.per_dof_pass.get(dof, True)
        combined_envelope[dof] = {"min": worst_min if worst_min != float("inf") else 0.0,
                                   "max": worst_max if worst_max != float("-inf") else 0.0}
        combined_per_dof[dof] = dof_pass
        all_passed = all_passed and dof_pass

    return ValidationReport(
        achieved_envelope=combined_envelope,
        passed=all_passed,
        per_dof_pass=combined_per_dof,
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
            objective = LoosestAllocation()
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

    @staticmethod
    def solve_multi(
        frame_graph: FrameGraph,
        targets: list[tuple[str, str, ToleranceSpec6]],
        objective: AllocationObjective | None = None,
    ) -> dict[str, ToleranceSpec6]:
        """Closed-form linear allocation step for multiple simultaneous point-pair constraints.

        Builds a stacked constraint matrix covering all (frame_a, frame_b, target) pairs,
        then maximises per-DoF bounds subject to all constraints simultaneously.

        Shared edges — those that appear on more than one pair's path — are handled
        correctly by construction: their columns appear in multiple constraint rows, so
        the optimizer cannot loosen them for one pair without affecting another.

        Parameters
        ----------
        targets : list of (frame_a, frame_b, target_tolerance)
            Each entry specifies one point-pair constraint.  All frames must be
            in the same connected component of frame_graph.

        Returns
        -------
        dict[str, ToleranceSpec6]
            Proposed tolerance for every free edge appearing on any pair's path.
        """
        if objective is None:
            objective = LoosestAllocation()

        frame_graph.validate_dag()

        # 1. Collect the union of free edges across all pair paths (preserving order).
        all_free_edges: list = []
        seen_names: set[str] = set()
        pair_free: list[list] = []
        for frame_a, frame_b, _ in targets:
            path = frame_graph.path_edges_between(frame_a, frame_b)
            p_free = [e for e, _ in path if not all(e.tolerance[j].locked for j in range(6))]
            pair_free.append(p_free)
            for edge in p_free:
                if edge.name not in seen_names:
                    all_free_edges.append(edge)
                    seen_names.add(edge.name)

        if not all_free_edges:
            raise ValueError("No free edges to allocate — all path edges are locked")

        N_total = len(all_free_edges)
        edge_name_to_global: dict[str, int] = {e.name: i for i, e in enumerate(all_free_edges)}
        free_mask_global = _build_free_mask(all_free_edges, N_total)
        free_idx_global = np.where(free_mask_global)[0]

        # 2. For each pair, build a full-width Jacobian (zeros for edges not on its path).
        A_rows: list[np.ndarray] = []
        b_vals: list[float] = []

        for (frame_a, frame_b, target), p_free_edges in zip(targets, pair_free):
            if not p_free_edges:
                continue
            p_names = [e.name for e in p_free_edges]
            J_compact = compute_sensitivity(frame_graph, frame_a, frame_b, p_names)

            J_full = np.zeros((6, 6 * N_total))
            for local_i, name in enumerate(p_names):
                g = edge_name_to_global[name]
                J_full[:, 6 * g : 6 * g + 6] = J_compact[:, 6 * local_i : 6 * local_i + 6]

            for k in range(6):
                row_free = np.abs(J_full[k, free_idx_global])
                if row_free.sum() > 0.0 and target[k].bound > 0.0:
                    A_rows.append(row_free)
                    b_vals.append(target[k].bound)

        if not A_rows:
            # All constraints inactive — equal-allocate from the tightest target bound.
            all_bounds = [t[k].bound for _, _, t in targets for k in range(6) if t[k].bound > 0]
            s = min(all_bounds) if all_bounds else 1.0
            return _build_result(all_free_edges, np.full(6 * N_total, s))

        A = np.array(A_rows)
        b = np.array(b_vals)

        # 3. Solve with the objective.
        if isinstance(objective, LoosestAllocation):
            x = objective._run_nlp(A, b)
            full_bounds = np.zeros(6 * N_total)
            full_bounds[free_idx_global] = x
            return _build_result(all_free_edges, full_bounds)

        # Fallback for EqualAllocation / RSSAllocation: single scale factor across all pairs.
        if isinstance(objective, RSSAllocation):
            s_values = [float(bk / np.sqrt(np.sum(row ** 2))) for row, bk in zip(A, b) if np.sum(row ** 2) > 0]
        else:  # EqualAllocation or unknown
            s_values = [float(bk / row.sum()) for row, bk in zip(A, b) if row.sum() > 0]
        s = min(s_values) if s_values else 1.0
        return _build_result(all_free_edges, np.full(6 * N_total, s))

    @staticmethod
    def allocate_multi(
        frame_graph: FrameGraph,
        targets: list[tuple[str, str, ToleranceSpec6]],
        objective: AllocationObjective | None = None,
        n_validate: int = 1000,
        gamma: float = 0.9,
        max_iter: int = 10,
        seed: int = 42,
    ) -> AllocationResult:
        """Iterative inverse allocation for multiple simultaneous point-pair constraints.

        Identical in structure to allocate() but optimises per-DoF bounds so that
        ALL specified (frame_a, frame_b, target_tolerance) pairs are satisfied at once.
        The per_pair_validation field of the returned AllocationResult contains individual
        ValidationReport objects for each pair.

        The damping loop applies gamma to ALL free angular DoFs when ANY pair fails MC
        validation.  The bisection step recovers the slack introduced by the fixed step.
        """
        if objective is None:
            objective = LoosestAllocation()
        method_name = type(objective).__name__

        baseline = AllocationEngine.solve_multi(frame_graph, targets, objective)

        all_passed, per_pair = _mc_validate_multi(frame_graph, baseline, targets, n_validate, seed)

        # Synthesise a combined ValidationReport (first pair is the canonical reference;
        # per_pair carries the full per-pair breakdown).
        combined_report = _combine_validation_reports(per_pair)

        if all_passed:
            return AllocationResult(
                baseline_linear_allocation=baseline,
                corrected_allocation=baseline,
                converged=True,
                iterations_used=0,
                status_message="",
                final_validation_report=combined_report,
                per_pair_validation=per_pair,
                per_pair_targets=targets,
                method=method_name,
            )

        current = copy.deepcopy(baseline)
        prev = baseline
        for iteration in range(max_iter):
            prev = current
            current = _damp_angular(current, gamma)
            all_passed, per_pair = _mc_validate_multi(
                frame_graph, current, targets, n_validate, seed
            )
            if all_passed:
                current, per_pair = _bisect_angular_multi(
                    frame_graph, current, prev, targets, n_validate, seed
                )
                combined_report = _combine_validation_reports(per_pair)
                return AllocationResult(
                    baseline_linear_allocation=baseline,
                    corrected_allocation=current,
                    converged=True,
                    iterations_used=iteration + 1,
                    status_message="",
                    final_validation_report=combined_report,
                    per_pair_validation=per_pair,
                    per_pair_targets=targets,
                    method=method_name,
                )

        combined_report = _combine_validation_reports(per_pair)
        return AllocationResult(
            baseline_linear_allocation=baseline,
            corrected_allocation=current,
            converged=False,
            iterations_used=max_iter,
            status_message="Allocation could not converge to target budget",
            final_validation_report=combined_report,
            per_pair_validation=per_pair,
            per_pair_targets=targets,
            method=method_name,
        )
