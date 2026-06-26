"""
tests/test_physical_validation.py — Section 9.1 Physical Validation Benchmarks.

Three module-level named regression tests that validate the forward engine against
concrete, hand-verifiable physical predictions. All three must pass before Milestone
B-2 (inverse allocation) begins.

9.1.1  Linear Stack-Up (RSS) Benchmark
9.1.2  Sine-Bar Lever Arm Benchmark
9.1.3  Common-Ancestor Cancellation Benchmark
"""

from __future__ import annotations

import numpy as np

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from postprocess.stats import (
    frame_envelope_box,
    point_pair_envelope_box,
    pose_error_vector_batch,
)
from sim.monte_carlo_fk import MonteCarloFKEngine

SMALL_ANGLE_ATOL = 1e-6  # mirrors tests/conftest.py; inline for direct readability


# ── Private helpers ───────────────────────────────────────────────────────────
# Not imported from tests/conftest.py because conftest is not a regular importable
# module in pytest's default discovery mode.

def _normal_tol(
    bound_t: float,
    bound_r: float = 0.0,
    sigma_level: float = 3.0,
) -> ToleranceSpec6:
    """Normal distribution on all 3 translation DoF; optional rotation bound."""
    t = ToleranceSpec("normal", bound=bound_t, sigma_level=sigma_level)
    z = ToleranceSpec("uniform", bound=0.0)
    r = ToleranceSpec("normal", bound=bound_r, sigma_level=sigma_level) if bound_r else z
    return ToleranceSpec6(dx=t, dy=t, dz=t, rx=r, ry=r, rz=r)


def _uniform_tol(bound: float) -> ToleranceSpec6:
    s = ToleranceSpec("uniform", bound=bound)
    return ToleranceSpec6(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)


def _zero_tol() -> ToleranceSpec6:
    return _uniform_tol(0.0)


def _rz_only_tol(bound: float, sigma_level: float = 3.0) -> ToleranceSpec6:
    """Normal distribution on rz only; all other DoF fixed at zero."""
    z = ToleranceSpec("uniform", bound=0.0)
    r = ToleranceSpec("normal", bound=bound, sigma_level=sigma_level)
    return ToleranceSpec6(dx=z, dy=z, dz=z, rx=z, ry=z, rz=r)


# ── Benchmark 9.1.1 ──────────────────────────────────────────────────────────

def test_rss_linear_stack_up():
    """Section 9.1.1 — Linear Stack-Up RSS Benchmark.

    A 5-link purely serial translation chain with independent normal-distribution
    tolerances must produce output variance exactly equal to the classical RSS sum:
        σ_total² = Σ σ_i²
    where σ_i = bound_i / sigma_level for each link.

    This is the simplest case where the MC engine has a closed-form analytical answer.
    The assertion is bounded by the quantified sampling error for the chosen n_trials
    (5 standard deviations of the sample-variance distribution), not an arbitrary
    loose tolerance.
    """
    # Build a 5-link purely translational chain (f0 → f1 → … → f5).
    # Each edge: 10 mm nominal x-translation, zero nominal rotation.
    SIGMA_LEVEL = 3.0
    BOUNDS = [0.001, 0.002, 0.003, 0.002, 0.001]  # meters; distinct per link

    fg = FrameGraph()
    frames = [f"f{i}" for i in range(6)]
    for name in frames:
        fg.add_frame(name)

    edges_in_order = []
    for i, bound in enumerate(BOUNDS):
        nom = HTM.from_xyz_euler([0.010, 0.0, 0.0], [0.0, 0.0, 0.0])
        tol = _normal_tol(bound_t=bound, bound_r=0.0, sigma_level=SIGMA_LEVEL)
        fg.add_edge(frames[i], frames[i + 1], nom, tol, name=f"e{i}")
        edges_in_order.append(nom)

    # Run Monte Carlo.
    N = 100_000
    td = MonteCarloFKEngine.run(fg, n_trials=N, seed=0)

    # Nominal endpoint matrix (f0 → f5): composition of all 5 edge nominals.
    nominal_f5 = np.linalg.multi_dot([e.matrix for e in edges_in_order])

    # Extract error vectors: shape (N, 6), columns [dx, dy, dz, rx, ry, rz].
    errors = pose_error_vector_batch(td.frame_poses["f5"], nominal_f5)

    # RSS expected variance (same for dx, dy, dz since all bounds are applied to
    # all 3 translation axes identically).
    expected_var = sum((b / SIGMA_LEVEL) ** 2 for b in BOUNDS)

    # Quantified sampling-error bound: 5 standard deviations of the sample-variance
    # estimator for a normal distribution, SE = σ² * sqrt(2 / (N-1)) ≈ σ² * sqrt(2/N).
    sampling_tol = 5.0 * expected_var * np.sqrt(2.0 / N)

    for axis_idx, axis_name in enumerate(["dx", "dy", "dz"]):
        measured_var = np.var(errors[:, axis_idx])
        assert abs(measured_var - expected_var) < sampling_tol, (
            f"RSS variance mismatch on {axis_name}: "
            f"measured={measured_var:.6e}, expected={expected_var:.6e}, "
            f"sampling_tol={sampling_tol:.6e}"
        )


# ── Benchmark 9.1.2 ──────────────────────────────────────────────────────────

def test_sine_bar_lever_arm():
    """Section 9.1.2 — Sine-Bar Lever Arm Benchmark.

    A single angular pivot (rz tolerance only) followed by a fixed translational
    arm of length L must produce lateral error Δy ≈ L·θ. Formally:

        var(dy) ≈ L² × var(rz)

    This directly validates the geometric-leverage effect that motivates the inverse
    allocation engine's damping loop — if this benchmark fails, the justification for
    that loop is unfounded.

    Setup: ≤ 1 mrad pivot angle (comfortably within the small-angle regime).
    """
    L = 0.100          # arm length (m): 100 mm
    RZ_BOUND = 0.001   # 1 mrad — at the spec's stated small-angle limit
    SIGMA_LEVEL = 3.0

    fg = FrameGraph()
    for name in ["pivot", "arm_mid", "arm_tip"]:
        fg.add_frame(name)

    # Edge 1: angular pivot at origin — rz tolerance only, zero nominal rotation.
    fg.add_edge(
        "pivot", "arm_mid",
        HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _rz_only_tol(RZ_BOUND, SIGMA_LEVEL),
        name="pivot_edge",
    )
    # Edge 2: fixed arm of length L along x — zero tolerance.
    fg.add_edge(
        "arm_mid", "arm_tip",
        HTM.from_xyz_euler([L, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _zero_tol(),
        name="arm_edge",
    )

    N = 100_000
    td = MonteCarloFKEngine.run(fg, n_trials=N, seed=0)

    # Nominal arm_tip matrix: identity × translate_x(L).
    nominal_tip = np.eye(4)
    nominal_tip[0, 3] = L

    # Extract errors: columns dy=1, rz=5.
    errors = pose_error_vector_batch(td.frame_poses["arm_tip"], nominal_tip)

    var_dy = np.var(errors[:, 1])
    var_rz = np.var(errors[:, 5])

    # Primary assertion: lever-arm variance cross-coupling.
    # At 1 mrad the second-order correction L·θ²/2 ≈ 5e-8 is < 0.01% of L·θ,
    # so 1% rtol is appropriate and well-motivated.
    np.testing.assert_allclose(
        var_dy, L ** 2 * var_rz,
        rtol=0.01,
        err_msg=(
            f"Lever-arm variance mismatch: var(dy)={var_dy:.4e}, "
            f"L²×var(rz)={L**2 * var_rz:.4e}"
        ),
    )

    # Secondary assertion: no systematic bias in the lateral direction.
    assert abs(np.mean(errors[:, 1])) < SMALL_ANGLE_ATOL, (
        f"Non-zero mean dy: {np.mean(errors[:, 1]):.4e}"
    )


# ── Benchmark 9.1.3 ──────────────────────────────────────────────────────────

def test_common_ancestor_cancellation():
    """Section 9.1.3 — Common-Ancestor Cancellation Benchmark.

    A large structural tolerance on a shared upstream edge (floor → granite base)
    must cancel completely from the relative measurement between two downstream
    frames (camera and sample) that both branch from the shared ancestor.

    Physical analogy (Section 1.4, "Mitigation Verification"): grounding a
    metrology rail and a sample stage to the same granite base — vibration or
    thermal drift of the base moves both instruments together, so the relative
    alignment is unaffected by that shared structural uncertainty.

    The test confirms:
    1. The relative camera↔sample envelope reflects only the camera and sample
       chains' own small tolerances, not the huge shared tolerance.
    2. The shared tolerance IS large in the absolute frame (sanity check that
       the test is meaningful and the cancellation property is non-trivial).
    """
    LARGE_BOUND = 1.0    # 1 m — deliberately huge shared structural uncertainty
    SMALL_BOUND = 0.001  # 1 mm — individual instrument mounting tolerances

    fg = FrameGraph()
    for name in ["room", "granite_base", "camera", "sample"]:
        fg.add_frame(name)

    # Shared structural edge: large, loosely-toleranced (floor → base attachment).
    fg.add_edge(
        "room", "granite_base",
        HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _uniform_tol(LARGE_BOUND),
        name="floor_to_base",
    )
    # Camera instrument mounting on the base — small tolerance.
    fg.add_edge(
        "granite_base", "camera",
        HTM.from_xyz_euler([0.050, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _uniform_tol(SMALL_BOUND),
        name="base_to_camera",
    )
    # Sample instrument mounting on the base — small tolerance.
    fg.add_edge(
        "granite_base", "sample",
        HTM.from_xyz_euler([0.0, 0.050, 0.0], [0.0, 0.0, 0.0]),
        _uniform_tol(SMALL_BOUND),
        name="base_to_sample",
    )

    N = 50_000
    td = MonteCarloFKEngine.run(fg, n_trials=N, seed=0)

    # ── Assertion 1: relative envelope must be bounded by the small tolerances ──
    # The shared floor_to_base perturbation appears identically in both camera and
    # sample trial poses, so it cancels when computing the relative pose.
    # Worst-case relative error for each DoF: ±2×SMALL_BOUND (both small chains
    # pull in opposite directions). The factor-3 margin is generous headroom while
    # remaining orders of magnitude below LARGE_BOUND = 1.0.
    rel_envelope = point_pair_envelope_box(td, fg, "camera", "sample")

    for dof in ["dx", "dy", "dz", "rx", "ry", "rz"]:
        assert rel_envelope[dof]["max"] < 3 * SMALL_BOUND, (
            f"Relative {dof} max={rel_envelope[dof]['max']:.4e} exceeds 3×SMALL_BOUND="
            f"{3*SMALL_BOUND:.4e}. Shared tolerance may not be cancelling."
        )
        assert rel_envelope[dof]["min"] > -3 * SMALL_BOUND, (
            f"Relative {dof} min={rel_envelope[dof]['min']:.4e} below -3×SMALL_BOUND="
            f"{-3*SMALL_BOUND:.4e}. Shared tolerance may not be cancelling."
        )

    # ── Assertion 2: the shared tolerance IS large in the absolute frame ─────
    # Without this check, the test would trivially pass even if LARGE_BOUND had
    # no effect (e.g., if the engine ignored that edge). Confirms the property is
    # non-trivial: large absolute uncertainty, cancelled in relative measurement.
    abs_envelope = frame_envelope_box(td, "camera")
    assert abs_envelope["dx"]["max"] > 0.5 * LARGE_BOUND, (
        f"Camera absolute dx max={abs_envelope['dx']['max']:.4e} should be large "
        f"(> 0.5×LARGE_BOUND={0.5*LARGE_BOUND:.4e}), confirming shared tolerance "
        f"is active in the absolute frame."
    )
