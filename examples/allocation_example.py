"""
Inverse Tolerance Allocation Example: Precision Wafer Inspection System.

Demonstrates the inverse problem that TolTransform is designed to solve:
given a required accuracy at the output, what tolerance must each joint
meet for the chain to stay within budget?

  ALLOCATION — Work backwards from a target end-effector envelope to
  individual joint tolerances. TolTransform's AllocationEngine does this
  in two steps: (1) a closed-form linear allocation via the first-order
  Jacobian, followed by (2) an iterative damping loop that tightens angular
  bounds if Monte Carlo validation reveals that lever-arm coupling makes the
  linear estimate too optimistic.

Scenario: 3-frame chain representing a wafer-inspection gantry.

    wafer_chuck  ──(stage_mount, 300 mm z)──►  stage
                                                  │
                                     (sensor_arm, 500 mm x)
                                                  │
                                                  ▼
                                             sensor_head

The 500 mm lateral sensor arm creates a pronounced lever-arm: even a small
tilt (rz or ry) of the stage mount produces a significant positional error
at sensor_head.  This causes the first-order linear allocation to
under-tighten angular bounds — the damping loop then corrects it.

Units throughout: metres / radians.

Run from the repo root:
    python examples/allocation_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from postprocess.stats import DOF_LABELS, frame_envelope_box
from sim.allocation import AllocationEngine
from sim.monte_carlo_fk import MonteCarloFKEngine


# ── Output helpers ─────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def _unit(dof: str) -> tuple[float, str]:
    """Return (scale, label) for a DoF — mm/rad for translations, rad for rotations."""
    if dof.startswith("d"):
        return 1000.0, "mm"
    return 1.0, "rad"


def print_envelope(env: dict, *, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{'DoF':6}  {'min':>12}  {'max':>12}  {'range':>10}")
    print(pad + "-" * 46)
    for dof in DOF_LABELS:
        scale, unit = _unit(dof)
        lo = env[dof]["min"] * scale
        hi = env[dof]["max"] * scale
        print(f"{pad}{dof:6}  {lo:>12.4f}  {hi:>12.4f}  {(hi-lo):>10.4f}  {unit}")


def print_allocation_comparison(
    baseline: dict[str, ToleranceSpec6],
    corrected: dict[str, ToleranceSpec6],
    *,
    indent: int = 2,
) -> None:
    """Print baseline vs corrected allocations side-by-side with delta column."""
    pad = " " * indent
    col_w = 14
    print(
        f"{pad}{'Edge':16}  {'DoF':6}  "
        f"{'Baseline (linear)':>{col_w}}  "
        f"{'Corrected (MC)':>{col_w}}  "
        f"{'Δ (tightened)':>14}"
    )
    print(pad + "-" * 76)
    for edge_name in sorted(baseline):
        tol_b = baseline[edge_name]
        tol_c = corrected[edge_name]
        for i, dof in enumerate(DOF_LABELS):
            scale, unit = _unit(dof)
            b_bound = tol_b[i].bound * scale
            c_bound = tol_c[i].bound * scale
            pct = (b_bound - c_bound) / b_bound * 100 if b_bound else 0.0
            delta_str = f"{pct:+.1f}%" if abs(pct) > 0.01 else "—"
            print(
                f"{pad}{edge_name:16}  {dof:6}  "
                f"{b_bound:>{col_w}.5f}  "
                f"{c_bound:>{col_w}.5f}  "
                f"{delta_str:>14}  {unit}"
            )


def print_validation(
    validation,
    target: ToleranceSpec6,
    *,
    indent: int = 2,
) -> None:
    """Print achieved envelope vs target with per-DoF pass/fail."""
    pad = " " * indent
    print(f"{pad}{'DoF':6}  {'Target ±':>12}  {'Achieved ½-range':>18}  {'Pass?':>6}")
    print(pad + "-" * 50)
    for i, dof in enumerate(DOF_LABELS):
        scale, unit = _unit(dof)
        target_bound = target[i].bound * scale
        achieved = validation.achieved_envelope[dof]
        half_range = max(abs(achieved["min"]), abs(achieved["max"])) * scale
        status = "✓" if validation.per_dof_pass[dof] else "✗"
        print(
            f"{pad}{dof:6}  {target_bound:>12.4f}  {half_range:>18.4f}  "
            f"{status:>6}  {unit}"
        )


# ── Chain construction ─────────────────────────────────────────────────────────

def _free_tol(dist: str, bound_t: float, bound_r: float) -> ToleranceSpec6:
    """Build a symmetric ToleranceSpec6 with equal bounds per axis type."""
    t = ToleranceSpec(dist, bound=bound_t)
    r = ToleranceSpec(dist, bound=bound_r)
    return ToleranceSpec6(dx=t, dy=t, dz=t, rx=r, ry=r, rz=r)


def build_chain(initial_bound_t: float = 0.002, initial_bound_r: float = 0.010) -> FrameGraph:
    """Build the wafer inspection gantry chain with loose initial tolerances.

    initial_bound_t / initial_bound_r are the starting tolerances for the
    unconstrained-baseline FK run.  AllocationEngine.allocate() replaces these
    with the allocated values — they are not carried into the allocation.
    """
    fg = FrameGraph()
    for name in ("wafer_chuck", "stage", "sensor_head"):
        fg.add_frame(name)

    # Edge 1: wafer_chuck → stage  (300 mm vertical lift, uniform manufacturing tol)
    fg.add_edge(
        "wafer_chuck", "stage",
        HTM.from_xyz_euler([0.0, 0.0, 0.300], [0.0, 0.0, 0.0]),
        _free_tol("uniform", initial_bound_t, initial_bound_r),
        name="stage_mount",
    )

    # Edge 2: stage → sensor_head  (500 mm lateral arm, tighter machined joint)
    fg.add_edge(
        "stage", "sensor_head",
        HTM.from_xyz_euler([0.500, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _free_tol("uniform", initial_bound_t, initial_bound_r),
        name="sensor_arm",
    )

    return fg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Unconstrained baseline
# ══════════════════════════════════════════════════════════════════════════════

section("TolTransform — Inverse Allocation Example: Wafer Inspection Gantry")

print("""
Chain:  wafer_chuck  ──(stage_mount, z=300 mm)──►  stage  ──(sensor_arm, x=500 mm)──►  sensor_head

The 500 mm lateral sensor arm creates a lever-arm: tilt at stage_mount
maps to large lateral displacement at sensor_head.

Target: sensor_head must land within ±0.1 mm translation, ±0.001 rad rotation
        relative to wafer_chuck.
""")

section("Section 1 — Unconstrained Baseline (loose ±2 mm / ±10 mrad initial tolerances)")

fg = build_chain()

N_FK = 20_000
SEED = 42

print(f"Running {N_FK:,}-trial MC with loose initial tolerances (seed={SEED}) ...")
td_loose = MonteCarloFKEngine.run(fg, n_trials=N_FK, seed=SEED)

print("\nEnvelope at sensor_head with loose tolerances:")
loose_env = frame_envelope_box(td_loose, "sensor_head")
print_envelope(loose_env)

print("""
  The unconstrained envelope spans ±mm-level translation and ±mrad-level rotation.
  We now back-solve: what tolerance per joint is needed to reach ±0.1 mm / ±0.001 rad?
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Inverse allocation (achievable target)
# ══════════════════════════════════════════════════════════════════════════════

section("Section 2 — Inverse Allocation: Target ±0.1 mm / ±0.001 rad")

t_trans = ToleranceSpec("uniform", bound=0.0001)   # ±0.1 mm
t_rot   = ToleranceSpec("uniform", bound=0.001)    # ±1 mrad
target = ToleranceSpec6(
    dx=t_trans, dy=t_trans, dz=t_trans,
    rx=t_rot,   ry=t_rot,   rz=t_rot,
)

N_VALIDATE = 2_000

print(f"Calling AllocationEngine.allocate() with n_validate={N_VALIDATE} ...")
result = AllocationEngine.allocate(
    fg,
    "wafer_chuck",
    "sensor_head",
    target,
    n_validate=N_VALIDATE,
    seed=SEED,
)

print(f"\n  converged      : {result.converged}")
print(f"  iterations_used: {result.iterations_used}")
if result.status_message:
    print(f"  status_message : {result.status_message}")

print("""
  ── What 'iterations_used' tells you ───────────────────────────────────────
  0  iterations: the first-order Jacobian allocation passed MC validation on
     the first try.  baseline_linear_allocation == corrected_allocation.

  N>0 iterations: the Jacobian-derived allocation was too optimistic — the
     lever-arm coupling caused the MC envelope to exceed the target.  The
     damping loop multiplied all angular bounds by gamma=0.9 each iteration
     until MC passed.  corrected_allocation has tighter angular bounds than
     baseline_linear_allocation.  This delta is the cost of the nonlinearity.
""")

print("\nBaseline (linear) vs Corrected (MC-validated) allocation per edge/DoF:")
print_allocation_comparison(result.baseline_linear_allocation, result.corrected_allocation)

print("""
  ── Interpreting the comparison table ──────────────────────────────────────
  Baseline (linear): what a first-order Jacobian inversion alone prescribes.
    It minimises output variance under a linearity assumption.

  Corrected (MC):    what actually passes MC validation.  Angular DoF in the
    'sensor_arm' and 'stage_mount' edges may be tighter than the linear
    estimate because a small tilt of the stage propagates through the 500 mm
    arm, causing lateral displacement at sensor_head.  The Jacobian at nominal
    (zero tilt) underestimates this coupling for non-infinitesimal rotations.
    Each Δ% in the last column is the damping loop's correction — zero means
    the linear estimate was already sufficient for that DoF.
""")

print("\nFinal MC validation — achieved envelope vs target:")
print_validation(result.final_validation_report, target)
print(f"\n  Overall pass: {result.final_validation_report.passed}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Non-convergence: locked imprecise component
# ══════════════════════════════════════════════════════════════════════════════

section("Section 3 — Non-Convergence: Locked Imprecise Component")

print("""
Engineering scenario: the stage_mount joint is an existing off-the-shelf
gantry stage with ±2 mm / ±10 mrad positioning accuracy.  It cannot be
replaced (locked=True for all DoF).  The only free joint is sensor_arm.

Question: can we allocate sensor_arm tightly enough to bring the combined
sensor_head error within ±0.1 mm / ±1 mrad?

Answer: No — the locked stage_mount contributes ±2 mm regardless of how
tight sensor_arm is.  AllocationEngine.allocate() confirms this.
""")

fg_locked = FrameGraph()
for name in ("wafer_chuck", "stage", "sensor_head"):
    fg_locked.add_frame(name)

# Edge 1: stage_mount — off-the-shelf stage, all DoF locked at ±2 mm / ±10 mrad.
# locked=True excludes this edge from the allocation Jacobian, but
# MonteCarloFKEngine still samples from its ±2 mm bounds during validation.
locked_t = ToleranceSpec("uniform", bound=0.002, locked=True)
locked_r = ToleranceSpec("uniform", bound=0.010, locked=True)
tol_locked = ToleranceSpec6(
    dx=locked_t, dy=locked_t, dz=locked_t,
    rx=locked_r, ry=locked_r, rz=locked_r,
)
fg_locked.add_edge(
    "wafer_chuck", "stage",
    HTM.from_xyz_euler([0.0, 0.0, 0.300], [0.0, 0.0, 0.0]),
    tol_locked,
    name="stage_mount",
)

# Edge 2: sensor_arm — free (the only joint we can tighten)
fg_locked.add_edge(
    "stage", "sensor_head",
    HTM.from_xyz_euler([0.500, 0.0, 0.0], [0.0, 0.0, 0.0]),
    _free_tol("uniform", 0.002, 0.010),
    name="sensor_arm",
)

print(f"Calling AllocationEngine.allocate() with max_iter=10, n_validate={N_VALIDATE} ...")
result_inf = AllocationEngine.allocate(
    fg_locked,
    "wafer_chuck",
    "sensor_head",
    target,
    n_validate=N_VALIDATE,
    seed=SEED,
)

print(f"\n  converged      : {result_inf.converged}")
print(f"  iterations_used: {result_inf.iterations_used}")
print(f"  status_message : {result_inf.status_message!r}")

print("\nAllocation for the one free edge (sensor_arm) — baseline vs corrected:")
print_allocation_comparison(result_inf.baseline_linear_allocation, result_inf.corrected_allocation)

print("""
  Note: stage_mount is locked so it does not appear in the allocation table.
  The corrected angular bounds on sensor_arm shrink toward zero each iteration
  (gamma=0.9 per iteration), but position errors from stage_mount's ±2 mm
  dominate the MC envelope regardless of how tight sensor_arm's angular DoF are.
""")

print("Achieved envelope after 10 damping iterations vs ±0.1 mm / ±1 mrad target:")
print_validation(result_inf.final_validation_report, target)

print("""
  ── Non-convergence as a diagnostic ────────────────────────────────────────
  converged=False does not indicate a software bug — it is a design signal.
  It means the target envelope cannot be met given the current chain structure
  and the set of free (adjustable) DoF.

  In this case the root cause is explicit: stage_mount's locked ±2 mm
  contribution dominates the output, and the damping loop (which only tightens
  angular bounds on free edges) cannot change that.

  Engineering options when allocation fails to converge:

    1.  REPLACE the imprecise component — swap the gantry stage for a
        higher-accuracy unit and un-lock its DoF, then re-run allocation.

    2.  RESTRUCTURE — add a fine-adjustment stage between the coarse
        gantry and the sensor arm so the coarse error can be compensated.

    3.  RELAX the target — accept that ±0.1 mm is unachievable with this
        hardware and negotiate a looser inspection specification.

  The corrected_allocation in the result holds the best allocation the damping
  loop could find in max_iter steps.  Comparing the achieved envelope to the
  target (above) quantifies exactly how far short the chain falls — useful for
  making the cost/benefit case for a hardware upgrade.
""")

section("Done")
print("Run `python examples/allocation_example.py` from the repo root to re-execute.\n")
