#!/usr/bin/env python3
"""
Multi-chain shared-frame example: optical bench with two lenses.

Four scenarios sweep the lens-seat tolerance from zero to full to isolate:

  Scenario 0 — Zero lens tolerances  : proves bench errors cancel exactly (0 relative error)
  Scenario 1 — Translational only    : direct seat-translation contribution
  Scenario 2 — Rotational only       : reveals lever-arm amplification (the key surprise)
  Scenario 3 — Full (trans + rot)    : the realistic combined case

Graph topology:

                     ┌──[lens_a_mount]──► lens_a
  world ──[bench_mount]──► optical_bench ─┤
                     └──[lens_b_mount]───► lens_b

The lenses are separated by 100 mm in X (lens_a at −50 mm, lens_b at +50 mm
relative to the bench centre). This 100 mm separation is the lever arm that
amplifies rotational seat errors into unexpectedly large relative translation errors.

Run from the repo root:
    python examples/multi_chain_shared_frame_example.py
"""
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.frame_graph import FrameGraph
from core.transforms import HTM
from core.tolerance import ToleranceSpec, ToleranceSpec6
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData
from postprocess.stats import frame_envelope_box, point_pair_envelope_box


# ── run parameters ────────────────────────────────────────────────────────────

N_TRIALS = 50_000
SEED = 42

# ── shared nominal geometry ───────────────────────────────────────────────────

LENS_SEPARATION_MM = 100.0  # centre-to-centre distance between lens seats in X

T_BENCH  = HTM.from_xyz_euler([0.0,  0.0, 100.0], [0.0, 0.0, 0.0])
T_LENS_A = HTM.from_xyz_euler([-50.0, 0.0, 0.0],  [0.0, 0.0, 0.0])
T_LENS_B = HTM.from_xyz_euler([+50.0, 0.0, 0.0],  [0.0, 0.0, 0.0])

# ── bench tolerances — identical across all four scenarios ────────────────────

BENCH_DX_BOUND = 0.100   # mm, uniform ±bound
BENCH_RY_BOUND = 0.002   # rad, uniform ±bound
BENCH_RZ_BOUND = 0.001   # rad, uniform ±bound

BENCH_TOL = ToleranceSpec6(
    dx=ToleranceSpec("uniform", bound=BENCH_DX_BOUND),   # ±0.100 mm lateral flex
    dy=ToleranceSpec("uniform", bound=BENCH_DX_BOUND),
    dz=ToleranceSpec("uniform", bound=0.050),            # ±0.050 mm axial compression
    rx=ToleranceSpec("uniform", bound=BENCH_RY_BOUND),   # ±0.002 rad tip
    ry=ToleranceSpec("uniform", bound=BENCH_RY_BOUND),
    rz=ToleranceSpec("uniform", bound=BENCH_RZ_BOUND),   # ±0.001 rad yaw
)

# ── canonical lens-seat bounds (used when scenario activates each DoF type) ───

SEAT_TRANS_BOUND  = 0.020   # mm, ±3σ (normal)  — dx, dy, dz
SEAT_ROT_XY_BOUND = 0.0005  # rad, ±3σ (normal) — rx, ry
SEAT_ROT_Z_BOUND  = 0.001   # rad, uniform       — rz (rotational play in the seat socket)

ZERO = ToleranceSpec("uniform", bound=0.0)


# ── helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def print_envelope(env: dict, indent: str = "  ") -> None:
    print(f"{indent}{'DoF':<6}  {'min':>12}  {'max':>12}  {'range':>12}")
    print(indent + "-" * 48)
    for dof, bounds in env.items():
        lo, hi = bounds["min"], bounds["max"]
        print(f"{indent}{dof:<6}  {lo:>12.4f}  {hi:>12.4f}  {hi - lo:>12.4f}")


def make_lens_tol(trans: bool, rot: bool) -> ToleranceSpec6:
    """Return a ToleranceSpec6 with translational/rotational errors selectively active."""
    t    = ToleranceSpec("normal",  bound=SEAT_TRANS_BOUND)  if trans else ZERO
    r_xy = ToleranceSpec("normal",  bound=SEAT_ROT_XY_BOUND) if rot   else ZERO
    r_z  = ToleranceSpec("uniform", bound=SEAT_ROT_Z_BOUND)  if rot   else ZERO
    return ToleranceSpec6(dx=t, dy=t, dz=t, rx=r_xy, ry=r_xy, rz=r_z)


def build_and_run(lens_tol: ToleranceSpec6) -> tuple[TrialData, FrameGraph]:
    """Build a fresh graph with the given lens-seat tolerances and run the MC engine."""
    fg = FrameGraph()
    for name in ("world", "optical_bench", "lens_a", "lens_b"):
        fg.add_frame(name)
    fg.add_edge("world",         "optical_bench", T_BENCH,  BENCH_TOL, name="bench_mount")
    fg.add_edge("optical_bench", "lens_a",        T_LENS_A, lens_tol,  name="lens_a_mount")
    fg.add_edge("optical_bench", "lens_b",        T_LENS_B, lens_tol,  name="lens_b_mount")
    return MonteCarloFKEngine.run(fg, N_TRIALS, SEED), fg


# ── scenario description ──────────────────────────────────────────────────────

section("TolTransform — Multi-Chain Shared-Frame Example: Optical Bench")

print(f"""
Scenario
--------
Two identical precision lenses share a common optical bench. The bench has
large flex tolerances (±{BENCH_DX_BOUND:.3f} mm lateral, ±{BENCH_RY_BOUND:.3f} rad tip/tilt).
Each lens is mounted in a kinematic seat (three-groove vee-flat style) bolted
to the bench, with tighter seat tolerances.

  world ──[bench_mount]──► optical_bench ──[lens_a_mount]──► lens_a
                                         └──[lens_b_mount]──► lens_b

The lenses are {LENS_SEPARATION_MM:.0f} mm apart in X (lens_a at −50 mm, lens_b at +50 mm
relative to the bench centre). This separation is the lever arm that converts
angular seat errors into surprisingly large relative translation errors.

Key claim: bench-flex errors cancel in the relative (lens_a → lens_b) transform
because both lenses ride the same bench. Only the independent seat errors survive.

This example runs four scenarios to decompose that claim:
  Scenario 0 — Zero seat tolerances    : prove bench cancels → 0 relative error
  Scenario 1 — Translational only      : just the seat-translation contribution
  Scenario 2 — Rotational only         : lever-arm effect of seat angular errors
  Scenario 3 — Full (trans + rotation) : the realistic combined case

Bench tolerances (same for all scenarios):
  dx/dy = ±{BENCH_DX_BOUND:.3f} mm (uniform),   dz = ±0.050 mm (uniform)
  rx/ry = ±{BENCH_RY_BOUND:.3f} rad (uniform),  rz = ±{BENCH_RZ_BOUND:.3f} rad (uniform)

Canonical seat tolerances (activated per-scenario):
  dx/dy/dz = ±{SEAT_TRANS_BOUND:.3f} mm @ 3σ (normal),  rx/ry = ±{SEAT_ROT_XY_BOUND:.4f} rad @ 3σ (normal)
  rz       = ±{SEAT_ROT_Z_BOUND:.3f} rad (uniform)

{N_TRIALS:,} trials, seed={SEED}. Units: mm (translations), rad (rotations).
""")

# ── print nominal ─────────────────────────────────────────────────────────────

# Just use a fresh graph for the nominal — no MC needed.
_fg_nom = FrameGraph()
for _n in ("world", "optical_bench", "lens_a", "lens_b"):
    _fg_nom.add_frame(_n)
_fg_nom.add_edge("world", "optical_bench", T_BENCH, make_lens_tol(False, False), name="bench_mount")
_fg_nom.add_edge("optical_bench", "lens_a", T_LENS_A, make_lens_tol(False, False), name="lens_a_mount")
_fg_nom.add_edge("optical_bench", "lens_b", T_LENS_B, make_lens_tol(False, False), name="lens_b_mount")

T_nom_ab = _fg_nom.nominal_transform_between("lens_a", "lens_b")
xyz_ab, _ = T_nom_ab.to_xyz_euler()
print(f"  Nominal lens_a → lens_b translation:  "
      f"x={xyz_ab[0]:.1f} mm, y={xyz_ab[1]:.1f} mm, z={xyz_ab[2]:.1f} mm")
print(f"  (lens_b is exactly {LENS_SEPARATION_MM:.0f} mm to the +X side of lens_a, same height)\n")


# ── SCENARIO 0: zero lens tolerances ─────────────────────────────────────────

section("Scenario 0 — Zero Seat Tolerances  (bench errors cancel → ≈0 relative)")

print("""
  With zero seat tolerances, every trial has lens_a and lens_b displaced by
  exactly the same bench perturbation. The relative transform should be
  identically zero to floating-point precision.

  Absolute envelope for lens_a (dominated by bench):
""")
td0, fg0 = build_and_run(make_lens_tol(False, False))
env0_abs = frame_envelope_box(td0, "lens_a")
print_envelope(env0_abs, indent="    ")

print("\n  Relative envelope lens_a → lens_b (should be ≈0):\n")
env0_rel = point_pair_envelope_box(td0, fg0, "lens_a", "lens_b")
print_envelope(env0_rel, indent="    ")

max_abs_rel = max(abs(v) for d in env0_rel.values() for v in (d["min"], d["max"]))
print(f"\n  Max absolute value in relative envelope: {max_abs_rel:.2e}  "
      f"({'≈ 0 ✓' if max_abs_rel < 1e-8 else 'non-zero — check implementation'})")


# ── SCENARIO 1: translational seat errors only ────────────────────────────────

section("Scenario 1 — Translational Seat Errors Only")

print(f"""
  Active seat tolerances: dx/dy/dz = ±{SEAT_TRANS_BOUND:.3f} mm @ 3σ (normal).
  Rotational seat tolerances: all zero.

  Because the seat translational errors are INDEPENDENT between the two seats,
  they do NOT cancel. The relative dx error is approximately RSS of the two seats:
    1-sigma each  = {SEAT_TRANS_BOUND:.3f}/3 = {SEAT_TRANS_BOUND/3:.5f} mm
    1-sigma of (dx_b − dx_a)  = √2 × {SEAT_TRANS_BOUND/3:.5f} = {math.sqrt(2)*SEAT_TRANS_BOUND/3:.5f} mm
  (The MC envelope shows the hard min/max across all trials, which exceeds the
   99th-percentile; expect the envelope range to be somewhat wider than 2 × 3σ.)

  There is NO lever-arm amplification when the seat offset is purely translational:
  a dx error on the seat shifts the lens by the same dx regardless of separation.
  So dy and dz from the rotational coupling are zero here.
""")
td1, fg1 = build_and_run(make_lens_tol(True, False))
env1_rel = point_pair_envelope_box(td1, fg1, "lens_a", "lens_b")
print_envelope(env1_rel, indent="  ")


# ── SCENARIO 2: rotational seat errors only ───────────────────────────────────

section("Scenario 2 — Rotational Seat Errors Only  (lever-arm amplification)")

print(f"""
  Active seat tolerances: rx/ry = ±{SEAT_ROT_XY_BOUND:.4f} rad @ 3σ (normal),
                          rz    = ±{SEAT_ROT_Z_BOUND:.3f} rad (uniform).
  Translational seat tolerances: all zero.

  This is the key scenario. Even though the rotational bounds are tiny, they
  produce surprisingly large TRANSLATION errors in the relative envelope.

  Why? The two lenses are {LENS_SEPARATION_MM:.0f} mm apart in X. A small angular error on
  one lens seat is seen, from the other lens, as a tilt PLUS a positional offset
  scaled by the {LENS_SEPARATION_MM:.0f} mm separation (the lever arm):

    rz on lens_a   →  relative dy ≈ {LENS_SEPARATION_MM:.0f} mm × rz
      At rz = ±{SEAT_ROT_Z_BOUND:.3f} rad:  relative dy contribution ≈ ±{LENS_SEPARATION_MM * SEAT_ROT_Z_BOUND:.1f} mm per seat
      Both seats are independent, so worst-case adds: ±{2 * LENS_SEPARATION_MM * SEAT_ROT_Z_BOUND:.1f} mm range

    ry on lens_a   →  relative dz ≈ {LENS_SEPARATION_MM:.0f} mm × ry
      At ry = ±{SEAT_ROT_XY_BOUND:.4f} rad (3σ):  relative dz contribution ≈ ±{LENS_SEPARATION_MM * SEAT_ROT_XY_BOUND:.2f} mm per seat

  Compare this to Scenario 1 (translational only):
    dx seat error of ±{SEAT_TRANS_BOUND:.3f} mm contributes only ±{SEAT_TRANS_BOUND:.3f} mm (no amplification).

  So rotational seat errors can DOMINATE the relative translation budget when the
  separation is large, even if the angular bounds look small in isolation.
""")
td2, fg2 = build_and_run(make_lens_tol(False, True))
env2_rel = point_pair_envelope_box(td2, fg2, "lens_a", "lens_b")
print_envelope(env2_rel, indent="  ")


# ── SCENARIO 3: full tolerances ────────────────────────────────────────────────

section("Scenario 3 — Full Tolerances  (translation + rotation combined)")

print(f"""
  Active seat tolerances: all of Scenario 1 + Scenario 2 simultaneously.
    dx/dy/dz = ±{SEAT_TRANS_BOUND:.3f} mm @ 3σ (normal)
    rx/ry    = ±{SEAT_ROT_XY_BOUND:.4f} rad @ 3σ (normal)
    rz       = ±{SEAT_ROT_Z_BOUND:.3f} rad (uniform)

  The rotational and translational effects add (they are independent), so the
  full-tolerance relative envelope is approximately the RSS of Scenarios 1 and 2.
  Expect dx to be slightly larger than Scenario 1 (the rz effect on dx is small),
  but dy and dz to be dominated by the Scenario 2 rotational lever-arm terms.
""")
td3, fg3 = build_and_run(make_lens_tol(True, True))
env3_abs = frame_envelope_box(td3, "lens_a")
env3_rel = point_pair_envelope_box(td3, fg3, "lens_a", "lens_b")

print("  Absolute envelope — lens_a vs world  (bench-dominated; rotational seat adds little):\n")
print_envelope(env3_abs, indent="    ")
print("\n  Relative envelope — lens_a → lens_b  (seat errors only; bench cancels):\n")
print_envelope(env3_rel, indent="    ")


# ── summary comparison table ──────────────────────────────────────────────────

section("Summary — Relative Envelope RANGES by Scenario  [mm translations, rad rotations]")

print("""
  Each cell = (max − min) of that DoF's relative error over all 50 k trials.
  The bench errors are identical across all rows — any difference is due to the
  seat tolerances alone.
""")

envs_rel = {
    "Scen 0  (zero)    ": env0_rel,
    "Scen 1  (trans)   ": env1_rel,
    "Scen 2  (rot)     ": env2_rel,
    "Scen 3  (full)    ": env3_rel,
}

dofs = list(env0_rel.keys())
col_w = 10
label_w = 22

header = f"  {'':>{label_w}}" + "".join(f"  {d:>{col_w}}" for d in dofs)
print(header)
print("  " + "-" * (label_w + (col_w + 2) * len(dofs)))

for lbl, env in envs_rel.items():
    row = f"  {lbl:>{label_w}}"
    for dof in dofs:
        rng = env[dof]["max"] - env[dof]["min"]
        row += f"  {rng:>{col_w}.4f}"
    print(row)


# ── interpretation ────────────────────────────────────────────────────────────

section("Interpretation and Design Takeaways")

dx0 = env0_rel["dx"]["max"] - env0_rel["dx"]["min"]
dx1 = env1_rel["dx"]["max"] - env1_rel["dx"]["min"]
dy2 = env2_rel["dy"]["max"] - env2_rel["dy"]["min"]
dz2 = env2_rel["dz"]["max"] - env2_rel["dz"]["min"]
dy3 = env3_rel["dy"]["max"] - env3_rel["dy"]["min"]

print(f"""
  1. Bench cancellation is exact (Scenario 0):
     Relative dx/dy/dz/rx/ry/rz ≈ 0 ({dx0:.1e} mm max) despite large bench flex.
     The shared ancestor's perturbation is identical on both branches in every trial.

  2. Translational seat errors are "what you'd expect" (Scenario 1):
     Relative dx range = {dx1:.4f} mm ≈ RSS of two ±{SEAT_TRANS_BOUND:.3f}mm seats × ~4σ envelope factor.
     No lever-arm: the 100 mm separation does not amplify a pure translation error.

  3. Rotational seat errors are the dominant budget item (Scenario 2):
     Relative dy range = {dy2:.4f} mm (from rz ±{SEAT_ROT_Z_BOUND:.3f}rad × {LENS_SEPARATION_MM:.0f}mm lever arm)
     Relative dz range = {dz2:.4f} mm (from ry ±{SEAT_ROT_XY_BOUND:.4f}rad × {LENS_SEPARATION_MM:.0f}mm lever arm)
     These are MUCH larger than the translational budget from Scenario 1 — even
     though the rotational bounds look small in isolation (sub-milliradian).

  4. Full case (Scenario 3) is dominated by rotation (dy, dz ≈ Scenario 2):
     Relative dy range = {dy3:.4f} mm — rotational lever arm overwhelms the seat translation.

  Design rule of thumb:
     When two frames share a common ancestor and are separated by a distance L,
     a seat angular error of δθ contributes approximately L × δθ to their relative
     translation. At L = {LENS_SEPARATION_MM:.0f} mm, even a {SEAT_ROT_Z_BOUND:.3f} rad rz error produces
     a {LENS_SEPARATION_MM * SEAT_ROT_Z_BOUND:.1f} mm relative translation — identical to a 0.1 mm translational error.
     Tight angular seat specs matter more as the inter-frame separation grows.
""")
