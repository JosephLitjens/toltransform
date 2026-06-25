#!/usr/bin/env python3
"""
Multi-chain shared-frame example: optical bench with two lenses.

Two kinematic chains share a common upstream frame (the optical bench).
Demonstrates that the RELATIVE tolerance between the two lens seats is
tighter than either lens's absolute tolerance — because the shared bench-flex
errors cancel out when computing the relative transform between the two branches.

Graph topology:

                     ┌──[lens_a_mount]──► lens_a
  world ──[bench_mount]──► optical_bench ─┤
                     └──[lens_b_mount]───► lens_b

Run from the repo root:
    python examples/multi_chain_shared_frame_example.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.frame_graph import FrameGraph
from core.transforms import HTM
from core.tolerance import ToleranceSpec, ToleranceSpec6
from sim.monte_carlo_fk import MonteCarloFKEngine
from postprocess.stats import frame_envelope_box, point_pair_envelope_box, relative_pose_nominal


# ── output helpers ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def print_envelope(env: dict) -> None:
    print(f"  {'DoF':<6}  {'min':>12}  {'max':>12}  {'range':>12}")
    print("  " + "-" * 50)
    for dof, bounds in env.items():
        lo, hi = bounds["min"], bounds["max"]
        print(f"  {dof:<6}  {lo:>12.4f}  {hi:>12.4f}  {hi - lo:>12.4f}")


# ── scenario description ──────────────────────────────────────────────────────

section("TolTransform — Multi-Chain Shared-Frame Example: Optical Bench")

print("""
Scenario
--------
Two precision lenses (lens_a, lens_b) are each bolted into their own kinematic
seat, which in turn are both mounted on a common optical bench (optical_bench).
The bench itself is mounted to the world frame via an isolation mount with
measurable flex.

  world ──[bench_mount]──► optical_bench ──[lens_a_mount]──► lens_a
                                         └──[lens_b_mount]──► lens_b

Key question: given the bench-flex tolerance (large, ±0.1 mm), what is the
relative position uncertainty between lens_a and lens_b?

Intuition: if the bench flexes by +0.05 mm in X, BOTH lenses move by +0.05 mm.
Their relative position is unchanged. The only residual relative error comes
from the two independent seat tolerances (small, ±0.02 mm each).

TolTransform verifies this cancellation numerically by computing the relative
transform in each Monte Carlo trial, using the same sampled bench perturbation
for both paths through the shared ancestor frame.

Units: mm for translations, rad for rotations.
""")

# ── build the frame graph ─────────────────────────────────────────────────────

fg = FrameGraph()
fg.add_frame("world")
fg.add_frame("optical_bench")
fg.add_frame("lens_a")
fg.add_frame("lens_b")

# Edge 1: world → optical_bench
# The optical bench is mounted on a vibration-isolation table. The dominant
# error is bench flex (tip/tilt and lateral drift) modelled as uniform worst-case
# bounds. These are large compared to the seat tolerances below.
T_bench = HTM.from_xyz_euler([0.0, 0.0, 100.0], [0.0, 0.0, 0.0])
tol_bench = ToleranceSpec6(
    dx=ToleranceSpec("uniform", bound=0.100),   # ±0.100 mm bench lateral flex
    dy=ToleranceSpec("uniform", bound=0.100),
    dz=ToleranceSpec("uniform", bound=0.050),   # ±0.050 mm axial bench compression
    rx=ToleranceSpec("uniform", bound=0.002),   # ±0.002 rad bench tip
    ry=ToleranceSpec("uniform", bound=0.002),
    rz=ToleranceSpec("uniform", bound=0.001),   # ±0.001 rad bench yaw
)
fg.add_edge("world", "optical_bench", T_bench, tol_bench, name="bench_mount")

# Edge 2: optical_bench → lens_a
# Lens A sits 50 mm to the left of the bench centre. The seat kinematic coupling
# (three-groove vee-flat) has tight repeatable position errors, modelled as normal.
T_lens_a = HTM.from_xyz_euler([-50.0, 0.0, 0.0], [0.0, 0.0, 0.0])
tol_lens_a = ToleranceSpec6(
    dx=ToleranceSpec("normal",  bound=0.020),   # ±0.020 mm seat repeatability @ 3σ
    dy=ToleranceSpec("normal",  bound=0.020),
    dz=ToleranceSpec("normal",  bound=0.010),   # ±0.010 mm axial seat height @ 3σ
    rx=ToleranceSpec("normal",  bound=0.0005),  # ±0.0005 rad seat tilt @ 3σ
    ry=ToleranceSpec("normal",  bound=0.0005),
    rz=ToleranceSpec("uniform", bound=0.001),   # ±0.001 rad rotational seat play
)
fg.add_edge("optical_bench", "lens_a", T_lens_a, tol_lens_a, name="lens_a_mount")

# Edge 3: optical_bench → lens_b
# Lens B sits 50 mm to the right, identical seat design. The key point:
# lens_a_mount and lens_b_mount are INDEPENDENT — their errors do NOT cancel.
# Only the shared bench_mount error cancels in the relative transform.
T_lens_b = HTM.from_xyz_euler([+50.0, 0.0, 0.0], [0.0, 0.0, 0.0])
tol_lens_b = ToleranceSpec6(
    dx=ToleranceSpec("normal",  bound=0.020),
    dy=ToleranceSpec("normal",  bound=0.020),
    dz=ToleranceSpec("normal",  bound=0.010),
    rx=ToleranceSpec("normal",  bound=0.0005),
    ry=ToleranceSpec("normal",  bound=0.0005),
    rz=ToleranceSpec("uniform", bound=0.001),
)
fg.add_edge("optical_bench", "lens_b", T_lens_b, tol_lens_b, name="lens_b_mount")

print("Graph built:")
print(f"  {len(fg.all_frames())} frames: {[f.name for f in fg.all_frames()]}")
for edge in fg.all_edges():
    print(f"  edge '{edge.name}':  {edge.parent} → {edge.child}")

# ── nominal relative transform ────────────────────────────────────────────────

section("Nominal Transform: lens_a → lens_b")
T_nom_ab = fg.nominal_transform_between("lens_a", "lens_b")
xyz_ab, euler_ab = T_nom_ab.to_xyz_euler()
print(f"  Translation (mm) :  x={xyz_ab[0]:8.4f}  y={xyz_ab[1]:8.4f}  z={xyz_ab[2]:8.4f}")
print(f"  Euler ZYX (rad)  :  z={euler_ab[0]:9.6f}  y={euler_ab[1]:9.6f}  x={euler_ab[2]:9.6f}")
print(f"\n  (Nominal: lens_b is 100 mm to the right of lens_a, both at the same Z height.)")

# ── Monte Carlo run ───────────────────────────────────────────────────────────

N_TRIALS = 50_000
SEED = 42

section(f"Monte Carlo FK Run  (n_trials={N_TRIALS:,}, seed={SEED})")
print(f"  Note: optical_bench perturbation is sampled ONCE per trial and applied")
print(f"  to BOTH downstream chains — this is what enables the cancellation.")
trial_data = MonteCarloFKEngine.run(fg, n_trials=N_TRIALS, seed=SEED)
print(f"  Done. Stored {len(trial_data.frame_poses)} frame pose arrays.")

# ── absolute envelope for each lens ──────────────────────────────────────────

section("Absolute Envelope: lens_a (relative to world)  [mm / rad]")
print("  Includes bench_mount error + lens_a_mount error — bench dominates.\n")
env_a = frame_envelope_box(trial_data, "lens_a")
print_envelope(env_a)

section("Absolute Envelope: lens_b (relative to world)  [mm / rad]")
print("  Includes bench_mount error + lens_b_mount error — bench dominates.\n")
env_b = frame_envelope_box(trial_data, "lens_b")
print_envelope(env_b)

# ── relative envelope between the two lenses ──────────────────────────────────

section("Relative Envelope: lens_a → lens_b  [mm / rad]")
print("  This is the position uncertainty of lens_b as seen FROM lens_a.\n")
env_rel = point_pair_envelope_box(trial_data, fg, "lens_a", "lens_b")
print_envelope(env_rel)

# ── cancellation explanation ──────────────────────────────────────────────────

section("Shared-Ancestor Cancellation — Explained")

dx_abs_a = env_a["dx"]["max"] - env_a["dx"]["min"]
dx_abs_b = env_b["dx"]["max"] - env_b["dx"]["min"]
dx_rel   = env_rel["dx"]["max"] - env_rel["dx"]["min"]
bench_bound = 0.100  # mm, the bench_mount dx uniform bound

print(f"""
  Absolute dx range of lens_a (vs world) : {dx_abs_a:.4f} mm
  Absolute dx range of lens_b (vs world) : {dx_abs_b:.4f} mm
  Relative dx range between the two      : {dx_rel:.4f} mm   ← much tighter!

  Why the relative tolerance is tighter
  ──────────────────────────────────────
  The path from lens_a to lens_b in the frame graph goes:

    lens_a ←[lens_a_mount inverse]── optical_bench ──[lens_b_mount]──► lens_b

  It traverses the bench_mount edge TWICE — once backward (from lens_a up to
  optical_bench) and once implicitly cancelled (the bench pose used to reach
  lens_b is the same bench pose used to reach lens_a in that trial).

  Concretely: if the bench flexes +{bench_bound:.3f} mm in X during a trial, then:
    - lens_a shifts by +{bench_bound:.3f} mm (absolute)
    - lens_b shifts by +{bench_bound:.3f} mm (absolute)
    - lens_b relative to lens_a: unchanged (net bench contribution = 0)

  The residual relative uncertainty comes only from the two INDEPENDENT seat
  tolerances (lens_a_mount and lens_b_mount), each ±0.020 mm @ 3σ. In the
  worst case these add (RSS ≈ {(0.020**2 + 0.020**2)**0.5:.4f} mm); that is what
  you see in the relative envelope above — far smaller than the ±{bench_bound:.3f} mm
  bench flex that dominates the absolute envelopes.

  Design implication
  ──────────────────
  If your optical system only cares about the RELATIVE position of the two lenses
  (e.g., inter-lens spacing and co-alignment), you do NOT need to control bench
  flex to ±0.020 mm — you only need to control the seat repeatability to that
  level. The bench can flex freely; it moves both lenses together.

  If, however, your system has an EXTERNAL reference (e.g., a detector fixed
  to world, not to the bench), then the bench flex enters the absolute path to
  each lens and you do need the tighter bench spec.

  TolTransform lets you evaluate both scenarios within a single Monte Carlo run
  by choosing which frame pair to compute the relative transform between.
""")
