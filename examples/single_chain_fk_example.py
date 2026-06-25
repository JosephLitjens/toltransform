#!/usr/bin/env python3
"""
Single-chain FK example: CNC spindle alignment stack-up.

Three-edge serial chain from world → spindle_housing → bearing_seat → tool_tip,
demonstrating a mix of uniform (worst-case) and normal (statistical) tolerances.
Runs MonteCarloFKEngine and prints the envelope box and percentile table for the
tool_tip frame — text output only (no plotting dependencies).

Run from the repo root:
    python examples/single_chain_fk_example.py
"""
from pathlib import Path
import sys

# Allow running as a standalone script from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.frame_graph import FrameGraph
from core.transforms import HTM
from core.tolerance import ToleranceSpec, ToleranceSpec6
from sim.monte_carlo_fk import MonteCarloFKEngine
from postprocess.stats import frame_envelope_box, frame_percentiles


# ── output helpers ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def print_envelope(env: dict) -> None:
    """Print a frame_envelope_box result as a formatted table."""
    print(f"  {'DoF':<6}  {'min':>12}  {'max':>12}  {'range':>12}")
    print("  " + "-" * 50)
    for dof, bounds in env.items():
        lo, hi = bounds["min"], bounds["max"]
        print(f"  {dof:<6}  {lo:>12.4f}  {hi:>12.4f}  {hi - lo:>12.4f}")


def print_percentiles(pct: dict) -> None:
    """Print a frame_percentiles result as a formatted table."""
    levels = sorted(next(iter(pct.values())).keys())
    header = f"  {'DoF':<6}" + "".join(f"  {p:>8.0f}th" for p in levels)
    print(header)
    print("  " + "-" * (8 + 11 * len(levels)))
    for dof, pmap in pct.items():
        row = f"  {dof:<6}" + "".join(f"  {pmap[p]:>9.4f}" for p in levels)
        print(row)


# ── scenario description ──────────────────────────────────────────────────────

section("TolTransform — Single-Chain FK Example: CNC Spindle Alignment")

print("""
Scenario
--------
A 3-edge serial kinematic chain representing the spindle mounting stack-up
in a 3-axis CNC machine.

  world → [mount] → spindle_housing → [bearing] → bearing_seat → [tool] → tool_tip

Each edge introduces small positional and angular errors. The goal is to bound
the worst-case and statistical position/orientation error at the tool tip
relative to the machine world frame.

Tolerance mix (illustrates both distribution types):
  mount   : uniform dx/dy/dz (machined pocket fits) + normal rx/ry (surface flatness)
  bearing : uniform dx/dy/dz (bearing clearance)    + normal rx/ry (bearing runout)
  tool    : normal  dx/dy/dz (collet runout)        + normal rx/ry + rz LOCKED (0 error)

Units: mm for translations, rad for rotations. A bound of b on a uniform
tolerance means the DoF is sampled from Uniform(−b, +b). A bound of b on a
normal tolerance means b is the ±3σ limit (sigma_level=3.0 default).
""")

# ── build the frame graph ─────────────────────────────────────────────────────

fg = FrameGraph()
fg.add_frame("world")
fg.add_frame("spindle_housing")
fg.add_frame("bearing_seat")
fg.add_frame("tool_tip")

# Edge 1: world → spindle_housing
# Represents the housing bolted to the machine column. The housing sits at the
# world origin nominally (identity transform). Pocket fits dominate lateral error;
# the face flatness governs angular seating — best modelled as normal (most
# assemblies cluster near nominal, tails from surface finish variation).
T_mount = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
tol_mount = ToleranceSpec6(
    dx=ToleranceSpec("uniform", bound=0.050),   # ±0.050 mm pocket fit
    dy=ToleranceSpec("uniform", bound=0.050),
    dz=ToleranceSpec("uniform", bound=0.020),   # ±0.020 mm axial seat depth
    rx=ToleranceSpec("normal",  bound=0.001),   # ±0.001 rad face flatness @ 3σ
    ry=ToleranceSpec("normal",  bound=0.001),
    rz=ToleranceSpec("uniform", bound=0.001),   # ±0.001 rad clocking slot play
)
fg.add_edge("world", "spindle_housing", T_mount, tol_mount, name="mount")

# Edge 2: spindle_housing → bearing_seat
# Angular contact bearing stack, 50 mm tall. Radial play is a hard
# clearance-fit uniform bound; angular misalignment (tilt introduced by bore
# runout and preload variation) is normal — most bearings seat well, with
# occasional outliers.
T_bearing = HTM.from_xyz_euler([0.0, 0.0, 50.0], [0.0, 0.0, 0.0])
tol_bearing = ToleranceSpec6(
    dx=ToleranceSpec("uniform", bound=0.020),   # ±0.020 mm radial play
    dy=ToleranceSpec("uniform", bound=0.020),
    dz=ToleranceSpec("uniform", bound=0.010),   # ±0.010 mm axial preload set
    rx=ToleranceSpec("normal",  bound=0.0005),  # ±0.0005 rad runout @ 3σ
    ry=ToleranceSpec("normal",  bound=0.0005),
    rz=ToleranceSpec("uniform", bound=0.002),   # ±0.002 rad angular runout
)
fg.add_edge("spindle_housing", "bearing_seat", T_bearing, tol_bearing, name="bearing")

# Edge 3: bearing_seat → tool_tip
# Collet + toolholder + 150 mm tool overhang. Angular errors here are amplified
# by the full tool length. rz is locked (spindle rotation is the intended motion,
# not an error source; bound=0.0 → zero contribution to FK error).
T_tool = HTM.from_xyz_euler([0.0, 0.0, 150.0], [0.0, 0.0, 0.0])
tol_tool = ToleranceSpec6(
    dx=ToleranceSpec("normal",  bound=0.010),   # ±0.010 mm collet lateral runout @ 3σ
    dy=ToleranceSpec("normal",  bound=0.010),
    dz=ToleranceSpec("normal",  bound=0.020),   # ±0.020 mm tool-length variation @ 3σ
    rx=ToleranceSpec("normal",  bound=0.0003),  # ±0.0003 rad collet tilt @ 3σ
    ry=ToleranceSpec("normal",  bound=0.0003),
    rz=ToleranceSpec("uniform", bound=0.0, locked=True),  # locked: no rz error
)
fg.add_edge("bearing_seat", "tool_tip", T_tool, tol_tool, name="tool")

print(f"Graph built: {len(fg.all_frames())} frames, {len(fg.all_edges())} edges")
for edge in fg.all_edges():
    print(f"  {edge.name:10s}  {edge.parent} → {edge.child}")

# ── nominal transform ─────────────────────────────────────────────────────────

section("Nominal Transform: world → tool_tip")
T_nom = fg.nominal_transform_between("world", "tool_tip")
xyz_nom, euler_nom = T_nom.to_xyz_euler()
print(f"  Translation (mm) :  x={xyz_nom[0]:8.4f}  y={xyz_nom[1]:8.4f}  z={xyz_nom[2]:8.4f}")
print(f"  Euler ZYX (rad)  :  z={euler_nom[0]:9.6f}  y={euler_nom[1]:9.6f}  x={euler_nom[2]:9.6f}")
print(f"\n  (Nominal: tool tip 200 mm above world origin, no angular offset.)")

# ── Monte Carlo run ───────────────────────────────────────────────────────────

N_TRIALS = 50_000
SEED = 42

section(f"Monte Carlo FK Run  (n_trials={N_TRIALS:,}, seed={SEED})")
print(f"  Sampling {len(fg.all_edges())} edges × {N_TRIALS:,} trials ...")
trial_data = MonteCarloFKEngine.run(fg, n_trials=N_TRIALS, seed=SEED)
print(f"  Done. Stored {len(trial_data.frame_poses)} frame pose arrays, shape (N,4,4) each.")

# ── worst-case envelope ───────────────────────────────────────────────────────

section("Worst-Case Envelope at tool_tip  [mm / rad]")
print("  min/max over all trials — the hard outer bound on error\n")
env = frame_envelope_box(trial_data, "tool_tip")
print_envelope(env)

# ── percentile table ──────────────────────────────────────────────────────────

section("Statistical Percentile Table at tool_tip  [mm / rad]")
print("  Values are signed deviations from nominal at the given percentile.\n")
pct = frame_percentiles(trial_data, "tool_tip", [50.0, 90.0, 95.0, 99.0])
print_percentiles(pct)

# ── interpretation ────────────────────────────────────────────────────────────

section("Interpretation")
dx_range = env["dx"]["max"] - env["dx"]["min"]
dz_range = env["dz"]["max"] - env["dz"]["min"]
rx_range = env["rx"]["max"] - env["rx"]["min"]
print(f"""
  Lateral error (dx) range : {dx_range:.4f} mm
  Axial error   (dz) range : {dz_range:.4f} mm
  Tilt error    (rx) range : {rx_range:.6f} rad

  The dominant lateral contributor is the mount edge (uniform ±0.050 mm),
  which accounts for most of the dx/dy range. However, the angular errors on
  the bearing and tool edges are amplified by the 150 mm tool overhang:
  a 0.001 rad rx tilt at the housing translates to ~0.150 mm tip displacement,
  which is comparable to the mounting pocket fit itself.

  The 99th-percentile values (from the percentile table) are substantially
  tighter than the hard-envelope values because the normal-distribution tolerances
  (mount rx/ry, bearing rx/ry, tool dx/dy/rx/ry) very rarely simultaneously reach
  their ±3σ extremes. Use the 99th-percentile as your design target when the
  tolerance budget is tight; use the hard envelope only for a safe worst-case guarantee.
""")
