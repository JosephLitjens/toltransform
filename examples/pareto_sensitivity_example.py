"""
Pareto Sensitivity Example: Precision Inspection Robot Arm.

Demonstrates the three core engineering decisions TolTransform is built to support
(Section 1.4 of the design spec):

  1. SENSITIVITY PINPOINTING — Which joint tolerance is the dominant driver of
     probe-tip error? Run the Pareto breakdown to get a ranked, quantified answer.

  2. COMPONENT SELECTION — The shoulder joint uses a cheap, loose-tolerance actuator.
     Can we justify the cost of a tighter unit? Upgrade it and re-run to see exactly
     how much the error budget improves and whether it's now the bottleneck.

  3. REPORTING — Save a multi-panel frame report and a standalone Pareto chart as PNG
     files for inclusion in a design review or sourcing discussion.

Scenario: 4-edge serial arm (base → shoulder → elbow → wrist → probe_tip).
          Total reach ≈ 500 mm. Units: metres / radians throughout.

Run from the repo root:
    python examples/pareto_sensitivity_example.py
"""
import matplotlib
matplotlib.use("Agg")  # must be the first matplotlib import — enables headless PNG output
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from postprocess.reporting import generate_frame_report, generate_sensitivity_report
from postprocess.stats import (
    compute_tolerance_sensitivities,
    frame_envelope_box,
    frame_percentiles,
)
from sim.monte_carlo_fk import MonteCarloFKEngine


# ── Output helpers ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def print_envelope(env: dict) -> None:
    print(f"  {'DoF':6}  {'min (mm/rad)':>14}  {'max (mm/rad)':>14}  {'range':>10}")
    print("  " + "-" * 52)
    for dof, bounds in env.items():
        lo, hi = bounds["min"], bounds["max"]
        # Convert to mm for translations, keep rad for rotations
        scale = 1000.0 if dof.startswith("d") else 1.0
        unit_label = "mm" if dof.startswith("d") else "rad"
        print(
            f"  {dof:6}  {lo*scale:>14.4f}  {hi*scale:>14.4f}  "
            f"{(hi-lo)*scale:>10.4f}  {unit_label}"
        )


def print_percentiles(pct: dict) -> None:
    levels = sorted(next(iter(pct.values())).keys())
    header = f"  {'DoF':6}" + "".join(f"  {p:>8.0f}th" for p in levels)
    print(header)
    print("  " + "-" * (8 + 11 * len(levels)))
    for dof, pmap in pct.items():
        scale = 1000.0 if dof.startswith("d") else 1.0
        unit_label = " mm" if dof.startswith("d") else "rad"
        row = f"  {dof:6}" + "".join(f"  {pmap[p]*scale:>9.4f}" for p in levels)
        print(row + f"  {unit_label}")


# ── Scenario helpers ──────────────────────────────────────────────────────────

def _tol(dist: str, bound_t: float, bound_r: float, sigma_level: float = 3.0) -> ToleranceSpec6:
    """Build a ToleranceSpec6 with identical bounds for all 3 translation and all 3 rotation DoF."""
    t = ToleranceSpec(dist, bound=bound_t, sigma_level=sigma_level)
    r = ToleranceSpec(dist, bound=bound_r, sigma_level=sigma_level)
    return ToleranceSpec6(dx=t, dy=t, dz=t, rx=r, ry=r, rz=r)


def build_arm(shoulder_rx_bound: float) -> FrameGraph:
    """Build the 4-edge inspection robot arm.

    shoulder_rx_bound controls the shoulder's angular tolerance — the variable
    we tune in the Component Selection section.

    All offsets are pure z-translations (the arm extends straight up).
    """
    fg = FrameGraph()
    for name in ["base", "shoulder", "elbow", "wrist", "probe_tip"]:
        fg.add_frame(name)

    # Edge 1: base → shoulder  (150 mm, cheap universal joint — loosest angular tol)
    fg.add_edge(
        "base", "shoulder",
        HTM.from_xyz_euler([0.0, 0.0, 0.150], [0.0, 0.0, 0.0]),
        _tol("uniform", bound_t=0.000050, bound_r=shoulder_rx_bound),
        name="shoulder",
    )

    # Edge 2: shoulder → elbow  (200 mm, mid-grade servo)
    fg.add_edge(
        "shoulder", "elbow",
        HTM.from_xyz_euler([0.0, 0.0, 0.200], [0.0, 0.0, 0.0]),
        _tol("normal", bound_t=0.000020, bound_r=0.001),
        name="elbow",
    )

    # Edge 3: elbow → wrist  (100 mm, high-grade servo)
    fg.add_edge(
        "elbow", "wrist",
        HTM.from_xyz_euler([0.0, 0.0, 0.100], [0.0, 0.0, 0.0]),
        _tol("normal", bound_t=0.000010, bound_r=0.0005),
        name="wrist",
    )

    # Edge 4: wrist → probe_tip  (50 mm, precision tooling — tightest)
    fg.add_edge(
        "wrist", "probe_tip",
        HTM.from_xyz_euler([0.0, 0.0, 0.050], [0.0, 0.0, 0.0]),
        _tol("normal", bound_t=0.000005, bound_r=0.0002),
        name="probe",
    )

    return fg


# ── Main script ───────────────────────────────────────────────────────────────

section("TolTransform — Pareto Sensitivity Example: Precision Inspection Robot Arm")

print("""
Scenario: 4-edge serial inspection arm (total reach = 500 mm).

  base → [shoulder, 150 mm] → [elbow, 200 mm] → [wrist, 100 mm] → [probe, 50 mm] → probe_tip

The shoulder joint uses a cheaper, loosely-toleranced universal joint (uniform ±3 mrad).
All other joints are higher-grade servos (normal distribution, tighter bounds).

Goal: Identify which tolerance is driving the probe-tip error budget and determine
      whether upgrading the shoulder joint is worth the cost.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BASELINE: Build and run Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════

N_TRIALS = 50_000
SEED = 42
SHOULDER_LOOSE = 0.003   # 3 mrad — cheap joint (baseline)
SHOULDER_TIGHT = 0.001   # 1 mrad — upgraded joint (component selection scenario)

section("Section 1 — Baseline Chain (shoulder rx/ry = ±3 mrad uniform)")

fg_baseline = build_arm(shoulder_rx_bound=SHOULDER_LOOSE)

print(f"Running {N_TRIALS:,}-trial Monte Carlo (seed={SEED}) ...")
td_baseline = MonteCarloFKEngine.run(fg_baseline, n_trials=N_TRIALS, seed=SEED)
print("Done.\n")

print("Worst-case envelope at probe_tip  [mm / rad]:")
env = frame_envelope_box(td_baseline, "probe_tip")
print_envelope(env)

print("\nStatistical percentile table at probe_tip:")
pct = frame_percentiles(td_baseline, "probe_tip", [50.0, 90.0, 95.0, 99.0])
print_percentiles(pct)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SENSITIVITY PINPOINTING: Pareto breakdown
# ═══════════════════════════════════════════════════════════════════════════════

section("Section 2 — Sensitivity Pinpointing: Pareto Breakdown (base → probe_tip)")

print("""
compute_tolerance_sensitivities() uses the first-order adjoint Jacobian to attribute
each edge/DoF's tolerance to its share of the total output variance.

NOTE: This is a first-order linear approximation. For long chains with significant
angular offsets, nonlinear contributions can differ from these estimates.
""")

report_baseline = compute_tolerance_sensitivities(fg_baseline, "base", "probe_tip")
print(report_baseline.to_ascii_chart(top_n=12))

# Identify the top contributor for commentary
top_edge, top_dof, top_pct = report_baseline.ranked_contributions[0]
print(f"\n>>> Top contributor: '{top_edge}' edge, {top_dof} DoF at {top_pct:.1f}%")
print(f"    This is the cheap shoulder joint's angular slop driving {top_pct:.0f}% of")
print(f"    the probe-tip error budget via lever-arm amplification over 500 mm reach.")
print(f"    (0.003 rad × 0.500 m ≈ {0.003 * 0.500 * 1000:.1f} mm lateral tip deflection)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT SELECTION: Upgrade the dominant contributor
# ═══════════════════════════════════════════════════════════════════════════════

section("Section 3 — Component Selection: Upgrade Shoulder Joint (±3 mrad → ±1 mrad)")

print(f"""
Engineering question: "Can we justify a tighter shoulder actuator (±{SHOULDER_TIGHT*1000:.0f} mrad)?
How much does the probe-tip error budget improve?"

Rebuilding with upgraded shoulder (all other edges unchanged) ...
""")

fg_upgraded = build_arm(shoulder_rx_bound=SHOULDER_TIGHT)
td_upgraded = MonteCarloFKEngine.run(fg_upgraded, n_trials=N_TRIALS, seed=SEED)
report_upgraded = compute_tolerance_sensitivities(fg_upgraded, "base", "probe_tip")

env_up = frame_envelope_box(td_upgraded, "probe_tip")

# Compare dx envelope (most affected by angular lever-arm)
dx_range_before = (env["dx"]["max"] - env["dx"]["min"]) * 1000  # mm
dx_range_after  = (env_up["dx"]["max"] - env_up["dx"]["min"]) * 1000

top_edge_up, top_dof_up, top_pct_up = report_upgraded.ranked_contributions[0]

print("Upgraded chain Pareto breakdown:")
print(report_upgraded.to_ascii_chart(top_n=12))

# Find the first non-shoulder contributor (next joint to target after shoulder)
next_entry = next(
    (e, d, p) for e, d, p in report_upgraded.ranked_contributions
    if e != "shoulder"
)
next_edge, next_dof, next_pct = next_entry

print(f"\nComparison summary:")
print(f"  Shoulder contribution : {top_pct:.1f}% → {report_upgraded.ranked_contributions[0][2]:.1f}%  "
      f"(shoulder still leads, but share has shrunk)")
print(f"  Probe dx range        : {dx_range_before:.3f} mm → {dx_range_after:.3f} mm  "
      f"({(1 - dx_range_after/dx_range_before)*100:.0f}% reduction)")
print(f"\n  Next bottleneck: '{next_edge}' edge, {next_dof} DoF at {next_pct:.1f}%")
print(f"  Further tightening the shoulder beyond ±{SHOULDER_TIGHT*1000:.0f} mrad gives diminishing returns")
print(f"  until the '{next_edge}' joint is also addressed.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — REPORTING: Save figures to disk
# ═══════════════════════════════════════════════════════════════════════════════

section("Section 4 — Reporting: Saving Figures")

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Frame report for the baseline scenario (4×3 grid: histograms + projections + rotation summary)
fig_frame = generate_frame_report(td_baseline, "probe_tip")
frame_report_path = OUT / "probe_tip_frame_report.png"
fig_frame.savefig(frame_report_path, dpi=120, bbox_inches="tight")
plt.close(fig_frame)
print(f"  Saved frame report  : {frame_report_path}")

# Pareto sensitivity chart for the baseline scenario
fig_pareto = generate_sensitivity_report(report_baseline)
pareto_report_path = OUT / "probe_tip_pareto_baseline.png"
fig_pareto.savefig(pareto_report_path, dpi=120, bbox_inches="tight")
plt.close(fig_pareto)
print(f"  Saved Pareto chart  : {pareto_report_path}")

# Pareto chart for the upgraded scenario (for comparison in a design review)
fig_pareto_up = generate_sensitivity_report(report_upgraded)
pareto_up_path = OUT / "probe_tip_pareto_upgraded.png"
fig_pareto_up.savefig(pareto_up_path, dpi=120, bbox_inches="tight")
plt.close(fig_pareto_up)
print(f"  Saved upgraded chart: {pareto_up_path}")

print(f"""
The frame report shows per-DoF histograms, 2D bounding-ellipse projections, and
the rotation cone summary — useful for communicating the full error-envelope shape
to a metrology or integration team.

The Pareto charts (baseline and upgraded) are the primary deliverable for a sourcing
or budget discussion: they assign a defensible percentage to each tolerance line item.

NOTE: The Pareto chart includes a first-order-linear-approximation caveat (printed
on the chart itself) because it may be shared standalone in procurement discussions.
""")

section("Done")
print("Run `python examples/pareto_sensitivity_example.py` to regenerate all output.\n")
