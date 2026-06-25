---
title: "TolTransform: Design Specifications & Project Plan"
subtitle: "A Kinematic Error-Budgeting Tool for Precision Machine Design"
author: "Living Engineering Document — Architecture, Module Specifications, and Phased Plan"
date: "Last updated: 2026-06-25"
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
colorlinks: true
---

\newpage

# 0. Purpose of This Document

**This document is not merely a project plan — it is the authoritative design specification for TolTransform.** It serves three combined functions at once:

1. **Software design specification** — the binding architectural and mathematical conventions, the module-by-module breakdown of the system, and (as the project matures) the detailed behavioral specification for each individual module.
2. **Project plan** — the phased task breakdown, time estimates, and milestone definitions.
3. **Decision record** — a changelog of who decided what, and when, so reasoning is never lost between sessions.

It is written to be re-readable from scratch by either the project owner (a mechanical/robotics engineer) or an AI coding assistant (Claude) with **no prior conversation context**. Every architectural decision below was deliberately chosen after discussion and trade-off analysis; where a decision was made, the rationale is recorded so it does not need to be re-derived or accidentally re-opened in a future session.

**This is a living document.** Section 6 (Module Specifications) in particular is expected to grow over time — each module currently has only a brief description, and these will be expanded into detailed behavioral specifications (inputs, outputs, edge cases, algorithms) as each module is designed and built.

**Rule for future sessions:** if a decision recorded here needs to change, update this document explicitly and log the change, with the name of who made it, in Section 11 (Changelog) — rather than silently diverging from it in code.

---

# 1. Project Overview

## 1.1 What TolTransform Is

TolTransform is a Python-based **system error budgeting tool** for mechanical and systems engineers, with a particular emphasis on **precision machine design** (e.g., precision mechanisms, optical mounting systems, instrument alignment). It allows an engineer to:

1. Define a system as a network of coordinate frames connected by homogeneous transformation matrices (HTMs), each with independent, uncorrelated tolerances in all 6 degrees of freedom (DoF).
2. Run **forward tolerance verification**: given input tolerances on each HTM, statistically and worst-case characterize the resulting pose uncertainty (position + orientation) at any frame of interest, using Monte Carlo methods.
3. Run **inverse tolerance allocation**: given a desired output tolerance envelope, back-solve for a feasible, evenly-allocated set of per-HTM tolerances that achieve it.
4. Analyze relative tolerances **between any two frames** in the system (not just chain endpoints) — critical for problems like relative alignment between two optical components mounted on a shared structure.

## 1.2 What TolTransform Is Explicitly NOT

To keep scope sane and prevent feature creep, the following are **intentionally excluded**:

- **Not a manufacturability/process-correlation tool.** Tolerances on different DoF are treated as statistically independent. We do not model correlated manufacturing error sources (e.g., a single machining setup error correlating x and y). This is a deliberate simplification — the tool is for system-level error budgeting, not detailed manufacturing tolerance analysis.
- **Not a full physics/multibody dynamics engine.** No forces, no stiffness, no contact, no closed kinematic loops, no parallel mechanisms.
- **Not capable of large-angle/large-displacement tolerance analysis.** The tool exclusively uses a **small-angle approximation** for rotational tolerances. This is valid for precision machine design (sub-degree alignment errors) and is what allows us to sidestep Euler angle order-of-rotation ambiguity entirely. If a tolerance is not "small" in the small-angle sense, this tool is the wrong tool.
- **Not a closed-loop / parallel-kinematics solver.** Forward kinematics chains are strictly **open, serial, directed** (a tree/DAG of frames). Parallel or closed-loop mechanisms are explicitly out of scope and should be handled by detailed design tools downstream of this one.

## 1.3 Target User & Use Case

A mechanical/systems/optical engineer who needs to rapidly answer: *"If I tolerance each interface in my assembly to X, what is the resulting positional/angular uncertainty at the point I care about? Or, conversely, if I need the final uncertainty to be Y, how tight do my interface tolerances need to be?"* This is currently done ad hoc in spreadsheets; TolTransform aims to make it fast, repeatable, visual, and statistically rigorous (offering both worst-case and statistical answers side by side).

## 1.4 Core Engineering Decisions Enabled

*(Added 2026-06-23 — Project Owner, per cross-review. The point of this section is to make the software's purpose unmistakable by naming the actual engineering decisions it exists to support, not just describing its mechanics.)*

TolTransform exists to let an engineer answer questions like these quickly, with a defensible number behind the answer rather than a gut call:

- **Component Selection & Sourcing:** *"Can we use a cheaper ±50 μm linear stage for the coarse alignment axis, or does our error budget demand a ±10 μm stage?"* — a direct application of Mode 1 (forward verification, Section 3.1): swap the candidate stage's tolerance into the relevant edge, re-run, and check whether the resulting envelope at the point of interest still meets spec.
- **Sensitivity Pinpointing:** *"Which specific angular or translational tolerance edge is the dominant driver of our end-of-arm volumetric error?"* — the direct motivation for the Pareto sensitivity breakdown added this revision (Section 6.8, Mod 2 below): an engineer should be able to point at one number and say "that joint's tilt tolerance is responsible for 45% of our total budget," not just see an aggregate envelope with no attribution.
- **Mitigation Verification:** *"If we spend money to ground our independent metrology rail directly to a shared optical base plate, how much absolute error cancellation do we achieve at our sensor plane?"* — a direct application of the point-pair / shared-ancestor analysis (Section 2.4, Section 3.3, and the Common-Ancestor Cancellation Benchmark in Section 9.1): model the proposed shared base plate as a junction Frame, run the relative-pose query between the two components, and see the cancellation effect quantified before spending the money.

Every module spec in Section 6 should trace back to enabling one of these three categories of decision — if a proposed feature doesn't serve component selection, sensitivity pinpointing, or mitigation verification, it's worth asking whether it belongs in the tool at all (per Section 1.2's scope discipline).

---

# 2. Core Engineering & Mathematical Conventions

These conventions are foundational. **Every module in the codebase must adhere to these exactly.** Inconsistency here is the single biggest risk to project correctness.

## 2.1 Coordinate Transform Representation

- Every HTM is stored canonically as a 4x4 homogeneous transformation matrix (NumPy `ndarray`, `dtype=float64`).
- The tool must support **input** of nominal transforms in multiple user-friendly forms, all converted to canonical HTM internally:
  - XYZ translation + Euler angles (specify convention, e.g., intrinsic ZYX — to be fixed in implementation and documented in code)
  - Raw 4x4 HTM
  - Screw / exponential coordinates (axis, angle, translation along axis)
  - Quaternion + translation
- Conversions are implemented using **pytransform3d** wherever possible rather than hand-rolled, to avoid subtle bugs (gimbal lock, normalization, convention errors). See Section 4 for library usage.
- The system must preserve a record of **which input form the user originally used** for each HTM (metadata only) so the GUI can round-trip the user's preferred representation without forcing canonicalization on display.

## 2.2 Tolerance Representation (Small-Angle, 6-DoF, Uncorrelated)

Each HTM edge has an associated **6-vector tolerance specification**:

```
delta = [dx, dy, dz, rx, ry, rz]
```

where `dx, dy, dz` are small translational perturbations and `rx, ry, rz` are small-angle rotational perturbations (radians), treated as if they commute (valid under the small-angle approximation — this is precisely why we don't need to worry about Euler rotation order for tolerances, even though the *nominal* transform itself may have been specified with explicit Euler angles).

**Per-DoF independence is a hard rule.** Each of the 6 DoF gets its own independent tolerance specification (distribution + bound). No covariance/correlation between DoF, and no covariance/correlation between different HTM edges. This is an intentional simplification (see Section 1.2).

### 2.2.1 Distribution Types (v1 scope)

Each DoF tolerance specification includes:

- `distribution`: one of `"uniform"` (hard bound, default) or `"normal"` (statistical)
- `bound`: the stated tolerance value (e.g., ±0.05 mm or ±0.001 rad)
- `sigma_level` (only relevant for `"normal"`): how many standard deviations the stated `bound` represents. **Default = 3.0** (i.e., stated tolerance = 3σ) unless overridden by the user, either globally or per-edge.
- `locked`: boolean flag — if true, this tolerance is fixed and excluded from inverse allocation optimization (Section 2.4).

**Default distribution = `"uniform"` (hard bound)**, per explicit user decision — worst-case bounding is the primary use case for this tool, with statistical (`"normal"`) treatment available as a secondary, comparative analysis mode.

### 2.2.2 Perturbation Application Convention

**Local-frame, right-multiplication.** For an HTM edge with nominal transform `T_nominal` and sampled perturbation vector `delta`, the perturbed transform for a given Monte Carlo trial is:

```
T_perturbed = T_nominal @ T_delta(delta)
```

where `T_delta(delta)` is the small-angle HTM built from `delta` (translation `[dx,dy,dz]` directly, rotation via small-angle approximation, i.e., the rotation matrix ≈ `I + skew([rx,ry,rz])`, re-orthonormalized if needed for numerical hygiene).

This was chosen because local-frame (child-relative) perturbation is the most intuitive convention for a mechanical engineer thinking about "this interface has this much wobble in its own local directions" — confirmed explicitly with the project owner. **Do not switch to left-multiplication / global-frame perturbation without updating this section and re-deriving all sensitivity/Jacobian math downstream.**

## 2.3 Frame Graph Topology

- The system topology is a **directed graph** where nodes are **Frames** (named coordinate frame identities with no inherent geometry) and edges are **HTM Edges** (the actual toleranced transform from a parent Frame to a child Frame).
- **Forward kinematics (FK) mode restriction: the graph must be a tree / DAG where every Frame has at most one incoming edge.** This enforces "strictly serial, open kinematic chains" and explicitly disallows closed loops or parallel mechanisms, per project scope (Section 1.2). This must be validated at model-build time with a clear, actionable error message if violated.
- **Shared/junction Frames** (a Frame referenced as the parent of more than one outgoing edge) are how multiple chains relate to each other — e.g., two separate optical-component chains both originate from a common mounting-frame Frame node. This is a first-class capability, not a special case: it falls naturally out of representing the system as a graph rather than independent chains.
- **"Chain"** is a GUI/organizational convenience only — an ordered path of edges from a root Frame to a leaf Frame for display purposes. The simulation engine operates on the graph directly and has no built-in concept of "chain."
- The relative transform between **any two Frames** in the graph is computed by finding the unique path between them through their lowest common ancestor (using graph traversal — see Section 4, NetworkX) and composing/inverting edges along that path. This generalizes both "chain output tolerance" and "point-pair analysis between frames on different chains" into one operation.

### 2.3.1 Disjoint Components: No Silent Auto-Connection (Locked Decision)

**The engine never silently connects disjoint sub-graphs.** If a project contains two or more weakly-connected components (e.g., two chains that don't share any Frame at all), `nominal_transform_between()` and any point-pair query spanning the two components **must raise a clear, specific error rather than fabricating a connection.**

**Rationale (this was explicitly considered and rejected as a "fix"):** An earlier review round proposed having the engine automatically designate a single global `world`/`base` Frame and silently attach every disjoint root to it via an invented zero-tolerance identity HTM, purely to guarantee that every relative-transform query succeeds. **This was rejected.** Auto-inserting a rigid, zero-tolerance connection the user never specified would make the tool's worst-case bounds silently optimistic in exactly the case where two sub-assemblies are *not* actually rigidly connected (e.g., two components on separate machine bases with real alignment uncertainty between them) — and the user might never notice, because the tool would simply stop raising the error that would otherwise have surfaced the missing information. For a tool whose entire purpose is producing trustworthy error budgets, an engine that refuses to answer an ill-posed question is strictly safer than one that guesses.

**The correct modeling pattern — "Common Physical Base":** If two (or more) chains are physically referenced to a shared structure (a machine base, an optical table, a common mounting plate), the user must **explicitly model that structure as a Frame** and connect each chain's root to it via a real `HTMEdge` — using an identity nominal transform with zero tolerance if the connection is genuinely treated as rigid/ideal for this analysis, or a real nonzero tolerance if there is actual assembly/alignment uncertainty between the chain's root and the shared base. This requires no engine changes — it is the existing graph model used exactly as designed.

Example:

```
# Two optical-component chains, each independently rooted, made queryable
# against each other by explicitly tying both to a shared machine base.

graph.add_frame("base")                  # the shared physical reference
graph.add_frame("lens1_mount")
graph.add_frame("lens1_seat")
graph.add_frame("lens2_mount")
graph.add_frame("lens2_seat")

# Tie each chain's root to "base" with a real edge — identity + zero tolerance
# if the base attachment is treated as ideal/rigid for this analysis, or a
# nonzero tolerance if there is real assembly uncertainty at that joint.
graph.add_edge("base", "lens1_mount", nominal=HTM.identity(), tolerance=zero_tolerance_spec())
graph.add_edge("base", "lens2_mount", nominal=HTM.identity(), tolerance=zero_tolerance_spec())

graph.add_edge("lens1_mount", "lens1_seat", nominal=..., tolerance=...)
graph.add_edge("lens2_mount", "lens2_seat", nominal=..., tolerance=...)

# Now "lens1_seat" and "lens2_seat" share a common ancestor ("base") and a
# point-pair query between them succeeds, correctly propagating whatever
# tolerance (zero or nonzero) was actually assigned to the base attachment edges.
graph.nominal_transform_between("lens1_seat", "lens2_seat")  # succeeds
```

**Error message contract (implemented in `core/frame_graph.py`, Section 6.3):** when a point-pair query is attempted across two Frames with no path between them, the raised error must explicitly suggest this pattern rather than reporting a generic graph-traversal failure:

```
"Frames 'X' and 'Y' have no connected path between them. If these components
share a common physical reference (e.g., a machine base or optical table),
add an explicit joint Frame and define the structural edges connecting both
sub-assembly roots to it."
```

## 2.4 Monte Carlo Consistency Across Shared Edges (Critical)

**Each HTM edge is sampled exactly once per Monte Carlo trial, not once per chain that uses it.** Because the frame graph is the canonical model (not independent per-chain copies), a shared upstream edge naturally produces the *same* sampled perturbation for every downstream Frame that depends on it, within a given trial. This is what allows correct relative-tolerance analysis between frames on different chains that share a common ancestor edge — it is a structural property of the graph-based design, not a special-cased rule that has to be separately implemented or maintained.

## 2.5 Worst-Case vs. Statistical Analysis — Both Required, Same Engine

The simulation engine is **distribution-agnostic** — the same chain-composition and post-processing code runs regardless of whether a given DoF tolerance is sampled as `"uniform"` (worst-case) or `"normal"` (statistical). The user can mix distributions across different edges/DoF if desired, though the typical use case is to run the entire model once in pure worst-case mode and once in pure statistical mode, then compare.

A secondary, optional, **fast analytical worst-case estimate** (linear/Jacobian-based superposition of per-DoF max contributions) may be added as a quick sanity-check feature that does not require running a full Monte Carlo — this is a v1.x nice-to-have, not required for V0.5 or V1.0 (see Section 8).

---

# 3. Operating Modes

## 3.1 Mode 1 — Forward Tolerance Verification (FK)

**Input:** A frame graph with nominal HTMs and per-DoF tolerance specs on every edge.
**Process:** Monte Carlo sample every edge (N trials), compose transforms along the graph to get a sampled pose for every Frame in every trial, store all per-trial per-Frame pose data.
**Output:** For any Frame (or any pair of Frames), statistical and worst-case envelope of position and orientation error: bounding box, bounding ellipsoid/sphere (translation), bounding cone (rotation), histograms per DoF, percentile tables.

## 3.2 Mode 2 — Inverse Tolerance Allocation (IK)

**Input:** A frame graph with nominal HTMs (tolerances initially unknown/unset, or partially set with some edges "locked" to known fixed values), plus a **target**: a desired tolerance envelope on the relative pose between two specified Frames (commonly chain-root-to-leaf, but generalizes to any two Frames per Section 2.3).
**Process:**
1. Compute the linear sensitivity (Jacobian) of the target relative pose with respect to each free (unlocked) edge's 6-DoF perturbation. This is closed-form/cheap due to the small-angle linearity assumption.
2. Solve for a scale factor (or per-DoF scale factors) applied uniformly across free edges such that the linearly-propagated worst-case envelope meets the target exactly — this is the **"equal allocation"** objective, the only objective required for v1 (confirmed default; see Section 2.2 rationale — engineers can manually adjust afterward using their own judgment).
3. **Validate** the proposed allocation with a full nonlinear Monte Carlo run on the resulting tolerance set, since chain composition through multiple HTMs introduces some nonlinearity beyond the local linear approximation. Report any discrepancy between the linear-allocation prediction and the MC-validated result.
**Output:** A proposed per-edge, per-DoF tolerance set, plus a validation report showing the MC-simulated envelope achieved by that proposed set vs. the target.

**Locked edges:** Any edge (or specific DoF on an edge) the user marks as `locked` is excluded from the free-variable vector in step 2 above and is treated as a fixed, known contribution to the budget. This supports the real-world case where some interfaces are already fixed by an existing/selected component.

**Allocation objective extensibility:** The allocation objective is implemented behind a small interface (`AllocationObjective`) with exactly one built-in implementation (`EqualAllocation`) for v1. This costs almost nothing architecturally and avoids a rewrite if a weighted or cost-based objective is wanted later — but **do not build additional objectives speculatively; YAGNI until requested.**

## 3.3 Point-Pair Post-Processing (Available in Both Modes)

After any Monte Carlo run (FK or the validation step of IK), the user may select **any two Frames** in the graph and request the relative pose tolerance/envelope between them, computed from the *already-stored* per-trial pose data (Section 2.3/2.4) — no re-simulation required. This is the mechanism that satisfies the original optical-systems use case: relative alignment tolerance between two components on different chains sharing a common mount.

---

# 4. Third-Party Libraries (Explicit Decisions — Do Not Re-Litigate)

| Library | Used For | Rationale |
|---|---|---|
| **NumPy** | All numerical arrays, vectorized Monte Carlo trial batches | Standard; vectorize trials as batched `(N,4,4)` array ops, never loop over trials in Python |
| **pytransform3d** | HTM <-> euler/quaternion/screw/axis-angle conversions | Battle-tested, avoids hand-rolled convention/edge-case bugs. **Do not use its perturbation/composition conventions for the tolerance model** — our local-frame right-multiply small-angle perturbation (Section 2.2.2) is bespoke and must be hand-implemented/tested |
| **NetworkX** | Frame graph: DAG validation, cycle detection, path-finding between arbitrary Frames | Avoids hand-written graph traversal code; directly supports Section 2.3 |
| **SciPy** | `scipy.stats` for sampling distributions (uniform, normal incl. sigma-level conversion); `scipy.optimize` as the extensibility hook for `AllocationObjective` implementations beyond v1's closed-form equal allocation | Standard, reliable |
| **Pydantic** | Project file schema, validation on load (catches cycles, missing frame references, malformed tolerance specs with clear error messages before the engine ever runs) | Validation-by-construction; good error UX |
| **Matplotlib** | All plots: histograms, bounding-shape wireframes (box/ellipsoid/cone), 2D projections | Standard, embeds cleanly in PySide6 |
| **PySide6** | GUI framework (V1.0 only — not needed for V0.5) | Native desktop, no browser/server split, mature Matplotlib embedding, good file dialogs — matches "simple, reliable, local" requirement |

**Explicitly hand-built, not outsourced to any library** (this is the actual intellectual core of the tool):
- The 6-DoF tolerance perturbation/sampling model (Section 2.2)
- The Monte Carlo engine loop and trial-data storage structure
- The inverse allocation solver and its Jacobian/sensitivity computation
- The bounding-shape fitting logic (ellipsoid/cone fitting from point clouds)

---

# 5. Software Architecture

## 5.1 Directory Structure

```
toltransform/
├── core/
│   ├── transforms.py       # HTM class; canonical 4x4 storage; conversions via pytransform3d
│   ├── tolerance.py        # ToleranceSpec (per-DoF), sampling, perturbation composition
│   ├── frame_graph.py      # Frame, HTMEdge, FrameGraph (NetworkX-backed), DAG validation,
│   │                       #   path-finding/relative-transform-between-any-two-frames
│   ├── conversions.py      # Thin wrappers around pytransform3d for xyz/euler/quat/screw <-> HTM
│   └── sampling.py         # Distribution sampling (uniform/normal via scipy.stats), sigma-level handling
│   │                       #   [relocated from sim/ — see Section 6 architecture correction note]
├── sim/
│   ├── monte_carlo_fk.py   # Forward verification engine: batched sampling + chain composition;
│   │                       #   owns TrialData and per-edge RNG sub-stream derivation
│   └── allocation.py       # Inverse allocation engine: Jacobian/sensitivity + EqualAllocation
│                           #   objective + MC validation pass
├── postprocess/
│   ├── stats.py            # Per-Frame and per-point-pair envelope stats: box, percentiles, sigma
│   ├── bounding_shapes.py  # Bounding box / ellipsoid / sphere (translation), bounding cone (rotation)
│   └── reporting.py        # Plot generation (histograms, 2D projections of bounding shapes)
├── io/
│   ├── schema.py           # Pydantic models: Project, Frame, HTMEdge, ToleranceSpec, SimSettings,
│   │                       #   SavedAnalysis
│   └── serializer.py       # JSON save/load, validation-on-load
├── gui/                    # V1.0 ONLY — not built in V0.5
│   ├── main_window.py
│   ├── graph_editor/
│   ├── tolerance_editor/
│   ├── run_panel/
│   ├── results_viewer/
│   └── point_pair_panel/
├── examples/                # Hand-verified example scripts/cases (see Section 9)
└── tests/
    ├── conftest.py
    ├── test_transforms.py
    ├── test_tolerance.py
    ├── test_frame_graph.py
    ├── test_monte_carlo_fk.py
    ├── test_allocation.py
    ├── test_stats.py
    ├── test_bounding_shapes.py
    ├── test_reporting.py
    ├── test_schema.py
    └── test_serializer.py
```

## 5.2 Key Class Sketch (Conceptual — Not Final Code)

This is a conceptual sketch to align understanding; actual implementation will be written collaboratively, not dictated wholesale here.

- **`HTM`** (`core/transforms.py`): wraps a 4x4 NumPy array. Constructors: `from_xyz_euler(...)`, `from_screw(...)`, `from_quaternion(...)`, `from_matrix(...)`. Methods: `compose(other)`, `inverse()`, `to_xyz_euler()`, etc. Stores `input_representation` metadata.
- **`ToleranceSpec`** (`core/tolerance.py`): one per DoF, fields `distribution`, `bound`, `sigma_level`, `locked`. A `ToleranceSpec6` aggregates 6 of these for one edge. Method `sample(n_trials, rng)` returns an `(n_trials, 6)` array of perturbation vectors.
- **`Frame`** (`core/frame_graph.py`): just a name/id + optional metadata. No geometry.
- **`HTMEdge`** (`core/frame_graph.py`): `parent_frame`, `child_frame`, `nominal: HTM`, `tolerance: ToleranceSpec6`.
- **`FrameGraph`** (`core/frame_graph.py`): wraps a NetworkX `DiGraph`. Methods: `add_frame`, `add_edge`, `validate_dag()`, `path_between(frame_a, frame_b)`, `relative_transform_nominal(frame_a, frame_b)`.
- **`MonteCarloFKEngine`** (`sim/monte_carlo_fk.py`): `run(frame_graph, n_trials, rng_seed) -> TrialData`, where `TrialData` stores, for every Frame, an `(n_trials, 4, 4)` array of absolute pose per trial (locked decision: full 4x4, not reduced 6-vector — see Section 6) — vectorized, no per-trial Python loop. This module also owns per-edge RNG sub-stream derivation (Section 6.6).
- **`AllocationEngine`** (`sim/allocation.py`): `solve(frame_graph, target_frame_pair, target_tolerance, objective=EqualAllocation()) -> ProposedToleranceSet`, plus `validate(proposed_set, n_trials) -> ValidationReport`.
- **`BoundingShapeFitter`** (`postprocess/bounding_shapes.py`): takes an `(n_trials, 3)` translation point cloud or `(n_trials, 3)` rotation-vector cloud, returns box/ellipsoid/cone parameters.

## 5.3 GUI-Engine Decoupling Principle (V1.0)

The GUI **never touches `core`/`sim` objects directly during editing** — it reads and writes the `io.schema` Pydantic models exclusively, and only constructs `core`/`sim` objects at "Run" time from the validated schema. This keeps the engine fully scriptable/headless-usable independent of the GUI, and means a future CLI or batch-runner needs zero GUI code.

---

# 6. Module Specifications

This section gives every code module in the architecture (Section 5.1) its own dedicated subsection, broken down into **granular, ordered task lists**, explicit **deliverables**, and explicit **interfaces** (what each module depends on, what depends on it, and its public API surface). This is the detailed planning layer of the project: the goal is that any module's subsection below should be specific enough that implementation work can begin directly from it, with no further design decisions required mid-build.

**Convention for future expansion:** every module subsection carries a `*(Last revised: ...)*` line under its heading. When a module's spec is materially changed, update that line and log the change (with name) in Section 11 (Changelog).

**Architecture correction made during this revision:** `sampling.py` was originally placed under `sim/`, but `core/tolerance.py` (a `core` module) depends on it for distribution sampling — that's a backwards dependency, since `sim/` is supposed to depend on `core/`, never the reverse. **`sampling.py` has been relocated to `core/sampling.py`.** The directory tree in Section 5.1 and all module numbering below reflect this correction. Module count and overall numbering (6.1–6.20) are unaffected — `core/` now owns 5 modules (6.1–6.5) and `sim/` owns 2 (6.6–6.7).

**Shared data contract established this revision — `TrialData`:** Because this structure is consumed by nearly every downstream module, its definition is centralized here for reference (full ownership and construction detail is in Section 6.6):

```
TrialData
├── n_trials: int
├── seed: int                                  # master seed used for this run
├── frame_poses: dict[str, np.ndarray]          # frame_name -> (N,4,4) absolute pose per trial
├── nominal_poses: dict[str, np.ndarray]        # frame_name -> (4,4) unperturbed reference pose
└── edge_seed_log: dict[str, int]               # edge_name -> derived sub-stream spawn key (traceability)
```

Per Section 2.4/2.5 and the decisions locked this session: pose data is stored as **full 4x4 matrices** (not reduced 6-vectors) for simplicity of composition; each weakly-connected component's root Frame (zero in-degree) is anchored at identity for every trial (it has no incoming edge, hence no tolerance, hence no perturbation); and `frame_poses[name][i]` is always the frame's pose in its own component's root frame for trial `i` — meaning the relative transform between *any* two frames in the *same component* can be computed directly from stored data via `inverse(T_a[i]) @ T_b[i]`, with no graph re-traversal needed at post-processing time.

---

## 6.1 `core/transforms.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session; **implemented A1, commit `a31218e`, 26 tests passing**)*

**Responsibility:** Defines the canonical `HTM` class — the single representation of a homogeneous transformation matrix used everywhere else in the codebase — plus its constructors, converters, and core operations.

**Deliverables:**

- `HTM` class wrapping a validated 4x4 `float64` NumPy array
- Constructors: `from_xyz_euler`, `from_matrix`, `from_quaternion`, `from_screw`
- Converters: `to_xyz_euler`, `to_quaternion`, `to_screw`
- `compose()`, `inverse()` methods
- Input-representation metadata, preserved and retrievable
- Construction-time validation with actionable error messages
- `tests/test_transforms.py` with hand-calculated and round-trip coverage

**Granular Task List:**

1. Lock and document the Euler convention as a module-level constant: **intrinsic ZYX** (locked this session). State this explicitly in the module docstring, not buried in a default parameter, since it is a foundational, hard-to-spot-if-wrong convention.
2. Define an `InputRepresentation` data structure: `{kind: "xyz_euler" | "matrix" | "quaternion" | "screw", raw_params: dict}`.
3. Implement `HTM.__init__(matrix, input_representation=None)`: validate shape is `(4,4)`, bottom row is `[0,0,0,1]` (within tolerance), and the rotation block is approximately orthonormal (e.g., `‖RᵗR - I‖ < 1e-6`); raise `ValueError` with a specific, actionable message on failure (state which check failed and the actual deviation).
4. Implement `HTM.from_xyz_euler(xyz, euler_angles, convention="intrinsic_zyx")`, delegating the rotation construction to `core/conversions.py`.
5. Implement `HTM.from_matrix(matrix)` — passthrough with the Step 3 validation, tagged `input_representation.kind = "matrix"`.
6. Implement `HTM.from_quaternion(quat_wxyz, xyz)`, delegating to `core/conversions.py`.
7. Implement `HTM.from_screw(axis, angle, translation_along_axis, point_on_axis=None)`, delegating to `core/conversions.py`; explicitly handle the zero-angle (pure translation) degenerate case rather than letting it propagate a divide-by-zero or NaN.
8. Implement `to_xyz_euler(convention="intrinsic_zyx")`, `to_quaternion()`, `to_screw()` as round-trip converters that operate on `self.matrix` directly — independent of how the instance was originally constructed.
9. Implement `compose(other: HTM) -> HTM` = matrix product; the result's `input_representation` is `None`/`"composed"` (composing two transforms has no single faithful "input representation").
10. Implement `inverse() -> HTM` using the closed-form rigid-body inverse (`R.T`, `-R.T @ t`) rather than `np.linalg.inv`, for numerical robustness and speed.
11. Implement `__repr__` (human-readable, shows translation + Euler angles for quick debugging) and a tolerance-aware `__eq__` (or a dedicated `is_close(other, atol)` method, since exact float equality on transforms is rarely meaningful).
12. ✅ **Done (A1, `a31218e`)** Write `tests/test_transforms.py` (26 tests):
    - ✅ Hand-calculated cases: pure translation; pure single-axis rotation; combined rotation + translation.
    - ✅ Round-trip tests: all 4 constructors (xyz_euler, matrix, quaternion, screw) with round-trip recovery within tolerance.
    - ✅ Composition test: hand-multiplied result check.
    - ✅ Inverse test: `T.compose(T.inverse()) ≈ I` for several non-trivial T.
    - ✅ Edge case: near-gimbal-lock Euler angle input (89.9° pitch) handled gracefully.

**Interfaces:**

- *Depends on:* `core/conversions.py` (all actual `pytransform3d` calls are routed through there — `transforms.py` itself never imports `pytransform3d` directly).
- *Used by:* `core/tolerance.py` (perturbation composition), `core/frame_graph.py` (`HTMEdge.nominal`), `sim/monte_carlo_fk.py`, `sim/allocation.py`, `io/schema.py` (serialization round-trip), and eventually `gui/graph_editor/`.
- *Public API (conceptual):*
  ```
  HTM.from_xyz_euler(xyz, euler_angles, convention="intrinsic_zyx") -> HTM
  HTM.from_matrix(matrix) -> HTM
  HTM.from_quaternion(quat_wxyz, xyz) -> HTM
  HTM.from_screw(axis, angle, translation_along_axis, point_on_axis=None) -> HTM
  HTM.matrix -> np.ndarray (4,4)
  HTM.compose(other: HTM) -> HTM
  HTM.inverse() -> HTM
  HTM.to_xyz_euler(convention) / .to_quaternion() / .to_screw()
  HTM.is_close(other, atol) -> bool
  ```

---

## 6.2 `core/tolerance.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session; **implemented A2, commit `3ac0eed`, 21 tests passing**)*

**Responsibility:** Defines the per-DoF and per-edge tolerance specifications, and implements the locked-convention small-angle perturbation model (Section 2.2.2): turning a sampled 6-vector into an applied perturbation on a nominal `HTM`.

**Deliverables:**

- `ToleranceSpec` (single DoF: distribution, bound, sigma-level, locked flag)
- `ToleranceSpec6` (ordered aggregate of 6 `ToleranceSpec`, one per `[dx,dy,dz,rx,ry,rz]`)
- `skew()`, `small_angle_rotation_matrix_batch()`, `delta_to_htm_batch()`, `apply_perturbation_batch()` — the full vectorized perturbation pipeline
- `tests/test_tolerance.py` with hand-calculated perturbation checks

**Granular Task List:**

1. Implement `ToleranceSpec` as a validated dataclass: `distribution: Literal["uniform","normal"]`, `bound: float` (must be `>= 0`), `sigma_level: float = 3.0` (only meaningful when `distribution == "normal"`), `locked: bool = False`. Validate at construction; raise on negative bound or invalid distribution string.
2. Implement `ToleranceSpec6` as an ordered container of exactly six `ToleranceSpec` instances, exposing both indexed access (`[0..5]`) and named properties (`.dx, .dy, .dz, .rx, .ry, .rz`) for readability in calling code.
3. Implement `ToleranceSpec.sample(n_trials, rng) -> np.ndarray` shape `(n_trials,)`, delegating the actual distribution math to `core/sampling.py` (do not duplicate distribution logic here — keep it in one place, per Section 6.5).
4. Implement `ToleranceSpec6.sample(n_trials, rng) -> np.ndarray` shape `(n_trials, 6)` by calling `.sample()` on each of the six specs and stacking as columns, in the fixed `[dx,dy,dz,rx,ry,rz]` order.
5. **Explicit decision on `locked` and sampling:** `sample()` always samples every DoF regardless of `locked` — a locked tolerance still represents a real physical value that contributes to FK propagation. `locked` is consulted *only* by the allocation engine (Section 6.7) when selecting free variables; it has no effect inside `core/tolerance.py` itself. Document this explicitly in the module docstring to prevent future confusion.
6. Implement `skew(v: np.ndarray) -> np.ndarray`: batched skew-symmetric matrix builder, input shape `(...,3)`, output shape `(...,3,3)`.
7. Implement `small_angle_rotation_matrix_batch(rotvec_batch: np.ndarray) -> np.ndarray` shape `(N,3,3)`: build `R ≈ I + skew(rotvec)` per Section 2.2.2, then re-orthonormalize each matrix (e.g., one Newton/Schulz iteration or a single SVD-based projection) — document *why*: the first-order small-angle expansion is not exactly orthonormal, and downstream code (`HTM.inverse()`, `HTM.compose()`) assumes a valid rotation matrix.
8. Implement `delta_to_htm_batch(delta_batch: np.ndarray) -> np.ndarray` shape `(N,4,4)`: assemble the batched perturbation HTM directly from the translation columns and the Step 7 rotation block.
9. Implement `apply_perturbation_batch(nominal: HTM, delta_batch: np.ndarray) -> np.ndarray` shape `(N,4,4)`: per Section 2.2.2's locked local-frame right-multiply convention, `T_perturbed[i] = nominal.matrix @ delta_to_htm_batch(delta_batch)[i]`, implemented as a single vectorized batched matrix multiply (`np.einsum` or broadcasted `@`) — **no Python-level loop over `i`.**
10. ✅ **Done (A2, `3ac0eed`)** Write `tests/test_tolerance.py` (21 tests):
    - ✅ Zero-delta case: `apply_perturbation_batch` with all-zero `delta_batch` returns nominal exactly.
    - ✅ Known small-angle case: single nonzero `rx` produces rotation matrix matching `I + skew(rx,0,0)`.
    - ✅ Re-orthonormalization sanity check: distortion below documented threshold.
    - ✅ `ToleranceSpec6.sample` shape/bounds check: uniform samples within `[-bound, +bound]`; normal empirical σ ≈ `bound / sigma_level`.
    - ✅ `locked=True` specs are still sampled — regression test guarding Section 2.2.1 semantics.

**Interfaces:**

- *Depends on:* `core/transforms.py` (`HTM`), `core/sampling.py` (distribution sampling primitives — relocated here this revision; see note at top of Section 6).
- *Used by:* `core/frame_graph.py` indirectly (an `HTMEdge` carries a `ToleranceSpec6`), `sim/monte_carlo_fk.py` (calls `.sample()` and `apply_perturbation_batch()` per edge), `sim/allocation.py` (constructs candidate `ToleranceSpec6` instances during allocation), `io/schema.py` (serializes/deserializes `ToleranceSpec`/`ToleranceSpec6`).
- *Public API (conceptual):*
  ```
  ToleranceSpec(distribution, bound, sigma_level=3.0, locked=False)
  ToleranceSpec.sample(n_trials, rng) -> np.ndarray (n_trials,)
  ToleranceSpec6(dx, dy, dz, rx, ry, rz)   # six ToleranceSpec instances
  ToleranceSpec6.sample(n_trials, rng) -> np.ndarray (n_trials, 6)
  apply_perturbation_batch(nominal: HTM, delta_batch: np.ndarray) -> np.ndarray (N,4,4)
  ```

---

## 6.3 `core/frame_graph.py`

*(Last revised: 2026-06-24 — Claude, design-spec update per cross-review of disjoint-component handling; further revised same day to relocate `adjoint()`/`compute_sensitivity()` here per Mod 2; **implemented A3, commit `d81645c`, 24 tests passing; adjoint formula and sensitivity formula corrected per finite-difference cross-check — see changelog 2026-06-24**)*

**Responsibility:** Defines `Frame`, `HTMEdge`, and `FrameGraph` — the NetworkX-backed directed graph that is the canonical topological model of the entire system, including DAG validation, root identification, and relative-transform queries between arbitrary Frames.

**Deliverables:**

- `Frame`, `HTMEdge` data structures
- `FrameGraph` wrapping a NetworkX `DiGraph`, with build, validation, and query methods
- Clear, actionable validation errors (cycles, multiple-incoming-edges, disconnected references)
- `adjoint(T: HTM) -> np.ndarray` and `compute_sensitivity(frame_graph, frame_a, frame_b, edge_names) -> np.ndarray` — **relocated here this revision** (see Step 13) as the single, shared small-angle sensitivity primitive used by both `postprocess/stats.py`'s Pareto sensitivity breakdown and `sim/allocation.py`'s inverse allocator
- `tests/test_frame_graph.py` covering validation and relative-transform queries, including multi-component (multi-chain) cases, plus the relocated sensitivity primitive's correctness

**Granular Task List:**

1. Implement `Frame`: `name: str` (must be unique within a `FrameGraph`), optional `metadata: dict`.
2. Implement `HTMEdge`: `parent: str`, `child: str`, `nominal: HTM`, `tolerance: ToleranceSpec6`, `name: str` (defaults to `f"{parent}->{child}"` if not explicitly given, but must be unique within the graph — required for the per-edge RNG sub-stream keying in Section 6.6).
3. Implement `FrameGraph.__init__()` wrapping an empty `networkx.DiGraph`.
4. Implement `FrameGraph.add_frame(name, metadata=None)` — raises if `name` already exists.
5. Implement `FrameGraph.add_edge(parent, child, nominal, tolerance, name=None)` — raises if `parent`/`child` Frames don't exist, or if `name` collides with an existing edge.
6. Implement `FrameGraph.validate_dag()`:
   - Raise (with the specific cycle path printed) if the graph contains a cycle — use `networkx.find_cycle` to extract and report it, don't just say "graph is invalid."
   - Raise (naming the offending Frame and listing all its incoming edges) if any Frame has more than one incoming edge — this is the explicit "strictly serial, open chains only" rule from Section 2.3/1.2.
   - This method must be called (and must pass) before any simulation engine (Section 6.6) accepts the graph as input — engines should call it internally as a precondition, not rely on the caller remembering to.
7. Implement `FrameGraph.root_frames() -> list[str]`: returns all Frames with in-degree zero — one root per weakly-connected component. Document explicitly: **a root Frame's pose is defined as identity for every Monte Carlo trial**, since it has no incoming edge and therefore no tolerance to sample (Section 6 top-level `TrialData` note).
8. Implement `FrameGraph.weakly_connected_components() -> list[set[str]]` (thin wrapper over `networkx.weakly_connected_components`) — used to validate that point-pair queries (Step 10) involve Frames in the same component.
9. Implement `FrameGraph.topological_edge_order() -> list[str]` (edge names in an order consistent with a topological sort of the underlying DAG) — this is the order the FK engine (Section 6.6) must process edges in to compose poses correctly (a child's pose depends on its parent's already being computed).
10. Implement `FrameGraph.nominal_transform_between(frame_a, frame_b) -> HTM`: for Frames in the same component, compose nominal edges along the unique path from each Frame up to their lowest common ancestor (which, given the "max one incoming edge" rule, is just each Frame's unique path back to its shared root or nearer common ancestor), returning the transform from `frame_a` to `frame_b`.
    - **Disjoint-component error contract (locked decision, Section 2.3.1):** if `frame_a` and `frame_b` are in different weakly-connected components, raise a dedicated exception (e.g., `DisjointFramesError`) carrying exactly the following message, with `frame_a`/`frame_b` substituted in:
      ```
      "Frames '{frame_a}' and '{frame_b}' have no connected path between them.
      If these components share a common physical reference (e.g., a machine
      base or optical table), add an explicit joint Frame and define the
      structural edges connecting both sub-assembly roots to it."
      ```
    - This message must be raised at this layer (not left to bubble up as a raw NetworkX `NetworkXNoPath` exception) — wrap the underlying graph-traversal failure explicitly. **Do not auto-connect the components under any circumstance** (Section 2.3.1) — this error is the intended, correct behavior, not a defect to be silently engineered around.
11. Implement `FrameGraph.all_edges() -> list[HTMEdge]` and `FrameGraph.all_frames() -> list[Frame]` (simple accessors, needed by the simulation engines to iterate).
12. Implement `FrameGraph.path_edges_between(frame_a, frame_b) -> list[HTMEdge]`: returns the ordered list of edges on the unique path between two Frames in the same component (the edge-level counterpart to `nominal_transform_between()`, Step 10) — this is what both Step 13's sensitivity computation and `sim/allocation.py`'s allocator need to know *which* edges to differentiate with respect to, not just the composed nominal transform between the two endpoints.
13. **Relocated this revision (Mod 2, cross-review):** Implement the shared small-angle sensitivity primitives here, in `core/frame_graph.py`, rather than in `sim/allocation.py` where they originally lived. **Rationale:** both `postprocess/stats.py`'s new Pareto sensitivity breakdown (Section 6.8) and `sim/allocation.py`'s inverse allocator (Section 6.7) need the identical adjoint-transform sensitivity computation; keeping one implementation in `core/frame_graph.py` (a pure function of chain geometry, depending on neither the forward-stats nor inverse-allocation use case) prevents the two call sites from holding two independent copies of the same math that could silently drift apart.
    - Implement `adjoint(T: HTM) -> np.ndarray` shape `(6,6)`: the standard small-angle adjoint transformation block matrix. **Corrected formula (2026-06-24, confirmed by A3 finite-difference cross-check):**
      ```
      Ad_T = [[R,          skew(t)@R],
              [0,          R        ]]
      ```
      using the `[v, ω]` ordering that matches `delta = [dx, dy, dz, rx, ry, rz]`. The top-right block `skew(t)@R` couples the ω input to the v output; the bottom-left block is zero. An earlier draft of this spec mistakenly wrote `[[R, 0],[skew(t)@R, R]]`, which is the `[ω, v]` convention — the finite-difference cross-check in `tests/test_frame_graph.py` caught this and confirmed the corrected form. The adjoint convention must agree with the perturbation convention (Section 2.2.2) or any downstream sensitivity computation is silently wrong.
    - Implement `compute_sensitivity(frame_graph, frame_a, frame_b, edge_names: list[str]) -> np.ndarray` shape `(6, 6*len(edge_names))`: for each named edge on the path between `frame_a` and `frame_b` (typically the full `path_edges_between()` result, or a caller-supplied subset), compute its 6×6 sensitivity block as `adjoint(T_{frame_a → exit_node_i})`, where `exit_node_i` is the node arrived at after traversing edge i (i.e., `edge.child` for forward traversal, `edge.parent` for reverse). **Corrected formula (2026-06-24):** an earlier draft specified `adjoint(T_{exit_node → frame_b})` (the adjoint from the exit node to frame_b), which is incorrect. The correct derivation is: a small perturbation δT at edge i produces `T_perturbed @ T_nominal⁻¹ = T_{frame_a→exit_i} @ T_delta @ T_{frame_a→exit_i}⁻¹`, giving `J_i = Ad_{T_{frame_a→exit_i}}`. Confirmed by the finite-difference cross-check in `tests/test_frame_graph.py`. This function makes no assumption about *why* the caller wants the sensitivity — it is equally valid input to a Pareto variance breakdown (Section 6.8) or an inverse allocation solve (Section 6.7).
14. ✅ **Done (A3, `d81645c`)** Write `tests/test_frame_graph.py` (24 tests):
    - ✅ Cycle detection: `validate_dag()` raises with cycle path.
    - ✅ Multiple-incoming-edge detection: `validate_dag()` raises and names the offending Frame.
    - ✅ Root identification: multi-component graph returns both roots.
    - ✅ Shared-frame (junction) case: `nominal_transform_between()` correctly composes through shared ancestor.
    - ✅ Different-component case: `nominal_transform_between()` raises `DisjointFramesError` with exact locked message text (Section 2.3.1).
    - ✅ Common-physical-base pattern: two independent roots connected via a shared `"base"` Frame; query between the two chains succeeds and propagates base-attachment tolerances. (Section 9.1.3 Common-Ancestor Cancellation Benchmark adds the MC cancellation-quantity assertion on top — that test is in B1-6.)
    - ✅ **Adjoint/sensitivity FD cross-check:** `compute_sensitivity()` finite-difference cross-check (4 tests) — corrected formulas confirmed here before being depended upon by `sim/allocation.py` and `postprocess/stats.py`.

**Interfaces:**

- *Depends on:* `core/transforms.py` (`HTM`), `core/tolerance.py` (`ToleranceSpec6`), `networkx`.
- *Used by:* `sim/monte_carlo_fk.py` (consumes `topological_edge_order()`, `all_edges()`, `root_frames()`), `sim/allocation.py` (consumes `path_edges_between()`, `adjoint()`, and `compute_sensitivity()` for its inverse solve, per the Mod 2 relocation), `postprocess/stats.py` (consumes `weakly_connected_components()` to validate point-pair queries, **and now also `path_edges_between()`/`compute_sensitivity()` for the Pareto sensitivity breakdown, per Mod 2**), `io/schema.py` (constructs a `FrameGraph` from a loaded `Project`), eventually `gui/graph_editor/`.
- *Public API (conceptual):*
  ```
  FrameGraph.add_frame(name, metadata=None)
  FrameGraph.add_edge(parent, child, nominal, tolerance, name=None)
  FrameGraph.validate_dag() -> None  # raises on violation
  FrameGraph.root_frames() -> list[str]
  FrameGraph.weakly_connected_components() -> list[set[str]]
  FrameGraph.topological_edge_order() -> list[str]
  FrameGraph.nominal_transform_between(frame_a, frame_b) -> HTM
  FrameGraph.path_edges_between(frame_a, frame_b) -> list[HTMEdge]
  FrameGraph.all_edges() -> list[HTMEdge]
  FrameGraph.all_frames() -> list[Frame]
  adjoint(T: HTM) -> np.ndarray (6,6)                                          # relocated from sim/allocation.py
  compute_sensitivity(frame_graph, frame_a, frame_b, edge_names) -> np.ndarray  # relocated from sim/allocation.py
  ```

---

## 6.4 `core/conversions.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session; **implemented A1, commit `a31218e`; updated to pytransform3d 3.15.0 non-deprecated API: `matrix_from_euler(e, i=2, j=1, k=0, extrinsic=False)` / `euler_from_matrix(R, i=2, j=1, k=0, extrinsic=False)`. Covered indirectly by `test_transforms.py` round-trip tests.**)*

**Responsibility:** The sole, isolated point of contact with `pytransform3d`. Every conversion between a supported user-facing input format and a raw 4x4 matrix is implemented here; `core/transforms.py` calls into this module rather than importing `pytransform3d` itself.

**Deliverables:**

- One conversion function per supported format, both directions (to-matrix and from-matrix)
- All functions operate on/return plain `np.ndarray`, never a `pytransform3d`-specific type, so the rest of the codebase has zero `pytransform3d` type exposure
- `tests/test_transforms.py` round-trip coverage exercises this module indirectly (no separate test file needed, since this module has no behavior beyond what `HTM`'s round-trip tests already cover)

**Granular Task List:**

1. Implement `euler_to_rotation_matrix(euler_angles, convention="intrinsic_zyx") -> np.ndarray (3,3)`, wrapping the appropriate `pytransform3d.rotations` function for the locked convention.
2. Implement `rotation_matrix_to_euler(R, convention="intrinsic_zyx") -> np.ndarray (3,)`, the inverse of Step 1.
3. Implement `quaternion_to_rotation_matrix(quat_wxyz) -> np.ndarray (3,3)` and `rotation_matrix_to_quaternion(R) -> np.ndarray (4,)` (wxyz order — document explicitly, since `pytransform3d` and other libraries vary on wxyz vs. xyzw ordering, a classic source of silent bugs).
4. Implement `screw_to_matrix(axis, angle, translation_along_axis, point_on_axis=None) -> np.ndarray (4,4)` and `matrix_to_screw(T) -> dict(axis, angle, translation_along_axis, point_on_axis)`, wrapping `pytransform3d`'s exponential-coordinate utilities; explicitly branch on the zero-angle degenerate case (pure translation has no well-defined rotation axis).
5. Add a single module-level test or assertion confirming the installed `pytransform3d` version matches what's pinned in `requirements.txt` (cheap insurance against silent convention changes in a future library upgrade).
6. Document, in the module docstring, the exact ordering convention for every format (quaternion wxyz, Euler intrinsic ZYX, screw axis normalization) — this is the single reference point for "which convention did we pick" so it never needs to be re-derived from code.

**Interfaces:**

- *Depends on:* `pytransform3d` (the only module in the entire codebase that imports it directly).
- *Used by:* `core/transforms.py` exclusively.
- *Public API (conceptual):*
  ```
  euler_to_rotation_matrix(euler_angles, convention) -> np.ndarray (3,3)
  rotation_matrix_to_euler(R, convention) -> np.ndarray (3,)
  quaternion_to_rotation_matrix(quat_wxyz) -> np.ndarray (3,3)
  rotation_matrix_to_quaternion(R) -> np.ndarray (4,)
  screw_to_matrix(axis, angle, translation_along_axis, point_on_axis=None) -> np.ndarray (4,4)
  matrix_to_screw(T) -> dict
  ```

---

## 6.5 `core/sampling.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session — relocated from `sim/sampling.py` this revision (see architecture correction note at top of Section 6); **implemented A2, commit `3ac0eed`; covered indirectly by `test_tolerance.py`**)*

**Responsibility:** Pure distribution-sampling math: given an already-constructed `np.random.Generator`, a bound, and (for normal) a sigma-level, draw samples. This module knows nothing about edges, Frames, or simulation runs — that bookkeeping lives in `sim/monte_carlo_fk.py` (Section 6.6). This module is intentionally "dumb" and reusable.

**Deliverables:**

- `sample_uniform(bound, n_trials, rng) -> np.ndarray`
- `sample_normal(bound, sigma_level, n_trials, rng) -> np.ndarray`
- A single dispatch function used by `ToleranceSpec.sample()`
- `tests/test_tolerance.py` covers this indirectly (no dedicated test file needed — see Section 6.2 Step 10)

**Granular Task List:**

1. Implement `sample_uniform(bound, n_trials, rng) -> np.ndarray` shape `(n_trials,)`: draw from `Uniform(-bound, +bound)` via `rng.uniform(-bound, bound, size=n_trials)`.
2. Implement `sample_normal(bound, sigma_level, n_trials, rng) -> np.ndarray` shape `(n_trials,)`: convert `bound`/`sigma_level` to a standard deviation (`sigma = bound / sigma_level`), draw via `rng.normal(0.0, sigma, size=n_trials)`.
3. Implement `sample(distribution, bound, sigma_level, n_trials, rng) -> np.ndarray`: a single dispatch function (`if distribution == "uniform": ... elif "normal": ...`) — this is the function `ToleranceSpec.sample()` actually calls, keeping `core/tolerance.py` free of any distribution-specific branching.
4. Edge case: `bound == 0` should deterministically return an all-zero array (not error, not draw degenerate noise) for both distributions — this is the valid representation of "no tolerance on this DoF."

**Interfaces:**

- *Depends on:* `numpy` only.
- *Used by:* `core/tolerance.py` (`ToleranceSpec.sample()`).
- *Public API (conceptual):*
  ```
  sample_uniform(bound, n_trials, rng) -> np.ndarray (n_trials,)
  sample_normal(bound, sigma_level, n_trials, rng) -> np.ndarray (n_trials,)
  sample(distribution, bound, sigma_level, n_trials, rng) -> np.ndarray (n_trials,)
  ```

---

## 6.6 `sim/monte_carlo_fk.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session; **implemented A4, commit `744c562`, 18 tests passing. RNG scheme: SHA-256(edge_name)[:8] → spawn_key + SeedSequence([master_seed, spawn_key]). Composition: np.einsum("nij,njk->nik", ...). edge_seed_log stores integers (spawn keys), not generators.**)*

**Responsibility:** The forward tolerance verification engine (Mode 1, Section 3.1) and the owner of the `TrialData` structure (defined at the top of Section 6). Also owns per-edge RNG sub-stream derivation, per this session's locked decision.

**Deliverables:**

- `TrialData` dataclass (defined at top of Section 6)
- `make_edge_rng(master_seed, edge_name) -> np.random.Generator` — deterministic, edge-keyed sub-stream derivation
- `MonteCarloFKEngine.run(frame_graph, n_trials, seed) -> TrialData`
- Fully vectorized chain composition (no Python-level loop over individual trials)
- `tests/test_monte_carlo_fk.py`, including the dedicated shared-edge consistency regression test required by Section 9, Item 3

**Granular Task List:**

1. Implement `make_edge_rng(master_seed: int, edge_name: str) -> np.random.Generator`: derive a deterministic spawn key from `edge_name` (e.g., a stable string hash, such as the first 8 bytes of `hashlib.sha256(edge_name.encode()).digest()` interpreted as an integer), and construct `np.random.default_rng(np.random.SeedSequence([master_seed, spawn_key]))`. Document the exact hashing scheme in the module docstring — this is the kind of detail that must never silently change, since changing it would break reproducibility of every previously-recorded result.
   - **Why this matters (document in code):** this guarantees that adding, removing, or modifying *other* edges in the graph never changes the random draws for *this* edge, for a fixed `master_seed` and `n_trials` — a property that would not hold under a single shared global RNG stream consumed in topological order.
2. Define the `TrialData` dataclass exactly per the Section 6 top-level contract (`n_trials`, `seed`, `frame_poses`, `nominal_poses`, `edge_seed_log`).
3. Implement `MonteCarloFKEngine.run(frame_graph: FrameGraph, n_trials: int, seed: int) -> TrialData`:
   a. Call `frame_graph.validate_dag()` as a precondition (do not trust the caller to have already done this).
   b. Compute `root_frames = frame_graph.root_frames()`; initialize `frame_poses[root] = np.tile(np.eye(4), (n_trials,1,1))` and `nominal_poses[root] = np.eye(4)` for each root.
   c. Iterate edges in `frame_graph.topological_edge_order()`. For each edge:
      - Derive `rng = make_edge_rng(seed, edge.name)`; record `edge_seed_log[edge.name]`.
      - Sample `delta_batch = edge.tolerance.sample(n_trials, rng)` (shape `(N,6)`).
      - Compute `perturbed_batch = apply_perturbation_batch(edge.nominal, delta_batch)` (shape `(N,4,4)`) — from `core/tolerance.py`.
      - Compose: `frame_poses[edge.child] = frame_poses[edge.parent] @ perturbed_batch` (batched matrix multiply, vectorized over the leading `N` axis — verify this is implemented as one vectorized operation, not a Python loop over `range(n_trials)`).
      - Compute `nominal_poses[edge.child] = nominal_poses[edge.parent].matrix @ edge.nominal.matrix` (single 4x4, unperturbed reference).
   d. Return the populated `TrialData`.
4. Performance check: confirm step 3c's batched composition is implemented via `np.einsum('nij,njk->nik', ...)` or equivalent broadcasted `@` — add a one-line comment/assertion in code (or a test) confirming no per-trial Python loop exists, since this is explicitly called out as a non-negotiable performance requirement (Section 5.1/2.5).
5. ✅ **Done (A4, `744c562`)** Write `tests/test_monte_carlo_fk.py` (18 tests):
   - ✅ **2-edge chain hand-check:** `TestTwoEdgeChainHandCheck` — uses `_FixedToleranceSpec6` duck-typed helper (more robust than seed-replication) to give exact deterministic deltas; checks `frame_poses["B"]` and `frame_poses["C"]` translation columns against hand-computed values.
   - ✅ **3-edge chain hand-check:** `TestThreeEdgeChainHandCheck` — includes a π/4 nominal rotation on the middle edge, hand-computes expected D pose by matrix multiplication.
   - ✅ **Shared-edge consistency test:** `TestSharedEdgeConsistency` — 3 tests covering leaf1==shared, leaf2==shared, and shared-poses-are-nontrivial. **Note:** Section 9 Item 3 also requires a standalone named module-level function, which was added in A6 as `test_shared_edge_sampling_consistency` in `tests/test_integration.py` (Section 6.20 Item 3).
   - ✅ **Reproducibility test:** `TestReproducibility` — same seed gives bit-for-bit identical output; adding an unrelated disconnected edge doesn't change existing frames' draws.
   - ✅ **Root-anchor test:** `TestRootAnchor` — root frame is identity every trial; multi-component both roots are identity.

**Interfaces:**

- *Depends on:* `core/frame_graph.py` (`FrameGraph`, `HTMEdge`), `core/tolerance.py` (`apply_perturbation_batch`), `core/transforms.py` (`HTM`), `numpy`.
- *Used by:* `postprocess/stats.py` and `postprocess/bounding_shapes.py` (consume `TrialData`), `sim/allocation.py` (uses this engine internally for its MC validation pass), eventually `gui/run_panel/`.
- *Public API (conceptual):*
  ```
  TrialData(n_trials, seed, frame_poses, nominal_poses, edge_seed_log)
  make_edge_rng(master_seed: int, edge_name: str) -> np.random.Generator
  MonteCarloFKEngine.run(frame_graph: FrameGraph, n_trials: int, seed: int) -> TrialData
  ```

---

## 6.7 `sim/allocation.py`

*(Last revised: 2026-06-23 — Claude, design-spec update adding the iterative nonlinear damping loop per cross-review; further revised same day to remove the now-relocated `adjoint()`/`compute_sensitivity()` implementation per Mod 2 — see Section 6.3)*

**Responsibility:** The inverse tolerance allocation engine (Mode 2, Section 3.2): consumes the shared sensitivity primitive from `core/frame_graph.py` (Section 6.3) to solve the equal-allocation objective, then validates and iteratively corrects the proposed tolerance set via Monte Carlo.

**Deliverables:**

- `AllocationObjective` interface + `EqualAllocation` implementation, producing the **baseline linear allocation** (built on top of `core/frame_graph.py`'s relocated `compute_sensitivity()` — this module implements no Jacobian math of its own)
- `AllocationEngine.allocate(...)` — the top-level entry point, wrapping the baseline linear solve in an **iterative nonlinear damping/correction loop** (locked decision, this revision — see Step 5 below)
- `AllocationEngine.solve(...)` (baseline linear step) and `.validate(...)` (single MC validation pass), both retained as the internal building blocks `allocate()` composes
- `AllocationResult` output structure explicitly preserving **both** the uncorrected linear baseline and the final nonlinearly-corrected allocation, plus convergence status
- `tests/test_allocation.py`, covering allocation-specific behavior (the Jacobian's own correctness is now tested in `tests/test_frame_graph.py`, Section 6.3) and dedicated tests for the damping loop's convergence and non-convergence paths

**Granular Task List:**

1. **Sensitivity derivation — relocated this revision (Mod 2, cross-review):** `compute_sensitivity()` and its underlying `adjoint()` helper now live in `core/frame_graph.py` (Section 6.3, Step 13), not here — both `sim/allocation.py` and `postprocess/stats.py`'s new Pareto sensitivity breakdown (Section 6.8) call the same shared primitive, so the math is implemented exactly once. This module's job is to call `frame_graph.path_edges_between(frame_a, frame_b)` to get the relevant edges, then `compute_sensitivity(frame_graph, frame_a, frame_b, free_edge_names)` to get the `(6, 6*len(free_edge_names))` sensitivity matrix used by `EqualAllocation.solve()` (Step 3) — do not re-implement the adjoint transform here.
2. Implement the `AllocationObjective` abstract interface: a `solve(sensitivity_matrix, target_bound_vector, free_edge_names) -> dict[edge_name, ToleranceSpec6]` method signature, to allow future objectives beyond v1's `EqualAllocation` without changing the calling code in `AllocationEngine`.
3. Implement `EqualAllocation(AllocationObjective)`: solves for a single uniform scale factor `s` applied to all free edges' (currently-unset or placeholder) per-DoF bounds such that the linear worst-case sum of contributions (via the sensitivity matrix from Step 1) equals the target bound, per DoF. This reduces to a simple linear equation per target DoF (closed-form, no iterative optimizer needed for v1) — document the exact formula once derived, including how multiple target DoF constraints are reconciled into a single scale factor (e.g., take the most restrictive/binding DoF, or solve a small least-squares system — **decide and document explicitly once this task is reached; flag as an open implementation decision for the Milestone B-2 session that builds this module**).
4. Implement `AllocationEngine.solve(frame_graph, frame_a, frame_b, target_tolerance: ToleranceSpec6, objective=EqualAllocation()) -> dict[edge_name, ToleranceSpec6]`:
   - Identify all edges on the path between `frame_a` and `frame_b` via `frame_graph.path_edges_between(frame_a, frame_b)` (Section 6.3, Step 12).
   - Partition into free (unlocked) and locked edges.
   - Compute locked edges' fixed contribution to the budget (their existing `ToleranceSpec6` propagated through the same `compute_sensitivity()` primitive, Section 6.3).
   - Call `objective.solve(...)` on the remaining free-edge budget.
   - Return the full proposed per-edge `ToleranceSpec6` set (locked edges unchanged, free edges populated per the objective's solution).
5. **Iterative nonlinear damping/correction loop (locked decision, this revision):** Long serial chains with high-leverage joints exhibit meaningful geometric cross-coupling (`dx ≈ L·θ` — a small angular tolerance on an upstream joint sweeps a large positional arc by the time it reaches a distant downstream Frame). The closed-form linear allocation from Steps 1–4 does not account for this, and can produce an allocation that passes the linear sensitivity math but fails nonlinear Monte Carlo validation. `AllocationEngine.allocate()` is the top-level method that wraps `solve()` and `validate()` in a correction loop to address this:
   a. Call `solve(...)` (Steps 1–4) to produce the **baseline linear allocation**. Retain this unmodified — it is never overwritten (Step 5e).
   b. Call `validate(frame_graph, baseline_allocation, frame_a, frame_b, n_trials=1000, seed)` (Step 6) — a deliberately low-overhead `N_validate = 1000`-trial Monte Carlo pass, fast enough to run on every loop iteration without materially slowing down an interactive allocation workflow.
   c. If the validated achieved envelope is within the target bound (per DoF) for every DoF, the baseline allocation is accepted as final with no correction — record `iterations_used = 0`, `converged = True`.
   d. If validation fails (the achieved envelope exceeds the target on at least one DoF), apply a uniform damping factor `gamma` to the **free edges' angular DoF bounds only** (`rx, ry, rz` — translational DoF bounds are not damped by this loop, since the geometric-leverage failure mode this addresses is specifically angular-to-positional coupling): `new_bound = current_bound * gamma`, with `gamma` a tunable parameter, **default range `[0.7, 0.95]`** (locked this revision — document the exact default value chosen within that range once implemented, e.g., a fixed `gamma = 0.9` per iteration, or a configurable parameter exposed to the caller). Re-run `validate()` on the damped allocation and repeat.
   e. **Termination:** cap the loop at `max_iter = 10` iterations (locked this revision). If the loop satisfies validation before reaching the cap, return with `converged = True` and the iteration count used. If the cap is reached without satisfying validation, **terminate gracefully (do not raise an exception that crashes the caller)** and return `converged = False` with the status message **exactly**: `"Allocation could not converge to target budget"`. The caller (eventually the GUI's run panel, Section 6.16) is responsible for surfacing this status to the user — the engine itself does not treat non-convergence as a fatal error, since a non-converging allocation is itself useful diagnostic information (it tells the engineer their target is infeasible given the current locked edges and chain geometry, not just that the tool failed).
   f. **Output structure — `AllocationResult`:** the dataclass returned by `allocate()` must explicitly carry **both** allocations, never overwriting one with the other:
      ```
      AllocationResult
      ├── baseline_linear_allocation: dict[str, ToleranceSpec6]   # uncorrected, from Step 5a
      ├── corrected_allocation: dict[str, ToleranceSpec6]          # final, possibly == baseline if no correction was needed
      ├── converged: bool
      ├── iterations_used: int
      ├── status_message: str | None   # set to the locked non-convergence string on failure, else None
      └── final_validation_report: ValidationReport                # from the last validate() call in the loop
      ```
      The explicit preservation of both allocations is intentional and load-bearing: the *difference* between the baseline and corrected allocations is itself a direct, quantitative diagnostic of how much geometric-leverage coupling exists in the user's chain — a chain needing heavy damping (many iterations, large cumulative `gamma` reduction) is telling the engineer something real about their design's sensitivity to high-leverage joints, and that information would be lost if only the final corrected result were exposed.
6. Implement `AllocationEngine.validate(frame_graph, proposed_tolerances, frame_a, frame_b, n_trials, seed) -> ValidationReport`:
   - Build a temporary `FrameGraph` (or mutate a copy) with the proposed tolerances applied to the free edges.
   - Run `MonteCarloFKEngine.run(...)` (Section 6.6) on it.
   - Compute the achieved relative-pose envelope between `frame_a` and `frame_b` from the resulting `TrialData` (via `postprocess/stats.py`, Section 6.8).
   - Compare achieved vs. target per DoF; populate a `ValidationReport` with both values and the discrepancy (absolute and percentage), and a per-DoF pass/fail flag (consumed directly by Step 5's loop termination check).
7. Write `tests/test_allocation.py`:
   - **Sensitivity primitive correctness:** not re-derived here — `compute_sensitivity()`'s finite-difference cross-check now lives in `tests/test_frame_graph.py` (Section 6.3, Step 14), since the primitive itself was relocated there this revision. This module's tests assume that primitive is already validated and focus on allocation-specific behavior below.
   - **Equal-allocation sanity check:** simple 2–3 edge chain, all free, confirm the solved baseline tolerances are indeed equal (or equally scaled) across edges per the objective's definition.
   - **Locked-edge case:** one edge locked to a known value, confirm the solver only adjusts the free edges and correctly accounts for the locked edge's fixed contribution.
   - **All-edges-locked edge case:** confirm the solver detects an infeasible/over-constrained situation (no free edges to solve for) and raises a clear, specific error rather than silently returning a meaningless result.
   - **MC validation discrepancy reporting:** confirm `validate()` correctly flags a case where the linear allocation under- or over-shoots the nonlinear MC-validated result (construct a case with deliberately large nominal rotation offsets between edges to induce a meaningful nonlinearity, since pure small-angle propagation through near-zero nominal offsets may not exercise this path).
   - **Damping loop convergence test (new, locked decision):** using the same high-leverage geometry constructed for the discrepancy test above, confirm `allocate()` converges within `max_iter`, confirm `corrected_allocation`'s angular bounds are strictly tighter than `baseline_linear_allocation`'s, and confirm the final validation pass actually satisfies the target.
   - **Damping loop non-convergence test (new, locked decision):** construct a deliberately infeasible target (e.g., an unreasonably tight target on a chain with most edges locked) and confirm `allocate()` returns `converged=False` with `status_message == "Allocation could not converge to target budget"` after exactly `max_iter` iterations, without raising an uncaught exception.

**Interfaces:**

- *Depends on:* `core/frame_graph.py` (`FrameGraph`, `path_edges_between()`, `adjoint()`, `compute_sensitivity()` — relocated here per Mod 2), `core/tolerance.py` (`ToleranceSpec6`), `core/transforms.py` (`HTM`), `sim/monte_carlo_fk.py` (`MonteCarloFKEngine`, for the validation pass — called once per damping-loop iteration, up to `max_iter` times), `postprocess/stats.py` (envelope computation for validation), `scipy.optimize` (reserved for future non-closed-form objectives — not required for v1's `EqualAllocation`).
- *Used by:* eventually `gui/run_panel/` and `gui/results_viewer/` (Milestone B-2) — the GUI's "Run" action for IK mode calls `allocate()`, not `solve()` directly, so the damping correction is always applied by default.
- *Public API (conceptual):*
  ```
  AllocationObjective.solve(sensitivity_matrix, target_bound_vector, free_edge_names) -> dict
  EqualAllocation(AllocationObjective)
  AllocationEngine.solve(frame_graph, frame_a, frame_b, target_tolerance, objective) -> dict[str, ToleranceSpec6]
  AllocationEngine.validate(frame_graph, proposed_tolerances, frame_a, frame_b, n_trials, seed) -> ValidationReport
  AllocationEngine.allocate(frame_graph, frame_a, frame_b, target_tolerance, objective=EqualAllocation(),
                             n_validate=1000, gamma=0.9, max_iter=10, seed=...) -> AllocationResult
  AllocationResult(baseline_linear_allocation, corrected_allocation, converged, iterations_used,
                    status_message, final_validation_report)
  # Note: adjoint() and compute_sensitivity() are NOT part of this module's API —
  # they live in core/frame_graph.py (Section 6.3) and are imported from there.
  ```

---

## 6.8 `postprocess/stats.py`

*(Last revised: 2026-06-24 — Claude, design-spec update adding the Pareto sensitivity breakdown per Mod 2, cross-review; **Steps 1–7 implemented in A5, commit `019eb34`, 26 tests passing. Steps 8–9 deferred to Milestone B-1 task B1-3.** Rotation-error extraction decision: `scipy.spatial.transform.Rotation.from_matrix(R_error).as_rotvec()` — exact matrix logarithm, equivalent to skew-extraction at small angles but more robust at moderate angles; documented in module docstring.)*

**Responsibility:** Computes envelope statistics for any single Frame, and for the relative pose between any two Frames in the same component, directly from a `TrialData` instance — no re-simulation, no graph traversal beyond a same-component validation check. **As of this revision, also computes the Pareto-sorted per-edge sensitivity breakdown** (Mod 2) — answering "which tolerance edge dominates this error budget" as a standalone forward-analysis capability, independent of whether inverse allocation is ever run.

**Deliverables:**

**A5 scope (Steps 1–7) — ✅ Implemented (commit `019eb34`, 26 tests):**
- Per-Frame envelope/percentile/histogram-data functions (`frame_envelope_box`, `frame_percentiles`, `frame_histogram_data`)
- Point-pair relative-pose statistics: `relative_pose_trials`, `relative_pose_nominal`, `point_pair_envelope_box` — exploiting the "absolute pose already stored" property (Section 6 top-level note)
- `pose_error_vector_batch()` — the shared extraction primitive producing `(N,6)` error vectors in `[dx,dy,dz,rx,ry,rz]` order with rotvec `ω=θu` rotation columns compatible with `postprocess/bounding_shapes.py`
- `tests/test_stats.py` — 26 tests, including shared-ancestor cancellation integration check

**B1-3 scope (Steps 8–9) — Deferred to Milestone B-1:**
- `compute_tolerance_sensitivities()` — Pareto-sorted percentage-contribution breakdown per edge/DoF, built on the shared `compute_sensitivity()` primitive relocated to `core/frame_graph.py` (Section 6.3)
- `ParetoSensitivityReport` dataclass + `to_ascii_chart()` rendering hook
- Pareto breakdown correctness tests (to be added to `tests/test_stats.py` in B1-3)

**Granular Task List:**

1. Implement `pose_error_vector_batch(poses: np.ndarray, nominal: np.ndarray) -> np.ndarray` shape `(N,6)`: extract translation error directly (`poses[:, :3, 3] - nominal[:3, 3]`) and rotation error as the small-angle rotation vector **ω = θu** (axis `u` scaled by angle `θ` — the same notation locked for `postprocess/bounding_shapes.py`, Section 6.9 Step 4) via the small-angle log map (`ω ≈` the skew-symmetric part of `R_error - I` extracted via the inverse of the Section 6.2 skew operation, or via `scipy.spatial.transform.Rotation`'s `as_rotvec()` on `R_nominal.T @ R_perturbed` — decide and document which, confirm both give equivalent results to small-angle order, then standardize on one for consistency with `core/tolerance.py`'s forward construction). This function's columns `[3:6]` of the `(N,6)` output are exactly the `(N,3)` `rotvecs` array that `postprocess/bounding_shapes.py`'s `fit_rotation_cone()`/`fit_rotation_box()` require — no further conversion needed between the two modules.
2. Implement `frame_envelope_box(trial_data, frame_name) -> dict`: per-DoF min/max of `pose_error_vector_batch` for the named frame — the axis-aligned worst-case box (Section 1.1, primary deliverable).
3. Implement `frame_percentiles(trial_data, frame_name, percentiles: list[float]) -> dict`: per-DoF percentile table (e.g., 0.1/2.5/50/97.5/99.9), useful for the statistical (`"normal"`-distribution) comparison mode.
4. Implement `frame_histogram_data(trial_data, frame_name, dof_index, bins=50) -> tuple[counts, bin_edges]`: thin wrapper over `np.histogram` for the named DoF, feeding `postprocess/reporting.py`.
5. Implement `relative_pose_trials(trial_data, frame_graph, frame_a, frame_b) -> np.ndarray` shape `(N,4,4)`:
   - Validate `frame_a` and `frame_b` are in the same weakly-connected component (via `frame_graph.weakly_connected_components()`) — raise a clear, specific error if not (this is the one place `postprocess/stats.py` needs `FrameGraph` at all, purely for this validation).
   - Compute `inverse(trial_data.frame_poses[frame_a][i]) @ trial_data.frame_poses[frame_b][i]` for every trial `i`, vectorized (batched inverse + batched matmul, no Python loop).
6. Implement `relative_pose_nominal(trial_data, frame_a, frame_b) -> np.ndarray` shape `(4,4)`: same as Step 5 but using `trial_data.nominal_poses`, for use as the reference point in error-vector extraction.
7. Implement `point_pair_envelope_box(trial_data, frame_graph, frame_a, frame_b) -> dict`: combines Steps 1, 5, and 6 to produce the same kind of per-DoF min/max box as Step 2, but for the *relative* pose between two arbitrary Frames — this is the function that directly satisfies the cross-chain optical-alignment use case (Section 3.3).
8. **[DEFERRED TO B1-3] New this revision (Mod 2, cross-review) — `compute_tolerance_sensitivities(frame_graph, frame_a, frame_b) -> ParetoSensitivityReport`:** computes the first-order percentage contribution of each toleranced edge/DoF on the path between `frame_a` and `frame_b` to the total variance of the target relative pose.
   - Use `frame_graph.path_edges_between(frame_a, frame_b)` (Section 6.3, Step 12) to get the relevant edges, and `compute_sensitivity(frame_graph, frame_a, frame_b, edge_names)` (Section 6.3, Step 13) to get the shared `(6, 6*len(edges))` sensitivity matrix — **do not implement a second copy of the Jacobian here; this is the entire point of the Mod 2 relocation.**
   - For each edge's each DoF, compute its variance contribution as `(sensitivity_block_column)² * variance(DoF)`, where `variance(DoF)` is `bound²/3` for `"uniform"` tolerances and `sigma²` (`= (bound/sigma_level)²`) for `"normal"` tolerances — both reduce consistently to the same variance-propagation math, which is also what the RSS benchmark in Section 9.1 validates against.
   - Sum each edge/DoF's variance contribution across all 6 output DoF (weighted by which output DoF the user actually cares about, or summed unweighted across all 6 if no specific weighting is given — **decide and document the exact weighting scheme once this task is reached**), normalize by the total variance, and sort descending — this produces the Pareto breakdown.
   - Return a `ParetoSensitivityReport` dataclass: an ordered list of `(edge_name, dof_label, percentage_contribution)` tuples, plus the total variance they sum to, ready for direct display as the Pareto chart shown in Mod 2's example output (`Stage_X_Tilt (rx) 45.2%`, etc.) or for `postprocess/reporting.py` to render as a horizontal bar chart.
   - **Document explicitly, next to this function's docstring and in any rendered report:** this is a **first-order linear approximation** (the same small-angle adjoint Jacobian used by `sim/allocation.py`'s baseline allocation, Section 6.7). For chains with significant geometric leverage (`dx ≈ L·θ`, the same nonlinearity that motivates the allocation engine's damping loop), the true nonlinear variance contribution of a given edge can differ from this linear estimate — this caveat must not be silently omitted, since a Pareto chart that looks authoritative but is quietly approximate could mislead a sourcing decision (Section 1.4).
9. **[DEFERRED TO B1-3]** Implement `ParetoSensitivityReport`'s rendering hook (a `to_ascii_chart()` or similar convenience method) that produces the bar-chart-style text output shown in Mod 2 — useful for quick terminal/script output in Milestone B-1's example scripts (Section 6.19) ahead of any GUI rendering.
10. Write `tests/test_stats.py` — **A5 tests ✅ implemented (26 tests); B1-3 Pareto tests deferred:**
   - ✅ Construct a small synthetic `TrialData` with known, hand-computed pose errors (don't run the full MC engine — directly build the `TrialData` fields) and confirm `frame_envelope_box` returns the expected min/max.
   - ✅ Confirm `relative_pose_trials` between a Frame and itself returns identity for every trial (trivial sanity check).
   - ✅ Confirm `point_pair_envelope_box` between two Frames sharing a common upstream edge correctly reflects that the shared edge's contribution cancels out of the *relative* error (a key qualitative check that validates the whole point of Section 2.4's shared-sampling design — relative tolerance between two downstream points should be tighter than either point's absolute tolerance when they share a noisy common ancestor edge).
   - ✅ Confirm the different-component case raises a clear error.
   - **[DEFERRED TO B1-3] Pareto sensitivity correctness (new, Mod 2):** construct a simple chain with one dominant-tolerance edge and several much-tighter edges, confirm `compute_tolerance_sensitivities()` ranks the dominant edge first with a percentage contribution that matches a hand-computed variance ratio.
   - **[DEFERRED TO B1-3] Pareto contributions sum to ~100%:** confirm the reported percentages across all edge/DoF entries sum to approximately 100% (allowing for the "Others" bucketing shown in Mod 2's example output, if implemented) for a representative chain.
   - **[DEFERRED TO B1-3] Uniform vs. normal consistency:** confirm the variance formula (`bound²/3` for uniform, `sigma²` for normal) produces self-consistent Pareto rankings when the same chain is run once with all-uniform and once with all-normal tolerances at equivalent variance — this is the same consistency the RSS benchmark (Section 9.1) checks at the FK level, applied here at the sensitivity-breakdown level.

**Interfaces:**

- *Depends on:* `sim/monte_carlo_fk.py` (`TrialData`), `core/frame_graph.py` (`FrameGraph`, for same-component validation, and **as of this revision** `path_edges_between()`/`compute_sensitivity()` for the Pareto breakdown — Mod 2), `core/tolerance.py` (`ToleranceSpec6`, to read each edge's distribution/bound/sigma_level for the variance calculation in Step 8), `numpy`, `scipy.spatial.transform` (if used for rotation error extraction per Step 1).
- *Used by:* `postprocess/bounding_shapes.py` (consumes the error vectors/point clouds this module produces), `postprocess/reporting.py` (consumes histogram data, envelope boxes, **and now the `ParetoSensitivityReport`** for plotting/rendering), `sim/allocation.py` (consumes `point_pair_envelope_box` during the MC validation pass), eventually `gui/results_viewer/` and `gui/point_pair_panel/`.
- *Public API (conceptual):*
  ```
  pose_error_vector_batch(poses, nominal) -> np.ndarray (N,6)
  frame_envelope_box(trial_data, frame_name) -> dict
  frame_percentiles(trial_data, frame_name, percentiles) -> dict
  frame_histogram_data(trial_data, frame_name, dof_index, bins) -> (counts, edges)
  relative_pose_trials(trial_data, frame_graph, frame_a, frame_b) -> np.ndarray (N,4,4)
  point_pair_envelope_box(trial_data, frame_graph, frame_a, frame_b) -> dict
  compute_tolerance_sensitivities(frame_graph, frame_a, frame_b) -> ParetoSensitivityReport   # new, Mod 2
  ParetoSensitivityReport(ranked_contributions: list[tuple[str, str, float]], total_variance: float)
  ParetoSensitivityReport.to_ascii_chart() -> str
  ```

---

## 6.9 `postprocess/bounding_shapes.py`

*(Last revised: 2026-06-23 — Claude, design-spec update hardening the rotation-vector type contract per cross-review)*

**Responsibility:** Fits bounding shapes (box, ellipsoid/sphere for translation; cone or per-axis box for rotation) to the error-vector point clouds produced by `postprocess/stats.py`. This is the module responsible for the "bounding shape the engineer can use to make decisions" deliverable (Section 1.1, Section 3.1).

**Deliverables:**

- Axis-aligned bounding box fitting (translation and rotation, trivial — already substantially covered by `stats.py`'s envelope functions; this module focuses on the non-trivial shapes)
- Bounding ellipsoid/sphere fitting for translation point clouds
- Bounding cone fitting for rotation-vector point clouds — **confirmed lead representation, locked 2026-06-23** (per-axis box still implemented as a cheap secondary cross-check, not as the primary reported value)
- `tests/test_bounding_shapes.py` (new test file — add to the tree)

**Granular Task List:**

1. Implement `fit_bounding_sphere(points: np.ndarray) -> dict(center, radius)` for a `(N,3)` translation point cloud — minimum enclosing sphere (e.g., Welzl's algorithm, or a simpler/conservative approach such as centroid + max distance, which is non-optimal but simple and always-correct as a bound — **recommend the simpler conservative approach for v1**, since this is an error-budgeting tool, not a computational-geometry showcase, and a slightly loose conservative bound is preferable to added complexity/risk of an exact-minimum-enclosing-sphere implementation bug).
2. Implement `fit_bounding_ellipsoid(points: np.ndarray, coverage=1.0) -> dict(center, axes_lengths, axes_directions)`: for `coverage=1.0` (full worst-case), fit via the covariance-matrix/eigenvalue approach scaled to just enclose all points; for `coverage<1.0` (e.g., 0.997 for a 3σ-equivalent statistical ellipsoid), scale using the appropriate chi-squared quantile for 3 degrees of freedom. Document this distinction clearly, since "ellipsoid that bounds 100% of points" and "ellipsoid that bounds 99.7% of points statistically" are different objects answering different questions (worst-case vs. statistical, per Section 2.5).
3. **Rotation bounding shape — locked decision (2026-06-23): the cone is the lead representation.** Implement both of the following, but treat the cone as the primary reported value everywhere a single number/shape is needed (e.g., a results-viewer headline figure, a one-line summary in a report) — the box remains available as a secondary, more granular cross-check:
   - `fit_rotation_cone(rotvecs: np.ndarray) -> dict(max_angle, mean_axis)` — **primary.** Single worst-case tilt-angle-from-nominal magnitude, regardless of direction, plus the mean tilt axis for reference. This is the number an engineer reads first when asking "how far off-axis could this interface tip?"
   - `fit_rotation_box(rotvecs: np.ndarray) -> dict(min, max)` — **secondary.** Simple per-axis worst-case bounds (3 independent angle bounds), kept available for cases where the *direction* of angular error matters (e.g., pitch is much more sensitive than yaw for a given optical system) and a single isotropic cone magnitude would obscure that asymmetry.
4. **Interface type-hardening (locked decision, this revision):** `fit_rotation_cone()` and `fit_rotation_box()` must be explicitly typed and documented to accept **only** pre-extracted small-angle rotation-vector arrays — `rotvecs: np.ndarray` of shape `(N,3)`, where each row is the axis-scaled-by-angle vector **ω = θu** (`u` the unit rotation axis, `θ` the small rotation angle; this is exactly what `postprocess/stats.py`'s `pose_error_vector_batch()`, Section 6.8 Step 1, already produces as the rotation half of its `(N,6)` output). Document this convention by name (`ω = θu`) in both functions' docstrings, not just by shape/dtype.
   - These two functions must **never** accept a raw `(N,4,4)` HTM/pose array directly, and must not internally re-derive a rotation vector from one — doing so would risk reintroducing multi-axis coordinate-coupling artifacts (e.g., naively reading off individual matrix entries as if they were independent per-axis angles) that the dedicated small-angle log-map extraction in `postprocess/stats.py` is specifically designed to avoid. Enforce this with a type hint and a runtime shape check (`assert rotvecs.shape[1] == 3`, raising a clear error on a `(N,4,4)` array passed by mistake) rather than relying on the type hint alone.
5. Implement `fit_bounding_box(points: np.ndarray) -> dict(min, max)`: thin, explicit axis-aligned box fit (kept here for symmetry with the ellipsoid/cone fitters, even though `postprocess/stats.py`'s `frame_envelope_box` already effectively computes this for the 6-DoF case — this version operates on arbitrary 3D point clouds for reuse by both translation and rotation-box fits).
6. Write `tests/test_bounding_shapes.py`:
   - Synthetic point cloud with a known, constructed bounding sphere/ellipsoid (e.g., points placed exactly on a known ellipsoid surface) — confirm the fitted shape matches within numerical tolerance.
   - Confirm the `coverage<1.0` statistical ellipsoid is strictly smaller than the `coverage=1.0` worst-case ellipsoid for the same point cloud (sanity check on the chi-squared scaling).
   - Confirm `fit_rotation_box` and `fit_rotation_cone` agree on simple, symmetric synthetic cases (e.g., isotropic rotation-vector noise) where both representations should imply the same effective bound — this remains a useful cross-check even with the cone as the lead representation, since it catches a fitting bug in either function.
   - **Type-hardening test (new, locked decision):** confirm passing a `(N,4,4)` array to `fit_rotation_cone()`/`fit_rotation_box()` raises a clear shape-mismatch error rather than silently proceeding with misinterpreted data.

**Interfaces:**

- *Depends on:* `postprocess/stats.py` (consumes the error-vector point clouds it produces — e.g., calls `pose_error_vector_batch` and slices out translation or rotation columns), `numpy`, `scipy.stats` (chi-squared quantiles for statistical ellipsoid scaling).
- *Used by:* `postprocess/reporting.py` (renders the fitted shapes), eventually `gui/results_viewer/`.
- *Public API (conceptual):*
  ```
  fit_bounding_box(points: np.ndarray) -> dict(min, max)
  fit_bounding_sphere(points: np.ndarray) -> dict(center, radius)
  fit_bounding_ellipsoid(points: np.ndarray, coverage=1.0) -> dict(center, axes_lengths, axes_directions)
  fit_rotation_box(rotvecs: np.ndarray)   -> dict(min, max)        # rotvecs shape (N,3), rows = ω = θu
  fit_rotation_cone(rotvecs: np.ndarray)  -> dict(max_angle, mean_axis)  # rotvecs shape (N,3), rows = ω = θu
  # fit_rotation_box / fit_rotation_cone never accept raw (N,4,4) pose/HTM arrays — locked decision
  ```

---

## 6.10 `postprocess/reporting.py`

*(Last revised: 2026-06-23 — Claude, design-spec update adding Pareto sensitivity chart rendering per Mod 2, cross-review)*

**Responsibility:** Generates Matplotlib plots from the statistics (`postprocess/stats.py`) and fitted shapes (`postprocess/bounding_shapes.py`): per-DoF histograms, 2D projections of bounding shapes, and **as of this revision, the Pareto sensitivity breakdown chart** (Mod 2), per the locked decision to use 2D projections rather than a rotatable 3D viewer (Section 8).

**Deliverables:**

- Per-DoF histogram plotting function
- 2D projection plotting for translation bounding shapes (XY/XZ/YZ)
- 2D plotting for rotation bounding box/cone representations
- Pareto sensitivity bar chart plotting (new, Mod 2) — the graphical counterpart to `ParetoSensitivityReport.to_ascii_chart()` (Section 6.8)
- `tests/test_reporting.py` (smoke tests only — rendering correctness is best verified visually, not asserted numerically)

**Granular Task List:**

1. Implement `plot_histogram(counts, bin_edges, dof_label, ax=None) -> matplotlib.axes.Axes`: single-DoF histogram, returns the Axes so callers (GUI or examples scripts) can embed or further customize it rather than this function owning figure-level layout decisions.
2. Implement `plot_translation_projection(points: np.ndarray, bounding_shape: dict, plane: Literal["xy","xz","yz"], ax=None) -> matplotlib.axes.Axes`: scatter the trial point cloud (or a representative subsample if `N` is large — define a practical subsampling cutoff, e.g., 2000 points, to keep plots legible and fast) projected onto the requested plane, overlay the bounding box/ellipse outline for that projection.
3. Implement `plot_rotation_summary(rotvecs: np.ndarray, cone: dict, box: dict, ax=None) -> matplotlib.axes.Axes`: a combined view with the cone (locked 2026-06-23 as the lead representation, per Section 6.9 Step 3) drawn as the prominent primary element — e.g., a shaded cone/circle at `max_angle` with the `mean_axis` marked — and the per-axis box rendered as a smaller, secondary annotation (e.g., three thin tick marks or a corner inset) rather than as a co-equal visual.
4. **New this revision (Mod 2) — implement `plot_pareto_sensitivity(report: ParetoSensitivityReport, ax=None, top_n=10) -> matplotlib.axes.Axes`:** a horizontal bar chart of the `top_n` ranked edge/DoF contributions (descending), with any remaining contributions grouped into a single "Others" bar — matching the example layout in Mod 2's directive. Must visibly include the first-order-linear-approximation caveat from Section 6.8 Step 8 as an annotation or caption on the chart itself, not just in surrounding prose, since this chart is the one most likely to be screenshotted/shared standalone in a sourcing discussion (Section 1.4) where the caveat could otherwise get lost.
5. Implement a top-level convenience function `generate_frame_report(trial_data, frame_name) -> matplotlib.figure.Figure`: assembles a multi-panel figure (3 translation histograms + 3 rotation histograms + 2–3 translation projections + 1 rotation summary) for a single Frame in one call — this is what `examples/` scripts and eventually the GUI results viewer will call most often.
6. Implement `generate_sensitivity_report(report: ParetoSensitivityReport) -> matplotlib.figure.Figure`: a standalone single-panel figure wrapping Step 4's `plot_pareto_sensitivity()` — kept separate from `generate_frame_report()` since the sensitivity breakdown is queried per point-pair target, not per Frame, and is a distinct enough deliverable (per Section 1.4's "Sensitivity Pinpointing" use case) to warrant its own report-generation entry point.
7. Write `tests/test_reporting.py` as smoke tests: confirm each plotting function runs without raising on representative synthetic data and returns a valid `Axes`/`Figure` object — do not attempt to assert on pixel content; visual correctness is a human-review concern, not an automated-test concern.

**Interfaces:**

- *Depends on:* `postprocess/stats.py` (including the new `ParetoSensitivityReport`, Mod 2), `postprocess/bounding_shapes.py`, `matplotlib`.
- *Used by:* `examples/` scripts directly (Milestone A and B-1), eventually `gui/results_viewer/` (Milestone B-2, likely embedding these same Figures/Axes into Qt widgets via Matplotlib's Qt backend rather than re-implementing plotting logic in the GUI layer).
- *Public API (conceptual):*
  ```
  plot_histogram(counts, bin_edges, dof_label, ax=None) -> Axes
  plot_translation_projection(points, bounding_shape, plane, ax=None) -> Axes
  plot_rotation_summary(rotvecs, cone, box, ax=None) -> Axes
  plot_pareto_sensitivity(report: ParetoSensitivityReport, ax=None, top_n=10) -> Axes   # new, Mod 2
  generate_frame_report(trial_data, frame_name) -> Figure
  generate_sensitivity_report(report: ParetoSensitivityReport) -> Figure                # new, Mod 2
  ```

---

## 6.11 `io/schema.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session)*

**Responsibility:** Pydantic models defining the on-disk project data model — the only interface the GUI is permitted to read from or write to (Section 5.3).

**Deliverables:**

- `FrameModel`, `HTMEdgeModel`, `ToleranceSpecModel`/`ToleranceSpec6Model`, `SimSettingsModel`, `SavedAnalysisModel`, `ProjectModel`
- Bidirectional conversion functions between these Pydantic models and the live `core`/`sim` objects (`FrameGraph`, `ToleranceSpec6`, etc.)
- `tests/test_schema.py` (new test file)

**Granular Task List:**

1. Implement `ToleranceSpecModel` (Pydantic): mirrors `core/tolerance.py`'s `ToleranceSpec` fields with Pydantic-level validation (`bound >= 0`, `distribution` as a `Literal["uniform","normal"]`).
2. Implement `ToleranceSpec6Model`: six `ToleranceSpecModel` fields, named `dx, dy, dz, rx, ry, rz` for on-disk readability (rather than an unlabeled list — a human or future tool opening the raw JSON should be able to read it directly).
3. Implement `HTMInputModel`: a tagged union (Pydantic discriminated union) covering all four input representations (`xyz_euler`, `matrix`, `quaternion`, `screw`) — this is what preserves the "original input representation" metadata from `core/transforms.py`'s `HTM` (Section 6.1) on disk, so reloading a project shows the user's original entry format, not a canonicalized matrix.
4. Implement `HTMEdgeModel`: `name`, `parent`, `child`, `nominal: HTMInputModel`, `tolerance: ToleranceSpec6Model`.
5. Implement `FrameModel`: `name`, optional `metadata: dict`.
6. Implement `SimSettingsModel`: `mode: Literal["fk_verification","ik_allocation"]`, `n_trials: int`, `seed: int`, `default_distribution: Literal["uniform","normal"]`, `default_sigma_level: float`.
7. Implement `SavedAnalysisModel`: persisted point-pair analysis definitions (Section 3.3) — `name`, `frame_a`, `frame_b`, optionally a cached last-run result summary.
8. Implement `ProjectModel`: top-level container — `frames: list[FrameModel]`, `edges: list[HTMEdgeModel]`, `sim_settings: SimSettingsModel`, `saved_analyses: list[SavedAnalysisModel]`.
9. Implement `ProjectModel.validate_references()` (a Pydantic model validator, not just field validators): confirm every edge's `parent`/`child` refers to a declared Frame, and every `SavedAnalysisModel`'s `frame_a`/`frame_b` refers to declared Frames — catch dangling references at schema-validation time, before a `FrameGraph` is ever constructed from this data (Section 6.3's `validate_dag()` catches *topological* errors; this catches *referential* errors one layer earlier).
10. Implement `project_model_to_frame_graph(project: ProjectModel) -> FrameGraph`: constructs a live `core/frame_graph.py` `FrameGraph` from a validated `ProjectModel`, using `core/transforms.py`'s constructors to rebuild each `HTM` from its tagged `HTMInputModel` representation.
11. Implement `frame_graph_to_project_model(frame_graph: FrameGraph, sim_settings, saved_analyses) -> ProjectModel`: the inverse — used when saving.
12. Write `tests/test_schema.py`:
    - Round-trip test: build a `FrameGraph` directly in Python, convert to `ProjectModel`, back to `FrameGraph`, confirm the result is equivalent (same Frames, edges, nominal transforms within tolerance, same tolerance specs).
    - Dangling-reference test: construct a `ProjectModel` with an edge referencing a non-existent Frame, confirm `validate_references()` raises with a specific, actionable message naming the bad reference.
    - Input-representation round-trip: confirm a Frame originally entered via, e.g., screw coordinates is still tagged as `screw` after a save/load round-trip (not silently canonicalized to `matrix`).

**Interfaces:**

- *Depends on:* `core/transforms.py`, `core/tolerance.py`, `core/frame_graph.py` (for the conversion functions in Steps 10–11), `pydantic`.
- *Used by:* `io/serializer.py` (Section 6.12), eventually every GUI panel (Section 5.3 — the GUI reads/writes only these models).
- *Public API (conceptual):*
  ```
  ProjectModel, FrameModel, HTMEdgeModel, ToleranceSpec6Model, SimSettingsModel, SavedAnalysisModel
  ProjectModel.validate_references() -> None  # raises on violation
  project_model_to_frame_graph(project: ProjectModel) -> FrameGraph
  frame_graph_to_project_model(frame_graph, sim_settings, saved_analyses) -> ProjectModel
  ```

---

## 6.12 `io/serializer.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session)*

**Responsibility:** JSON save/load for `io.schema` models, surfacing clear, actionable validation errors before the engine is ever invoked.

**Deliverables:**

- `save_project(project: ProjectModel, path: str) -> None`
- `load_project(path: str) -> ProjectModel`
- Actionable error wrapping around raw Pydantic/JSON errors
- `tests/test_serializer.py` (new test file)

**Granular Task List:**

1. Implement `save_project(project: ProjectModel, path: str) -> None`: serialize via Pydantic's `.model_dump_json(indent=2)` (human-readable, diff-friendly indentation) and write to disk.
2. Implement `load_project(path: str) -> ProjectModel`: read the file, parse via `ProjectModel.model_validate_json(...)`, then explicitly call `.validate_references()` (Section 6.11 Step 9) as a second validation pass.
3. Wrap both raw `pydantic.ValidationError` and basic file errors (missing file, malformed JSON) in a single project-specific exception type (e.g., `ProjectLoadError`) with a clear, human-readable message — do not let a raw Pydantic stack trace be the only thing the user/GUI sees on a malformed file.
4. Add a `schema_version` field to `ProjectModel` (Section 6.11) at this stage, even though there's only one version today — write `load_project` to check it and raise a specific, friendly error if a future version mismatch occurs, rather than letting an old/new file format silently fail with a confusing generic error. (Cheap insurance now; expensive to retrofit later once real project files exist.)
5. Write `tests/test_serializer.py`:
   - Round-trip: save then load a representative `ProjectModel`, confirm equivalence.
   - Malformed JSON: confirm `load_project` raises `ProjectLoadError` with a clear message, not a raw `JSONDecodeError`.
   - Dangling reference in the file: confirm `load_project` surfaces the same actionable message as the direct `validate_references()` test in Section 6.11.
   - Missing file: confirm a clear `ProjectLoadError`, not an unhandled `FileNotFoundError`.

**Interfaces:**

- *Depends on:* `io/schema.py` (`ProjectModel`), `pydantic`, `json`/standard library file I/O.
- *Used by:* eventually `gui/main_window.py` (save/load/new-project actions), and any future CLI.
- *Public API (conceptual):*
  ```
  save_project(project: ProjectModel, path: str) -> None
  load_project(path: str) -> ProjectModel   # raises ProjectLoadError
  ```

---

## 6.13 `gui/main_window.py`

*(Last revised: 2026-06-23 — Claude, detailed planning session. GUI modules are specified at a coarser grain than core/sim/postprocess/io, consistent with Section 10's rule that GUI work does not begin until the engine is proven — these task lists will be revisited and sharpened immediately before Milestone C begins (Section 7.4), once Milestones A, B-1, and B-2 are complete.)*

**Responsibility:** Top-level PySide6 application window; owns the currently-loaded `ProjectModel`, hosts the other GUI panels, and owns save/load/new-project actions.

**Deliverables:**

- Main `QMainWindow` subclass with menu bar (New/Open/Save/Save As/Exit)
- Docked or tabbed hosting of the panels in Sections 6.14–6.18
- A single in-memory `ProjectModel` instance treated as the source of truth for the currently open project

**Granular Task List (to be sharpened before Milestone C):**
1. Implement the `QMainWindow` shell with a menu bar and status bar.
2. Implement New/Open/Save/Save As actions, calling `io/serializer.py` directly (Section 5.3 — GUI talks to `io.schema` only).
3. Implement a simple in-memory "dirty" flag (unsaved changes indicator) tied to edits made in any child panel.
4. Lay out the child panels (Sections 6.14–6.18) as dock widgets or tabs — decide layout once those panels' own specs are sharpened immediately before Milestone C.
5. Wire a top-level error-display mechanism (e.g., a status bar message or modal dialog) for surfacing `ProjectLoadError` and validation errors from `io/serializer.py` / `io/schema.py` in a user-friendly way.

**Interfaces:**

- *Depends on:* `io/schema.py`, `io/serializer.py`, all `gui/*` panel modules, `PySide6`.
- *Used by:* the application entry point (`main.py`, not yet listed in Section 5.1 — add it as the top-level launch script when Milestone C begins).

---

## 6.14 `gui/graph_editor/`

*(Last revised: 2026-06-23 — Claude, detailed planning session. Coarse-grained per Section 6.13's note.)*

**Responsibility:** Build/edit the Frame graph — add/remove Frames and Edges, enter each edge's nominal transform in any supported format.

**Granular Task List (to be sharpened before Milestone C):**
1. A graph/tree view widget listing Frames and Edges (likely `QTreeWidget` or a lightweight embedded NetworkX-to-Qt graph view — decide once this task is reached).
2. An "Add Frame" / "Add Edge" dialog flow, writing directly into the in-memory `ProjectModel` (never into a live `FrameGraph`, per Section 5.3).
3. A multi-format HTM entry widget supporting all four input representations (Section 2.1), with a format-selector control and live validation feedback (e.g., flagging a non-orthonormal raw-matrix entry immediately, reusing `core/transforms.py`'s `HTM` construction-time validation, Section 6.1 Step 3, as the validation backend).
4. Visual indication of which Frame(s) are roots and which are junctions (shared by multiple downstream edges), to make the multi-chain structure legible to the user.

**Interfaces:**

- *Depends on:* `io/schema.py` (reads/writes `ProjectModel` directly), `core/transforms.py` (validation reuse), `PySide6`.
- *Used by:* `gui/main_window.py`.

---

## 6.15 `gui/tolerance_editor/`

*(Last revised: 2026-06-23 — Claude, detailed planning session. Coarse-grained per Section 6.13's note.)*

**Responsibility:** Per-edge, per-DoF tolerance entry: distribution, bound, sigma-level, locked flag.

**Granular Task List (to be sharpened before Milestone C):**
1. A per-edge panel exposing all 6 DoF, each with a distribution selector (`uniform`/`normal`), bound entry, conditionally-shown sigma-level entry (only when `normal` selected), and a locked checkbox.
2. A "bulk apply" convenience action (e.g., "apply this distribution/sigma-level default to all DoF on this edge" or "...to all edges in this project") to avoid tedious repetitive entry — purely a UX nicety, not core to correctness.
3. Live validation reusing `core/tolerance.py`'s `ToleranceSpec` construction-time checks (Section 6.2 Step 1).

**Interfaces:**

- *Depends on:* `io/schema.py`, `core/tolerance.py` (validation reuse), `PySide6`.
- *Used by:* `gui/main_window.py`.

---

## 6.16 `gui/run_panel/`

*(Last revised: 2026-06-23 — Claude, detailed planning session. Coarse-grained per Section 6.13's note.)*

**Responsibility:** Configure and trigger a simulation run: mode selection, trial count, seed, distribution settings.

**Granular Task List (to be sharpened before Milestone C):**
1. Mode selector (FK verification vs. IK allocation), with the panel's visible fields changing based on selection (e.g., IK mode additionally needs a target Frame pair + target tolerance entry, reusing the `gui/tolerance_editor/` widget for the target's bound entry).
2. Trial count and seed entry, with the seed defaulting to a random value but always displayed/editable (so a specific run is always reproducible by recording the seed shown).
3. A "Run" button that constructs live `core`/`sim` objects from the current `ProjectModel` (the one and only place this conversion happens, per Section 5.3) and invokes the appropriate engine (`MonteCarloFKEngine` or `AllocationEngine`), likely on a background thread/`QThread` to avoid freezing the UI during larger runs.
4. Progress indication for longer runs (even a simple indeterminate spinner is acceptable for v1, given the modest trial counts in Section 5.1/13's performance targets).

**Interfaces:**

- *Depends on:* `io/schema.py`, `core/frame_graph.py`, `sim/monte_carlo_fk.py`, `sim/allocation.py`, `PySide6`.
- *Used by:* `gui/main_window.py`; produces the `TrialData`/`ValidationReport` that `gui/results_viewer/` consumes.

---

## 6.17 `gui/results_viewer/`

*(Last revised: 2026-06-23 — Claude, detailed planning session. Coarse-grained per Section 6.13's note.)*

**Responsibility:** Display simulation results — envelope tables, histograms, 2D bounding-shape projections — per Frame or per saved point-pair analysis.

**Granular Task List (to be sharpened before Milestone C):**
1. A Frame/analysis selector driving which result set is currently displayed.
2. An envelope summary table (reusing `postprocess/stats.py`'s `frame_envelope_box`/`point_pair_envelope_box` output directly).
3. Embedded Matplotlib canvases (via `matplotlib.backends.backend_qtagg`) displaying `postprocess/reporting.py`'s `generate_frame_report()` output — reuse the plotting module's Figures directly rather than re-implementing plotting in Qt-native widgets.
4. Display of rotation error per the locked 2026-06-23 decision: the bounding cone (`max_angle`, `mean_axis`) is the headline figure for angular uncertainty, with the per-axis box available as a secondary/expandable detail (e.g., a "show per-axis breakdown" toggle) rather than displayed with equal visual weight.

**Interfaces:**

- *Depends on:* `postprocess/stats.py`, `postprocess/reporting.py`, `PySide6`, `matplotlib`.
- *Used by:* `gui/main_window.py`.

---

## 6.18 `gui/point_pair_panel/`

*(Last revised: 2026-06-23 — Claude, detailed planning session. Coarse-grained per Section 6.13's note.)*

**Responsibility:** Define and persist point-pair analyses — select any two Frames, view their relative-pose envelope, save the analysis definition into the project file.

**Granular Task List (to be sharpened before Milestone C):**
1. A two-Frame selector (dropdowns or graph-click selection, reusing the Frame list from `gui/graph_editor/`'s underlying `ProjectModel`).
2. A "Save Analysis" action writing a new `SavedAnalysisModel` entry into the `ProjectModel` (Section 6.11 Step 7), so it persists across save/load.
3. Display of the resulting relative-pose envelope, reusing `gui/results_viewer/`'s display components rather than duplicating presentation logic.
4. A same-component validation check at selection time (reusing `core/frame_graph.py`'s `weakly_connected_components()`), giving immediate UI feedback (e.g., greying out an invalid second-Frame selection) rather than only erroring after a run.

**Interfaces:**

- *Depends on:* `io/schema.py`, `core/frame_graph.py`, `postprocess/stats.py`, `gui/results_viewer/` (shared display components), `PySide6`.
- *Used by:* `gui/main_window.py`.

---

## 6.19 `examples/`

*(Last revised: 2026-06-25 — Claude, A7 implementation, commits `3d7936d`, `8232c12`, `178fae0`)*

**Responsibility:** Hand-verified, fully worked example scripts demonstrating end-to-end engine usage — required Milestone A deliverable (Task A7), doubling as onboarding material and a regression-test reference.

**Deliverables:**

- At least one fully worked, commented script using a real or representative precision-machine-design problem
- At least one script demonstrating the multi-chain/shared-frame (optical-mount-style) use case explicitly, since this is the most architecturally distinctive capability and deserves a dedicated, legible demonstration separate from the unit tests

**Granular Task List:**

1. ✅ **Done (A7, `3d7936d`)** Write `examples/single_chain_fk_example.py`: 3-edge CNC spindle alignment chain (`world → spindle_housing → bearing_seat → tool_tip`), mixing `uniform` and `normal` tolerances, 50,000-trial MC run. Prints worst-case envelope box and 50/90/95/99th-percentile table for `tool_tip`. Text output only — `postprocess/reporting.py` (B1-4) deliberately not imported. Demonstrates angular error amplification: a 0.001 rad housing-tilt creates ~0.150 mm tip displacement over 150 mm tool overhang.
2. ✅ **Done (A7, `3d7936d` initial; `8232c12` extended)** Write `examples/multi_chain_shared_frame_example.py`: optical bench with two lenses sharing a common upstream frame. Extended beyond the original single-scenario plan to a **four-scenario sweep** that decomposes the shared-ancestor cancellation effect: (0) zero lens tolerances — proves bench errors cancel to floating-point zero (1.99e-13 mm); (1) translational seat errors only — shows direct RSS contribution; (2) rotational seat errors only — reveals lever-arm amplification (rz ±0.001 rad × 100 mm separation = ±0.1 mm relative dy, matching the analytical prediction exactly); (3) full tolerances — combination, dominated by rotational lever arm. Includes a summary comparison table and `L × δθ` design rule. Uses `frame_envelope_box()`, `point_pair_envelope_box()` only — no B-1 dependencies.
3. Write `examples/allocation_example.py` (added once Milestone B-2's `sim/allocation.py` exists): define a chain with unset/free tolerances, specify a target end-effector envelope, run `AllocationEngine.allocate()`, print the resulting `AllocationResult` — including the baseline linear allocation, the corrected allocation, and convergence status — with inline commentary on what the gap between baseline and corrected reveals about the chain's geometric leverage.
4. ✅ Each script runs standalone (`python examples/<script>.py`) with no GUI dependency, prints enough intermediate information that a reader can follow the logic without running it. Verified for tasks 1 and 2 above.

**Interfaces:**

- *Depends on:* the full `core`/`sim`/`postprocess` stack (this is integration-level usage, not a module with its own internal logic).
- *Used by:* the project owner directly (onboarding/reference), and indirectly by `tests/` as a source of representative scenarios worth turning into regression tests.

---

## 6.20 `tests/`

*(Last revised: 2026-06-24 — Claude, A6 implementation, commit `83b8ee3`. Global tolerance convention established: `DEFAULT_ATOL=1e-9` (exact composition checks), `SMALL_ANGLE_ATOL=1e-6` (checks where trig residuals at ~1 mrad apply). Three shared fixtures in `conftest.py`: `two_edge_chain`, `three_edge_chain`, `shared_frame_graph`. Integration tests cover: two-edge translation stack-up (exact), rotation→translation cross-coupling/lever-arm (hand-verified small-angle derivation), local-frame perturbation routing through nominal rotation. `test_shared_edge_sampling_consistency` written as required module-level function per Section 9 Item 3. `test_allocation_mc_validation_discrepancy` placeholder added to `test_allocation.py` per Section 9 Item 4 (`pytest.mark.skip` until B-2). README.md has CI entry point. 130 passed, 1 skipped.)*

**Responsibility:** The full unit/integration test suite. Houses every hand-calculable validation case described per-module above, plus the dedicated cross-cutting regression tests called out in Section 9.

**Deliverables (test files, consolidated from the per-module task lists above — listed here as the authoritative index):**

**✅ Implemented (Milestone A):**
- `conftest.py` — ✅ A6 (`83b8ee3`): `DEFAULT_ATOL=1e-9`, `SMALL_ANGLE_ATOL=1e-6`, fixtures `two_edge_chain` / `three_edge_chain` / `shared_frame_graph`, helpers `make_tol` / `make_zero_tol` / `make_htm`
- `test_transforms.py` — ✅ A1 (`a31218e`): 26 tests covering `HTM` and `core/conversions.py`
- `test_tolerance.py` — ✅ A2 (`3ac0eed`): 21 tests covering `ToleranceSpec6`, `apply_perturbation_batch`, `core/sampling.py`
- `test_frame_graph.py` — ✅ A3 (`d81645c`): 24 tests covering `FrameGraph`, `adjoint()`, `compute_sensitivity()`
- `test_monte_carlo_fk.py` — ✅ A4 (`744c562`): 18 tests covering `MonteCarloFKEngine`, `TrialData`, per-edge RNG
- `test_stats.py` — ✅ A5 (`019eb34`): 26 tests covering Steps 1–7 of Section 6.8 (B1-3 Pareto tests to be added in Milestone B-1)
- `test_integration.py` — ✅ A6 (`83b8ee3`): 15 end-to-end FK → stats tests; includes `test_shared_edge_sampling_consistency` (Section 9 Item 3 standalone named regression)
- `test_allocation.py` — ✅ A6 placeholder (`83b8ee3`): `test_allocation_mc_validation_discrepancy` named and skipped per Section 9 Item 4 (to be implemented in B-2)

**Pending (Milestone B-1):**
- `test_bounding_shapes.py` (Section 6.9 — add when `postprocess/bounding_shapes.py` is implemented)
- `test_reporting.py` (Section 6.10 — smoke tests only, add when `postprocess/reporting.py` is implemented)
- `test_schema.py` (Section 6.11 — add when `io/schema.py` is implemented)
- `test_serializer.py` (Section 6.12 — add when `io/serializer.py` is implemented)

**Granular Task List (cross-cutting, beyond what's already specified per-module above):**
1. ✅ **Done (A6, `83b8ee3`)** Set up `pytest` configuration with a shared `conftest.py` providing reusable fixtures: `two_edge_chain` (root→B→C, 5 mm + 10 mm translation nominals), `three_edge_chain` (Rz=π/4 + 50 mm + zero-tol), `shared_frame_graph` (shared-base multi-branch). No `pytest.ini` was needed — auto-discovery works from the project root. Module-level helpers `make_tol` / `make_zero_tol` / `make_htm` duplicated inline in `test_integration.py` rather than imported from conftest (conftest.py is not directly importable as a Python module in pytest's default discovery mode without additional path config).
2. ✅ **Done (A6, `83b8ee3`)** `DEFAULT_ATOL = 1e-9` and `SMALL_ANGLE_ATOL = 1e-6` defined in `conftest.py` and mirrored in `test_integration.py`. Convention: `DEFAULT_ATOL` for near-exact floating-point composition (no trig residual); `SMALL_ANGLE_ATOL` for checks where `sin(δθ) vs δθ` at ~1 mrad introduces ~1.7e-10 second-order residual.
3. ✅ **Done (A6, `83b8ee3`)** `test_shared_edge_sampling_consistency` implemented as a module-level function in `tests/test_integration.py` (not inside any class), findable by `pytest -k test_shared_edge_sampling_consistency`. Complementary class-based coverage exists in `test_monte_carlo_fk.py::TestSharedEdgeConsistency`, but the required standalone named function is in `test_integration.py` per this spec requirement.
4. ✅ **Done (A6, `83b8ee3` — placeholder)** `test_allocation_mc_validation_discrepancy` in `tests/test_allocation.py` as a `@pytest.mark.skip(reason="sim/allocation.py not yet implemented — Milestone B-2 task")` module-level function. Real implementation pending B-2.
5. ✅ **Done (A6, `83b8ee3`)** `README.md` "Running Tests" section: `source .venv/bin/activate && python -m pytest tests/ -q`. Notes that all tests must pass before GUI work begins (Section 10 rule).

**Interfaces:**

- *Depends on:* every module in the codebase, by design.
- *Used by:* the project owner and any future Claude Code session, as the primary mechanism for confirming nothing has silently broken between sessions.

---

# 7. Project Phases & Time Allocation

**Total estimated effort: ~165–230 hours**, split across four phases (revised this session, per Mod-2/Mod-3 cross-review — the increase from the original 130–170 hour estimate reflects genuinely new scope added this revision: the Pareto sensitivity engine, the explicit Physical Validation Test Suite, and a dedicated reporting-module task, not re-estimation of unchanged work). Hours assume AI-assisted ("vibe-coded") development with the project owner's light supervision and an ME/robotics background sufficient to validate the math by hand-checking simple cases.

**Revised milestone order (locked this session):** Milestone A is unchanged. What was previously a single "Milestone B" is now split into **B-1 (Forward Analysis & Sensitivities)** and **B-2 (Inverse Damped Allocation)**, with **B-2 explicitly gated on B-1 passing all three Section 9.1 physical validation benchmarks** — the riskier, more complex inverse-allocation work is not started until the forward engine and its sensitivity analysis have been proven against concrete physical anchors, not just internal hand-calculated test cases. GUI work is broken out into its own later phase, **Milestone C**, consistent with the existing rule (Section 10) that GUI work never starts before the engine is proven.

## 7.1 Milestone A — V0.5 ("Proof of Concept / Minimum Viable Tool")

**Goal:** A working, hand-verified, script/CLI-level tool (no GUI) that performs forward Monte Carlo tolerance verification on a serial chain, with worst-case and statistical modes, and point-pair post-processing. Usable immediately for the project owner's own real ME work.

**Target: ~40–45 hours, achievable in 4–6 weeks at 8–12 hrs/week.**

| # | Status | Task | Module(s) | Est. Hours | Commit |
|---|---|---|---|---|---|
| A1 | ✅ Done | `HTM` class: canonical 4x4 storage, constructors/converters wrapping pytransform3d, `compose`/`inverse` | `core/transforms.py`, `core/conversions.py` | 4–6 | `a31218e` |
| A2 | ✅ Done | `ToleranceSpec` / `ToleranceSpec6`: per-DoF uniform + normal sampling via `scipy.stats`, sigma-level handling, local-frame right-multiply perturbation composition (Section 2.2.2) | `core/tolerance.py`, `core/sampling.py` | 5–8 | `3ac0eed` |
| A3 | ✅ Done | `Frame` / `HTMEdge` / `FrameGraph`: NetworkX-backed graph, DAG validation with clear error messages, path-finding between arbitrary frames; also includes `adjoint()` / `compute_sensitivity()` relocated from B1-1 | `core/frame_graph.py` | 4–6 | `d81645c` |
| A4 | ✅ Done | Monte Carlo FK engine: vectorized batched sampling and chain composition, per-Frame trial-data storage, per-edge RNG sub-stream derivation | `sim/monte_carlo_fk.py` | 8–10 | `744c562` |
| A5 | ✅ Done | Post-processing stats: envelope (min/max box), percentile tables, basic histogram data, point-pair relative-transform stats from stored trial data | `postprocess/stats.py` | 6–8 | `019eb34` |
| A6 | ✅ Done | Hand-verified test cases: 2–3 simple chains (e.g., a 2-edge and a 3-edge chain) checked against manual small-angle calculations; shared-edge consistency regression test; `conftest.py` shared fixtures | `tests/` | 6–8 | `83b8ee3` |
| A7 | ✅ Done | Two standalone example scripts: `single_chain_fk_example.py` (3-edge CNC spindle, mixed tolerances, envelope + percentile output) and `multi_chain_shared_frame_example.py` (optical bench, 4-scenario sweep proving shared-ancestor cancellation and lever-arm amplification) | `examples/` | 3–5 | `3d7936d`, `8232c12` |

**Milestone A exit criteria:** Project owner can define a real kinematic chain from their own work in a Python script, run both worst-case and statistical Monte Carlo FK verification, and trust the printed/plotted envelope output because it has been checked against at least one hand calculation.

**✅ Milestone A is complete.** All 7 tasks done and pushed to `origin/main`. 130 tests passing, 1 skipped. Next: Milestone B-1 (Section 7.2).

## 7.2 Milestone B-1 — Forward Analysis & Sensitivities

**Goal:** Adds bounding-shape outputs, the Pareto sensitivity engine, plotting/reporting, and file save/load on top of the proven V0.5 forward engine — and, critically, **proves the forward engine and sensitivity analysis against the three concrete physical benchmarks in Section 9.1** before any inverse-allocation work begins. No rework of Milestone A code, only additive layers.

**Target: ~49–70 hours.**

| # | Task | Module(s) | Est. Hours |
|---|---|---|---|
| B1-1 | ✅ Done (built in A3, commit `d81645c`) — `adjoint()`, `compute_sensitivity()`, and `path_edges_between()` implemented and cross-validated with finite-difference tests; formulas corrected (see changelog 2026-06-24) | `core/frame_graph.py` | 6–9 |
| B1-2 | Bounding shape fitting: bounding box/ellipsoid/sphere (translation), bounding cone (rotation, locked as lead representation), rotation-vector type-hardening | `postprocess/bounding_shapes.py` | 8–12 |
| B1-3 | Pareto sensitivity engine: `compute_tolerance_sensitivities()`, `ParetoSensitivityReport`, variance-contribution math (uniform/normal consistency) — new this revision, Mod 2 | `postprocess/stats.py` | 8–12 |
| B1-4 | Plotting/reporting: histograms, 2D bounding-shape projections, Pareto sensitivity bar chart with embedded linear-approximation caveat | `postprocess/reporting.py` | 8–10 |
| B1-5 | Pydantic schema + JSON save/load, validation-on-load with actionable errors | `io/schema.py`, `io/serializer.py` | 8–10 |
| B1-6 | **Physical Validation Test Suite implementation (Section 9.1) — gating deliverable, not optional:** the Linear Stack-Up (RSS) Benchmark, the Sine-Bar Lever Arm Benchmark, and the Common-Ancestor Cancellation Benchmark, each as a dedicated, named regression test | `tests/` | 8–12 |
| B1-7 | Example scripts demonstrating the new forward-analysis capabilities (sensitivity Pareto breakdown, component-selection and mitigation-verification use cases from Section 1.4) | `examples/` | 3–5 |

**Milestone B-1 exit criteria:** All three Section 9.1 physical validation benchmarks pass. The Pareto sensitivity breakdown produces correct, hand-verifiable rankings on a representative chain. Bounding shapes, plots, and project save/load all function end-to-end via script/CLI usage (still no GUI). **Milestone B-2 does not begin until this exit criteria is met.**

## 7.3 Milestone B-2 — Inverse Damped Allocation

**Goal:** The inverse tolerance allocation engine, including the iterative nonlinear damping/correction loop (Section 6.7). Explicitly gated on Milestone B-1's physical validation passing — this is intentionally the last and riskiest piece of core engine work, built only once the forward engine and its sensitivity machinery are proven.

**Target: ~24–37 hours.**

| # | Task | Module(s) | Est. Hours |
|---|---|---|---|
| B2-1 | `AllocationObjective` interface + `EqualAllocation` implementation + `AllocationEngine.solve()` — built on top of B1-1's already-validated sensitivity primitive, no Jacobian math re-implemented here | `sim/allocation.py` | 10–15 |
| B2-2 | Iterative nonlinear damping/correction loop: `AllocationEngine.allocate()`, `AllocationResult`, convergence/non-convergence handling per the locked `gamma`/`max_iter`/status-message contract (Section 6.7, Step 5) | `sim/allocation.py` | 8–12 |
| B2-3 | Test coverage: equal-allocation sanity check, locked-edge and all-locked-edge cases, MC validation discrepancy reporting, damping loop convergence and non-convergence tests | `tests/` | 6–10 |

**Milestone B-2 exit criteria:** `AllocationEngine.allocate()` produces a correctly-converging allocation on the Sine-Bar Lever Arm Benchmark's geometry (the known high-leverage case from Section 9.1.2), correctly reports non-convergence on a deliberately infeasible target, and the baseline-vs-corrected diagnostic is verified to reflect real geometric leverage on at least one representative chain.

## 7.4 Milestone C — GUI

**Goal:** The full PySide6 desktop application on top of the proven engine (Milestones A, B-1, B-2). No engine logic is implemented here — this phase only builds the GUI layer described in Sections 6.13–6.18, talking exclusively to the `io.schema` data model (Section 5.3).

**Target: ~55–77 hours.**

| # | Task | Module(s) | Est. Hours |
|---|---|---|---|
| C1 | GUI: graph/chain editor (add/edit Frames & Edges, multi-format HTM entry) | `gui/graph_editor/` | 10–14 |
| C2 | GUI: tolerance editor (per-DoF distribution, bound, sigma-level, lock toggle) | `gui/tolerance_editor/` | 6–8 |
| C3 | GUI: run panel (mode select, N trials, seed, distribution, run trigger; IK mode calls `allocate()` by default) | `gui/run_panel/` | 4–6 |
| C4 | GUI: results viewer (envelope tables, histograms, 2D projections of bounding shapes, Pareto sensitivity chart, via Matplotlib) | `gui/results_viewer/` | 10–14 |
| C5 | GUI: point-pair analysis panel + saved-analysis persistence in project file | `gui/point_pair_panel/` | 5–8 |
| C6 | Additional/expanded test coverage at the GUI integration level | `tests/` | 8–12 |
| C7 | Integration, bug fixing, end-to-end polish pass | — | 12–15 |

**Milestone C exit criteria:** A double-clickable (or `python main.py`-launchable) desktop application where a user with no Python experience could define a system, set tolerances, run both modes, and extract bounding-shape and sensitivity decisions — backed by the same validated engine from Milestones A, B-1, and B-2.

## 7.5 Deferred / Explicitly Out of Scope for V1.0

Recorded here so they are not forgotten, but also not accidentally started early:

- Convex-hull (non-axis-aligned) bounding shapes for translation point clouds (v1.x nice-to-have)
- Fast analytical (non-Monte-Carlo) linear worst-case estimate as a quick sanity-check shortcut (v1.x nice-to-have)
- Allocation objectives beyond `EqualAllocation` (e.g., cost-weighted allocation)
- True 3D rotatable bounding-shape viewer (v1.0 uses 2D projections only, by explicit decision — simpler, more reliable)
- Correlated tolerances between DoF or between edges (explicitly excluded by design, Section 1.2 — do not add without a full re-discussion, as it changes the statistical independence assumptions throughout the sampling and post-processing layers)
- Parallel/closed-loop kinematic chains (explicitly excluded by design, Section 1.2)

---

# 8. Open Parameters Locked During Planning (Reference)

These were explicit decisions made during project scoping and should be treated as fixed defaults, overridable by the user only where noted:

| Parameter | Decision |
|---|---|
| Default tolerance distribution | `uniform` (hard bound / worst-case) |
| Default sigma level for `normal` distribution | 3.0 (i.e., stated tolerance = 3σ), user-overridable per-edge or globally |
| Perturbation application convention | Local-frame, right-multiplication (`T_nominal @ T_delta`) |
| Rotation tolerance model | Small-angle approximation throughout — no large-angle support |
| DoF/edge correlation | None — all 6 DoF per edge, and all edges, treated as statistically independent |
| FK chain topology | Strictly serial / open / DAG with max one incoming edge per Frame — no closed loops, no parallel mechanisms |
| Inverse allocation objective (v1) | Equal allocation across free/unlocked edges only |
| Rotation bounding shape (v1) | **Cone** (`max_angle` + `mean_axis`) is the confirmed lead representation, locked 2026-06-23. Per-axis box still implemented as a secondary/expandable cross-check (Section 6.9), not displayed with equal prominence. |
| Bounding shape visualization | 2D projections (not a true rotatable 3D viewer) |
| GUI framework | PySide6, desktop, local-only, no server/browser split |
| Project file format | JSON via Pydantic schema |
| TrialData pose storage | Full 4x4 HTM per Frame per trial (not a reduced 6-vector) — locked 2026-06-23 |
| Euler angle convention | Intrinsic ZYX — locked 2026-06-23 |
| IK Jacobian/sensitivity method | Analytical/closed-form via small-angle adjoint transformation; finite-difference used only as a test-suite correctness oracle, never in production code — locked 2026-06-23 |
| Monte Carlo RNG strategy | Per-edge keyed sub-streams (`SeedSequence` spawned from a deterministic hash of edge name), owned by `sim/monte_carlo_fk.py` — guarantees one edge's samples are unaffected by changes elsewhere in the graph — locked 2026-06-23 |
| `core/sampling.py` location | Relocated from originally-planned `sim/sampling.py` to `core/sampling.py` to fix a backwards dependency (`core/tolerance.py` needs it) — locked 2026-06-23 |
| Inverse allocation correction strategy | `AllocationEngine.allocate()` always applies an iterative nonlinear damping loop on top of the closed-form linear allocation: `N_validate=1000`, damping factor `gamma` default range `[0.7, 0.95]`, `max_iter=10`, damping applied to free *angular* DoF bounds only. Non-convergence returns `converged=False` with status message exactly `"Allocation could not converge to target budget"` (no exception raised) — locked 2026-06-23 |
| Disjoint frame-graph components | The engine **never** auto-connects disjoint sub-graphs or fabricates a global origin frame. `nominal_transform_between()` raises `DisjointFramesError` with a locked, exact message recommending the user explicitly model a shared "Common Physical Base" Frame (Section 2.3.1) — locked 2026-06-23 |
| Rotation-vector interface contract | `postprocess/bounding_shapes.py`'s `fit_rotation_cone()`/`fit_rotation_box()` accept only `(N,3)` rotation-vector arrays (convention named **ω = θu**); raw `(N,4,4)` pose arrays are explicitly rejected at runtime to prevent coordinate-coupling shortcuts — locked 2026-06-23 |

---

# 9. Testing & Validation Philosophy

Because this tool produces engineering decisions about physical hardware tolerances, **correctness validation is non-negotiable and is budgeted explicitly as real project hours, not an afterthought.** The approach:

1. **Every new core math module ships with at least one hand-calculable test case** before being considered done — e.g., a single toleranced HTM perturbed by a known delta, checked against manually computed small-angle math.
2. **Chain-level test cases** (2–3 edges) are checked by composing the hand calculations from step 1 manually and comparing to engine output, for both worst-case and statistical modes.
3. **Point-pair and shared-frame test cases** explicitly verify that two downstream Frames sharing a common upstream edge see *identical* sampled perturbation for that edge within a given trial (Section 2.4) — this is a correctness-critical property, not just a nice-to-have, and deserves a dedicated regression test.
4. **Allocation engine validation** always includes the MC-validation-pass discrepancy check (Section 3.2, step 3) as part of its own test suite, not just as a runtime feature.

## 9.1 Physical Validation Test Suite

*(Added 2026-06-23 — Project Owner, per cross-review, Mod 3. These three benchmarks are named, concrete physical anchors for the test suite — each one checks the engine's output against a classical, independently-derivable analytical result, rather than only against the engine's own internal hand-calculated cases from Section 9 above. Per the revised milestone order (Section 7), Milestone B-1 is not considered complete until all three pass.)*

### 9.1.1 The Linear Stack-Up Benchmark

**Setup:** A 5-link purely serial translation chain (no rotation, or zero nominal rotation between links — pure translational stack-up).

**Verification:** Run forward Monte Carlo (Section 6.6) with `"normal"`-distribution tolerances on each link's translational DoF. Confirm the resulting output variance at the chain's end **exactly matches the classical analytical Root-Sum-Square (RSS) calculation** — i.e., `σ_total² = Σ σ_i²` for independent per-link variances, which holds exactly (not just approximately) for a purely linear, uncoupled translation stack-up. This is the simplest possible case where the engine's Monte Carlo output has a closed-form analytical answer, and it must match to within Monte Carlo sampling error (quantify the expected sampling error for the chosen `n_trials` and assert within that bound, not an arbitrarily loose tolerance).

### 9.1.2 The Sine-Bar Lever Arm Benchmark

**Setup:** A single angular pivot (one edge with a nonzero rotational tolerance, zero translational tolerance) followed by a fixed translational offset vector of length `L` (a second edge, zero tolerance, pure translation along some axis).

**Verification:** For small angles (**≤ 1 mrad**, comfortably within the small-angle regime this entire tool is built on, per Section 1.2), confirm the lateral error generated in the forward kinematics pass **perfectly matches the analytical cross-coupling relationship Δx ≈ L·θ**. This is the direct physical validation of the exact geometric-leverage effect that motivated the inverse allocation engine's damping loop (Section 6.7) — if this benchmark doesn't hold, the entire justification for that damping loop is unfounded, so this test is load-bearing for more than just the FK engine.

### 9.1.3 The Common-Ancestor Cancellation Benchmark

**Setup:** A split-tree layout: an inspection camera Frame and a target sample Frame, each independently positioned, but both branching from a shared structural Frame representing a granite base — which itself sits beneath a large, loosely-toleranced edge representing the granite base's attachment to the room/building floor. **Precise mapping onto the model (stated explicitly to remove any ambiguity):** the room/building floor is the graph's root Frame; the granite base is a child Frame connected to the root via one edge carrying a large, deliberately loose tolerance (representing real-world floor-to-base mounting uncertainty); the camera and sample Frames are each children of the granite base via their own separate edges/chains.

**Verification:** Run a relative-transform point-pair query (`point_pair_envelope_box`, Section 6.8) between the camera Frame and the sample Frame. Confirm that the large absolute structural tolerance on the floor-to-granite-base edge **completely cancels out of the relative measurement** — the camera-to-sample envelope should reflect only the camera's and sample's own chain tolerances, with no contribution from the shared floor-to-base edge, regardless of how loose that shared edge's tolerance is. This is the formal validation of Section 2.4's "Monte Carlo Consistency Across Shared Edges" design property, given a sharper, physically-motivated framing — and it is the same underlying mechanism that powers the "Mitigation Verification" use case named in Section 1.4 (grounding a metrology rail to a shared base plate to achieve cancellation at the sensor plane).

---

# 10. Working Process Notes (For Future Claude Sessions)

- **This document is authoritative.** If a future session (with no memory of this conversation) is asked to continue this project, it should read this document in full before writing any code, and treat Sections 2 and 4 (conventions and library decisions) as fixed unless the project owner explicitly says otherwise.
- **Milestone A (V0.5) should be completed in full, with passing hand-verified tests, before any GUI work begins.** Do not let GUI work creep into early sessions even if it seems "quick to add" — the engine must be proven first.
- **When in doubt about a small ambiguous implementation detail not covered here, make the simplest reasonable choice, implement it, and add a one-line note to Section 11 (Changelog) describing the choice made** so it can be revisited later if needed, rather than blocking progress on a clarifying question.
- **Do not add scope from Section 7.3 (deferred items) without an explicit new discussion with the project owner.**

---

# 11. Changelog

**Every entry in this table must include the name of the person (or "Claude," if an AI session made an unprompted minor implementation choice per Section 10) responsible for the change.** Do not log changes anonymously — attribution is required so future sessions know who to ask if a decision needs revisiting.

| Date | Name | Change |
|---|---|---|
| 2026-06-22 | [Project Owner] | Initial project plan created. Architecture, conventions, library choices, and two-milestone phased plan (V0.5 / V1.0) established per project scoping discussion. |
| 2026-06-22 | [Project Owner] | Converted document to .docx for easier future editing. Reframed document as a combined design specification + project plan (not just a plan). Added required Name column to this changelog. Added Section 6 (Module Specifications) with a dedicated subsection for every code module in the architecture, to be expanded with detailed specs over time. |
| 2026-06-23 | Claude (detailed planning session, at Project Owner's request) | Expanded every module subsection in Section 6 from a brief description into a full granular spec: explicit deliverables, ordered task lists, and inter-module interfaces (depends-on / used-by / public API). Locked four previously-open implementation decisions: (1) `TrialData` stores full 4x4 poses per Frame per trial; (2) Euler convention fixed as intrinsic ZYX; (3) IK Jacobian computed analytically via small-angle adjoint transformation, with finite-difference reserved for test-suite validation only; (4) Monte Carlo RNG uses per-edge keyed sub-streams owned by the FK engine. Corrected an architectural error: relocated `sampling.py` from `sim/` to `core/` to fix a backwards dependency. Updated the Section 5.1 directory tree, the Milestone A task table, and the Section 8 reference table accordingly. Rotation bounding-shape representation (per-axis box vs. cone) remains explicitly open — both will be implemented so the project owner can compare directly. |
| 2026-06-23 | Project Owner | Confirmed `locked` semantics in `core/tolerance.py` (Section 6.2): a locked DoF is still sampled in FK mode (it has a real, fixed physical tolerance); `locked` only excludes it from the free-variable set in inverse allocation (Section 6.7). No change to the document, decision confirmed as already specified. |
| 2026-06-23 | Project Owner | Resolved the previously-open rotation bounding-shape decision: **the cone (`max_angle` + `mean_axis`) is the confirmed lead representation** for angular error, with the per-axis box retained as a secondary/expandable cross-check rather than a co-equal display. Updated Section 6.9 (`postprocess/bounding_shapes.py`), Section 6.10 (`postprocess/reporting.py`), Section 6.17 (`gui/results_viewer/`), and the Section 8 reference table accordingly. |
| 2026-06-23 | Project Owner (cross-checked by an independent reviewing agent; resolved by Claude) | Resolved three issues raised in an independent architectural review: (1) **`sim/allocation.py`** — added a top-level `AllocationEngine.allocate()` method wrapping the closed-form linear allocation in an iterative nonlinear damping/correction loop, addressing geometric cross-coupling (`dx ≈ L·θ`) in high-leverage serial chains: runs an `N_validate=1000`-trial MC check, applies a uniform damping factor (`gamma`, default range `[0.7, 0.95]`) to free edges' *angular* DoF bounds only when validation fails, capped at `max_iter=10`, returning a new `AllocationResult` that explicitly preserves **both** the uncorrected baseline linear allocation and the final corrected allocation (the gap between them is itself a diagnostic of chain leverage), with the exact non-convergence status message `"Allocation could not converge to target budget"` locked verbatim. (2) **`core/frame_graph.py` / Section 2.3** — explicitly rejected an auto-generated global-origin/silent-auto-connection proposal as unsafe for an error-budgeting tool (it would silently fabricate zero-tolerance rigid connections the user never specified); instead added new Section 2.3.1 documenting the "Common Physical Base" explicit-junction-Frame modeling pattern with a worked example, and locked the exact `DisjointFramesError` message text raised by `nominal_transform_between()` when two Frames share no path. (3) **`postprocess/bounding_shapes.py`** — hardened the `fit_rotation_cone()`/`fit_rotation_box()` interface to accept only pre-extracted `(N,3)` small-angle rotation-vector arrays (convention named explicitly: **ω = θu**), with a runtime shape check rejecting raw `(N,4,4)` pose arrays, preventing future shortcut implementations from reintroducing multi-axis coordinate-coupling artifacts. Updated Sections 2.3, 6.3, 6.7, 6.8, 6.9, and the corresponding test task lists accordingly. No code was written in this revision — design-specification update only, per explicit instruction. |
| 2026-06-23 | Project Owner (cross-checked by a second independent reviewing agent, end-user perspective; resolved by Claude) | Integrated three further conceptual additions (Mods 1–3): **(Mod 1)** added new Section 1.4 ("Core Engineering Decisions Enabled") naming the three concrete categories of engineering decision the tool exists to support — component selection/sourcing, sensitivity pinpointing, and mitigation verification — each tied directly to a specific tool capability. **(Mod 2)** added a standalone Pareto-sorted tolerance-sensitivity breakdown, `compute_tolerance_sensitivities()`, to `postprocess/stats.py` (Section 6.8), elevating sensitivity analysis out of the inverse allocator into its own forward-analysis deliverable; as a structural refinement, relocated the shared `adjoint()`/`compute_sensitivity()` primitive to `core/frame_graph.py` (Section 6.3) so `postprocess/stats.py` and `sim/allocation.py` consume one implementation rather than risking two independent copies of the same math drifting apart; documented the first-order-linear-approximation caveat on the Pareto output explicitly, including as an on-chart annotation in `postprocess/reporting.py` (Section 6.10), so the breakdown can't be mistaken for an exact nonlinear attribution. **(Mod 3)** added new Section 9.1 ("Physical Validation Test Suite") with three named, physically-grounded benchmarks — the Linear Stack-Up (RSS) Benchmark, the Sine-Bar Lever Arm Benchmark (`Δx ≈ L·θ`), and the Common-Ancestor Cancellation Benchmark (with an explicit, unambiguous mapping of "granite base" / "room floor" onto Frame-graph terms) — replacing generic "hand-calculated case" placeholders with concrete physical anchors. **Restructured Section 7's milestone plan accordingly:** Milestone A unchanged; the former single "Milestone B" split into **B-1 (Forward Analysis & Sensitivities** — bounding shapes, Pareto sensitivity engine, reporting, IO/schema, gated by Section 9.1 passing) and **B-2 (Inverse Damped Allocation** — explicitly gated on B-1's exit criteria); GUI work broken out into its own **Milestone C**. Updated the total estimated-effort range to ~165–230 hours, reflecting genuinely new scope rather than re-estimation. No code was written in this revision — design-specification update only, per explicit instruction. |
| 2026-06-24 | Claude (A1 implementation) | Implemented `core/transforms.py` (HTM class, 4 constructors, `compose`, `inverse`, `is_close`, `__repr__`) and `core/conversions.py` (pytransform3d 3.15.0 wrapper; updated to non-deprecated API: `matrix_from_euler(e, i=2, j=1, k=0, extrinsic=False)` / `euler_from_matrix(R, i=2, j=1, k=0, extrinsic=False)`). 26 tests written and passing in `tests/test_transforms.py`. SVD re-orthonormalization used in small-angle rotation construction. Commit `a31218e`. |
| 2026-06-24 | Claude (A2 implementation) | Implemented `core/tolerance.py` (`ToleranceSpec`, `ToleranceSpec6`, `skew`, `small_angle_rotation_matrix_batch`, `delta_to_htm_batch`, `apply_perturbation_batch`) and `core/sampling.py` (`sample_uniform`, `sample_normal`, `sample`). Key implementation decision documented in code: `ToleranceSpec.sample()` always samples regardless of `locked=True`; the locked flag is exclusively for the allocation engine's free-variable selection. 21 tests written and passing in `tests/test_tolerance.py`. One test regression resolved: SVD orthonormalization correctly yields `cos(rx)` on the diagonal (not `1.0`), so the test checks against the exact `Rx(rx)` matrix rather than the first-order approximation. Commit `3ac0eed`. |
| 2026-06-24 | Claude (A3 implementation — two math corrections required) | Implemented `core/frame_graph.py` (`DisjointFramesError`, `Frame`, `HTMEdge`, `FrameGraph`, `adjoint`, `compute_sensitivity`). 24 tests written and passing in `tests/test_frame_graph.py`. **Two mathematical errors in this spec were discovered and corrected via finite-difference cross-check during implementation:** (1) **Adjoint formula corrected:** the spec draft specified `[[R, 0],[skew(t)@R, R]]` (the `[ω,v]` convention), but the correct formula for our `[v,ω]` delta ordering is `[[R, skew(t)@R],[0, R]]` — top-right block is `skew(t)@R`, bottom-left is zero. (2) **Sensitivity formula corrected:** the spec draft specified `J_i = Ad_{T_{exit_node → frame_b}}` (adjoint from exit node to frame_b), but the correct formula derived from `T_perturbed @ T_nominal⁻¹ = T_{frame_a→exit_i} @ T_delta @ T_{frame_a→exit_i}⁻¹` is `J_i = Ad_{T_{frame_a → exit_node_i}}`. Both corrections are reflected in the Section 6.3 task list above and confirmed by passing finite-difference cross-check tests. Also: `path_edges_between()` returns `list[tuple[HTMEdge, bool]]` (bool=True for forward/parent→child traversal). This commit also completes Milestone B1-1 (sensitivity primitives), which was originally scoped for Milestone B-1 — the primitives were built here since they were needed for the A3 test suite's FD cross-check. Commit `d81645c`. |
| 2026-06-24 | Claude (A4 implementation) | Implemented `sim/monte_carlo_fk.py` (`make_edge_rng`, `TrialData`, `MonteCarloFKEngine`). Key implementation decisions: (1) **RNG scheme:** `spawn_key = int.from_bytes(sha256(edge_name.encode()).digest()[:8], "little")`, then `np.random.default_rng(SeedSequence([master_seed, spawn_key]))` — ensures each edge's draws are fully decoupled from topological position and other edges; the exact hashing scheme is documented in the module docstring as immutable (changing it would break reproducibility of all prior runs). (2) **Root anchoring:** root frames' `frame_poses` initialized to `np.tile(np.eye(4), (n_trials,1,1))` — identity for every trial, no incoming edge, no perturbation, exactly as specified. (3) **Vectorized composition:** `np.einsum("nij,njk->nik", frame_poses[parent], perturbed_batch)` — no Python loop over trials. (4) **`edge_seed_log` stores integers** (the spawn key), not generator objects — purely for offline traceability/debugging. 18 tests written and passing in `tests/test_monte_carlo_fk.py`, including the Section 9 Item 3 required shared-edge consistency test, reproducibility with bit-for-bit identity check, and RNG independence test (adding an unrelated edge does not shift existing edges' draws). Commit `744c562`. |
| 2026-06-24 | Claude (A5 implementation) | Implemented `postprocess/stats.py` (Steps 1–7 of Section 6.8). Key implementation decisions: (1) **Rotation-error extraction:** `scipy.spatial.transform.Rotation.from_matrix(R_error).as_rotvec()` chosen over skew-symmetric extraction — computes the exact matrix logarithm (Rodrigues), equivalent at small angles but more robust at moderately larger angles; documented in module docstring. `R_error[i] = R_nominal.T @ R_perturbed[i]` extracted via `np.einsum("ji,njk->nik", R_nominal, R_perturbed)`. Columns [3:6] of `pose_error_vector_batch` output are directly the `(N,3)` rotvecs that `postprocess/bounding_shapes.py`'s `fit_rotation_cone()`/`fit_rotation_box()` require — no intermediate conversion. (2) **Envelope dict structure:** `{dof_label: {"min": float, "max": float}}` for both `frame_envelope_box` and `point_pair_envelope_box` — consistent, easy for `sim/allocation.py`'s validation pass to extract per-DoF achieved bounds. (3) **Batched HTM inverse:** closed-form `_htm_inverse_batch()` helper using `R.transpose(0,2,1)` and `np.einsum("nij,nj->ni", R_inv, t)`, matching `HTM.inverse()` semantics without calling `np.linalg.inv`. (4) **Component validation:** `relative_pose_trials()` calls `frame_graph._assert_same_component(frame_a, frame_b)` which raises `DisjointFramesError` with the locked message text — no re-implementation of the validation logic. 26 tests written and passing in `tests/test_stats.py`, including the shared-ancestor cancellation integration test (Section 2.4 / 9.1.3 analog: relative leaf1↔leaf2 envelope is >10× tighter than absolute leaf1 envelope when a 10 mm shared-edge tolerance cancels from the relative measurement). **Steps 8–9 (Pareto sensitivity breakdown) are deferred to Milestone B-1 task B1-3.** Commit `019eb34`. |
| 2026-06-24 | Claude (A6 implementation) | Created cross-cutting test infrastructure for the full suite. (1) **`tests/conftest.py`:** established the project-wide numerical tolerance convention — `DEFAULT_ATOL=1e-9` for near-exact floating-point composition checks and `SMALL_ANGLE_ATOL=1e-6` for checks expected to carry small-angle trig residual (e.g. `sin(δθ)` vs `δθ` at 1 mrad leaves ~1.7e-10 error, within `DEFAULT_ATOL` but documented explicitly). Three shared pytest fixtures: `two_edge_chain` (root→B→C, pure translation nominals), `three_edge_chain` (Rz=π/4 nominal on first edge + translation), `shared_frame_graph` (root→shared→leaf1, shared→leaf2). Module-level helpers (`make_tol`, `make_zero_tol`, `make_htm`) available for direct import. (2) **`tests/test_integration.py`:** 15 end-to-end tests across the A4+A5 pipeline: (a) *Two-edge translation stack-up* — single and stacked deltas, hand-verified to exact fixed-delta values using `_FixedToleranceSpec6`. (b) *Lever-arm coupling* — pivot edge with δrz=1 mrad, arm L=100 mm; hand-derived `dx_error = -δrz×L = -0.1 mm`, `rz_error = δrz = 1 mrad`; both confirmed against engine output at `SMALL_ANGLE_ATOL`. (c) *Local-frame perturbation routing* — local dx delta of 1 mm through a Rz(π/4) nominal splits into `0.001/√2` equally in world dx and dy, confirming right-multiply Section 2.2.2 convention is correctly expressed through the stats layer. (d) *TestFixtureSmoke* — confirms all three conftest fixtures are valid and runnable. (e) **`test_shared_edge_sampling_consistency`** — Section 9, Item 3 required standalone named module-level function (not inside a class), findable by `pytest -k`; demonstrates the shared-edge single-sample architectural property at 1000 trials. (3) **`tests/test_allocation.py`:** added `test_allocation_mc_validation_discrepancy` as Section 9, Item 4 required named placeholder (`pytest.mark.skip` until B-2). (4) **`README.md`:** added "Running Tests" section with CI one-liner and reference to Section 10 rule. 130 passed, 1 skipped. Commit `83b8ee3`. |
| 2026-06-25 | Claude (A7 implementation, at Project Owner's request) | **Milestone A complete.** Implemented two standalone example scripts in `examples/`. (1) **`examples/single_chain_fk_example.py`** (commit `3d7936d`): 3-edge CNC spindle alignment chain (`world → spindle_housing → bearing_seat → tool_tip`) with mixed `uniform`/`normal` tolerances on 3 edges. Runs 50,000-trial MC, prints worst-case envelope box and 50/90/95/99th-percentile table for `tool_tip`. Demonstrates angular amplification: a 0.001 rad housing face-flatness error translates to ~0.15 mm tip displacement over the 150 mm tool overhang. Text-only output; no B-1 dependencies. (2) **`examples/multi_chain_shared_frame_example.py`** (commits `3d7936d` initial, `8232c12` extended at Project Owner's request): optical bench with two identical lens seats sharing a common upstream frame. Extended from the originally-planned single-scenario demonstration into a **four-scenario sweep**: *Scenario 0* — zero seat tolerances, proves bench errors cancel to floating-point zero (MC relative envelope max = 1.99e-13 mm, confirmed ≈ 0 ✓); *Scenario 1* — translational seat errors only, shows direct RSS contribution (~0.08 mm relative dx range), no lever-arm amplification; *Scenario 2* — rotational seat errors only, reveals lever-arm amplification: rz ±0.001 rad uniform seat error across 100 mm separation = 0.200 mm relative dy range, exactly matching the analytic prediction `L × 2 × rz_bound = 100 × 2 × 0.001`; *Scenario 3* — full tolerances, dy/dz dominated by rotational lever arm. Includes a summary comparison table and the `L × δθ` design rule of thumb: at 100 mm separation, a 0.001 rad seat rotation is equivalent to a 0.1 mm translational error. (3) Minor linter cleanup (shebang removal, commit `178fae0`). 130 tests passing, 1 skipped. **Milestone A is now fully complete.** |

