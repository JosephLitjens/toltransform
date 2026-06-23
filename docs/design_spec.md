---
title: "TolTransform: Design Specifications & Project Plan"
subtitle: "A Kinematic Error-Budgeting Tool for Precision Machine Design"
author: "Living Engineering Document — Architecture, Module Specifications, and Phased Plan"
date: "Last updated: 2026-06-22"
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
│   └── conversions.py      # Thin wrappers around pytransform3d for xyz/euler/quat/screw <-> HTM
├── sim/
│   ├── monte_carlo_fk.py   # Forward verification engine: batched sampling + chain composition
│   ├── allocation.py       # Inverse allocation engine: Jacobian/sensitivity + EqualAllocation
│   │                       #   objective + MC validation pass
│   └── sampling.py         # Distribution sampling (uniform/normal via scipy.stats), sigma-level handling
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
    ├── test_transforms.py
    ├── test_tolerance.py
    ├── test_frame_graph.py
    ├── test_monte_carlo_fk.py
    └── test_allocation.py
```

## 5.2 Key Class Sketch (Conceptual — Not Final Code)

This is a conceptual sketch to align understanding; actual implementation will be written collaboratively, not dictated wholesale here.

- **`HTM`** (`core/transforms.py`): wraps a 4x4 NumPy array. Constructors: `from_xyz_euler(...)`, `from_screw(...)`, `from_quaternion(...)`, `from_matrix(...)`. Methods: `compose(other)`, `inverse()`, `to_xyz_euler()`, etc. Stores `input_representation` metadata.
- **`ToleranceSpec`** (`core/tolerance.py`): one per DoF, fields `distribution`, `bound`, `sigma_level`, `locked`. A `ToleranceSpec6` aggregates 6 of these for one edge. Method `sample(n_trials, rng)` returns an `(n_trials, 6)` array of perturbation vectors.
- **`Frame`** (`core/frame_graph.py`): just a name/id + optional metadata. No geometry.
- **`HTMEdge`** (`core/frame_graph.py`): `parent_frame`, `child_frame`, `nominal: HTM`, `tolerance: ToleranceSpec6`.
- **`FrameGraph`** (`core/frame_graph.py`): wraps a NetworkX `DiGraph`. Methods: `add_frame`, `add_edge`, `validate_dag()`, `path_between(frame_a, frame_b)`, `relative_transform_nominal(frame_a, frame_b)`.
- **`MonteCarloFKEngine`** (`sim/monte_carlo_fk.py`): `run(frame_graph, n_trials, rng_seed) -> TrialData`, where `TrialData` stores, for every Frame, an `(n_trials, 4, 4)` array (or the reduced 6-vector pose-error representation) — vectorized, no per-trial Python loop.
- **`AllocationEngine`** (`sim/allocation.py`): `solve(frame_graph, target_frame_pair, target_tolerance, objective=EqualAllocation()) -> ProposedToleranceSet`, plus `validate(proposed_set, n_trials) -> ValidationReport`.
- **`BoundingShapeFitter`** (`postprocess/bounding_shapes.py`): takes an `(n_trials, 3)` translation point cloud or `(n_trials, 3)` rotation-vector cloud, returns box/ellipsoid/cone parameters.

## 5.3 GUI-Engine Decoupling Principle (V1.0)

The GUI **never touches `core`/`sim` objects directly during editing** — it reads and writes the `io.schema` Pydantic models exclusively, and only constructs `core`/`sim` objects at "Run" time from the validated schema. This keeps the engine fully scriptable/headless-usable independent of the GUI, and means a future CLI or batch-runner needs zero GUI code.

---

# 6. Module Specifications

This section gives every code module in the architecture (Section 5.1) its own dedicated subsection. **At this stage, each entry is intentionally brief** — a short description of the module's responsibility, consistent with the architecture already locked in Sections 2–5. As each module is actually designed and built, its subsection should be expanded in place with: detailed inputs/outputs, function/class signatures, algorithms, edge cases, and any module-specific decisions — so this section becomes the living, authoritative spec for the codebase itself, not just a directory listing.

**Convention for future expansion:** when a module's spec is fleshed out, add a dated, named note (consistent with Section 11 Changelog practice) at the top of its subsection indicating who last substantially revised it.

## 6.1 `core/transforms.py`

**Responsibility:** Defines the canonical `HTM` class — the single representation of a homogeneous transformation matrix used everywhere else in the codebase. Provides constructors from every supported user-facing input format (xyz + Euler angles, raw 4x4 matrix, screw/exponential coordinates, quaternion + translation) and corresponding "to" converters, implemented as thin wrappers around `pytransform3d`. Provides `compose()` and `inverse()` operations. Retains metadata on the original input representation a user supplied, so the GUI can round-trip a user's preferred entry format without forcing premature canonicalization in the display layer.

*(Detailed spec pending — to be expanded during Milestone A, Task A1.)*

## 6.2 `core/tolerance.py`

**Responsibility:** Defines `ToleranceSpec` (a single DoF's tolerance: distribution type, bound, sigma-level, locked flag) and `ToleranceSpec6` (the aggregate of all 6 DoF for one HTM edge). Implements sampling (via `scipy.stats`) for both `"uniform"` (hard bound) and `"normal"` (statistical, with sigma-level conversion) distributions. Implements the local-frame, right-multiplication small-angle perturbation composition described in Section 2.2.2 — i.e., turns a sampled 6-vector `delta` into a perturbation HTM and composes it with a nominal HTM.

*(Detailed spec pending — to be expanded during Milestone A, Task A2.)*

## 6.3 `core/frame_graph.py`

**Responsibility:** Defines `Frame` (a named coordinate frame identity with no inherent geometry), `HTMEdge` (a toleranced directed transform between two Frames), and `FrameGraph` (the NetworkX-backed directed graph wrapping them). Implements DAG validation (enforcing the "max one incoming edge per Frame" rule for FK mode, per Section 2.3) with clear, actionable error messages on violation. Implements path-finding and relative-transform computation between any two arbitrary Frames in the graph, via their lowest common ancestor — the single operation that underlies both chain-output analysis and cross-chain point-pair analysis.

*(Detailed spec pending — to be expanded during Milestone A, Task A3.)*

## 6.4 `core/conversions.py`

**Responsibility:** Thin, isolated wrapper layer around `pytransform3d` conversion functions (xyz/Euler, quaternion, screw/axis-angle, raw matrix, all to/from canonical HTM). Kept separate from `transforms.py`'s `HTM` class so the third-party dependency surface is contained in one file — if `pytransform3d` ever needs to be swapped out or patched around, this is the only file that should need to change.

*(Detailed spec pending — to be expanded during Milestone A, Task A1.)*

## 6.5 `sim/monte_carlo_fk.py`

**Responsibility:** The forward tolerance verification engine (Mode 1, Section 3.1). Given a validated `FrameGraph`, a trial count `N`, and a random seed, vectorized-samples every edge's `ToleranceSpec6` exactly once per trial (Section 2.4), composes transforms along the graph to produce a sampled pose for every Frame in every trial, and returns a `TrialData` structure storing all per-trial, per-Frame pose data (as batched `(N,4,4)` arrays or reduced 6-vector pose-error arrays) for later post-processing. No Python-level loop over individual trials — fully vectorized NumPy operations.

*(Detailed spec pending — to be expanded during Milestone A, Task A4.)*

## 6.6 `sim/allocation.py`

**Responsibility:** The inverse tolerance allocation engine (Mode 2, Section 3.2). Computes the linear sensitivity (Jacobian) of a target relative pose (between any two Frames) with respect to each free, unlocked edge's 6-DoF perturbation. Implements the `AllocationObjective` interface and its sole v1 implementation, `EqualAllocation`, which solves for a uniform scale factor across free edges to meet a user-specified target envelope. Excludes `locked` edges/DoF from the free-variable vector. Runs a Monte Carlo validation pass on the proposed allocation (reusing `monte_carlo_fk.py`) and reports any discrepancy between the linear prediction and the nonlinear MC-validated result.

*(Detailed spec pending — to be expanded during Milestone B, Task B2.)*

## 6.7 `sim/sampling.py`

**Responsibility:** Low-level distribution sampling utilities used by `core/tolerance.py` — wraps `scipy.stats` uniform and normal sampling, handles the sigma-level-to-standard-deviation conversion for `"normal"` distributions, and centralizes the random-number-generator/seed handling so all sampling in the engine is reproducible given a fixed seed (Section 2.5, Section 9).

*(Detailed spec pending — to be expanded during Milestone A, Task A2.)*

## 6.8 `postprocess/stats.py`

**Responsibility:** Computes envelope statistics (min/max axis-aligned box, percentile tables, basic per-DoF histogram-ready binned data) for any Frame's stored trial data, and for the relative pose between any two Frames (Section 3.3), directly from the `TrialData` produced by the Monte Carlo engine — no re-simulation required.

*(Detailed spec pending — to be expanded during Milestone A, Task A5.)*

## 6.9 `postprocess/bounding_shapes.py`

**Responsibility:** Fits bounding shapes to stored trial point clouds: axis-aligned box, bounding ellipsoid/sphere for translation `(N,3)` point clouds, and a bounding cone (or per-axis box — pending final decision, Section 8) for rotation-vector `(N,3)` point clouds. This is the module responsible for producing the "bounding shape the engineer can use to make decisions" deliverable.

*(Detailed spec pending — to be expanded during Milestone B, Task B1.)*

## 6.10 `postprocess/reporting.py`

**Responsibility:** Generates Matplotlib plots from post-processed statistics and bounding shapes: per-DoF histograms, and 2D projections (XY/XZ/YZ, plus a separate rotation-error plot) of the fitted bounding shapes, per the V1.0 decision to use 2D projections rather than a rotatable 3D viewer (Section 8).

*(Detailed spec pending — to be expanded during Milestone B.)*

## 6.11 `io/schema.py`

**Responsibility:** Pydantic models defining the on-disk project data model: `Project`, `Frame`, `HTMEdge`, `ToleranceSpec`/`ToleranceSpec6`, `SimSettings` (mode, N trials, seed, distribution defaults, sigma-level default), and `SavedAnalysis` (persisted point-pair analyses, per Section 3.3). This schema is the **only** interface the GUI is allowed to read from or write to (Section 5.3) — `core`/`sim` objects are constructed from this schema only at "Run" time.

*(Detailed spec pending — to be expanded during Milestone B, Task B4.)*

## 6.12 `io/serializer.py`

**Responsibility:** JSON save/load logic for the `io.schema` models, including validation-on-load that surfaces clear, actionable errors (e.g., graph cycles, dangling Frame references, malformed tolerance specs) before the engine is ever invoked.

*(Detailed spec pending — to be expanded during Milestone B, Task B4.)*

## 6.13 `gui/main_window.py`

**Responsibility:** Top-level PySide6 application window; owns the currently-loaded `io.schema.Project`, hosts the other GUI panels (Sections 6.14–6.17) as docked or tabbed widgets, and owns the save/load/new-project actions.

*(Detailed spec pending — to be expanded during Milestone B, Task B5–B9. GUI work does not begin until Milestone A is complete, per Section 10.)*

## 6.14 `gui/graph_editor/`

**Responsibility:** GUI panel for building and editing the Frame graph — adding/removing Frames and HTM Edges, and entering each edge's nominal transform in any supported input format (Section 2.1).

*(Detailed spec pending — to be expanded during Milestone B, Task B5.)*

## 6.15 `gui/tolerance_editor/`

**Responsibility:** GUI panel for setting each edge's per-DoF tolerance specification: distribution type, bound, sigma-level (where applicable), and the locked flag (Section 2.2, Section 3.2).

*(Detailed spec pending — to be expanded during Milestone B, Task B6.)*

## 6.16 `gui/run_panel/`

**Responsibility:** GUI panel for configuring and triggering a simulation run: mode selection (FK verification vs. IK allocation), number of trials, random seed, and distribution settings.

*(Detailed spec pending — to be expanded during Milestone B, Task B7.)*

## 6.17 `gui/results_viewer/`

**Responsibility:** GUI panel for displaying simulation results: envelope summary tables, histograms, and 2D bounding-shape projections (Section 6.10), per Frame or per saved point-pair analysis.

*(Detailed spec pending — to be expanded during Milestone B, Task B8.)*

## 6.18 `gui/point_pair_panel/`

**Responsibility:** GUI panel for defining and persisting point-pair analyses (Section 3.3) — letting the user select any two Frames, view their relative-pose envelope, and save that analysis definition into the project file for future re-use.

*(Detailed spec pending — to be expanded during Milestone B, Task B9.)*

## 6.19 `examples/`

**Responsibility:** Hand-verified, fully worked example scripts demonstrating end-to-end usage of the engine (Section 9) — not a code module in the traditional sense, but a required deliverable of Milestone A (Task A7) that doubles as both onboarding material and a regression-test reference.

## 6.20 `tests/`

**Responsibility:** Unit and integration test suite. Houses the hand-calculable validation cases described in Section 9 for every core math module, plus dedicated regression tests for the Monte-Carlo-consistency-across-shared-edges property (Section 2.4) and the allocation engine's MC-validation discrepancy check (Section 3.2).

---

# 7. Project Phases & Time Allocation

**Total estimated effort: 130–170 hours**, split into two milestones. Hours assume AI-assisted ("vibe-coded") development with the project owner's light supervision and an ME/robotics background sufficient to validate the math by hand-checking simple cases.

## 7.1 Milestone A — V0.5 ("Proof of Concept / Minimum Viable Tool")

**Goal:** A working, hand-verified, script/CLI-level tool (no GUI) that performs forward Monte Carlo tolerance verification on a serial chain, with worst-case and statistical modes, and point-pair post-processing. Usable immediately for the project owner's own real ME work.

**Target: ~40–45 hours, achievable in 4–6 weeks at 8–12 hrs/week.**

| # | Task | Module(s) | Est. Hours |
|---|---|---|---|
| A1 | `HTM` class: canonical 4x4 storage, constructors/converters wrapping pytransform3d, `compose`/`inverse` | `core/transforms.py` | 4–6 |
| A2 | `ToleranceSpec` / `ToleranceSpec6`: per-DoF uniform + normal sampling via `scipy.stats`, sigma-level handling, local-frame right-multiply perturbation composition (Section 2.2.2) | `core/tolerance.py` | 5–8 |
| A3 | `Frame` / `HTMEdge` / `FrameGraph`: NetworkX-backed graph, DAG validation with clear error messages, path-finding between arbitrary frames | `core/frame_graph.py` | 4–6 |
| A4 | Monte Carlo FK engine: vectorized batched sampling and chain composition, per-Frame trial-data storage | `sim/monte_carlo_fk.py` | 8–10 |
| A5 | Post-processing stats: envelope (min/max box), percentile tables, basic histogram data, point-pair relative-transform stats from stored trial data | `postprocess/stats.py` | 6–8 |
| A6 | Hand-verified test cases: 2–3 simple chains (e.g., a 2-edge and a 3-edge chain) checked against manual small-angle calculations | `tests/` | 6–8 |
| A7 | Example script(s) / minimal CLI demonstrating end-to-end usage on a real or representative problem | `examples/` | 3–5 |

**Milestone A exit criteria:** Project owner can define a real kinematic chain from their own work in a Python script, run both worst-case and statistical Monte Carlo FK verification, and trust the printed/plotted envelope output because it has been checked against at least one hand calculation.

## 7.2 Milestone B — V1.0 ("Full Vision")

**Goal:** Adds inverse allocation, bounding-shape outputs, file save/load, and the full GUI on top of the proven V0.5 engine — no rework of Milestone A code, only additive layers.

**Target: ~90–125 additional hours.**

| # | Task | Module(s) | Est. Hours |
|---|---|---|---|
| B1 | Bounding shape fitting: bounding box/ellipsoid/sphere (translation), bounding cone (rotation), from stored trial point clouds | `postprocess/bounding_shapes.py` | 8–12 |
| B2 | Inverse allocation engine: Jacobian/sensitivity derivation, `EqualAllocation` objective, closed-form/near-closed-form solve, locked-edge handling | `sim/allocation.py` | 18–25 |
| B3 | MC validation pass for proposed allocations + discrepancy reporting | `sim/allocation.py` | (included in B2) |
| B4 | Pydantic schema + JSON save/load, validation-on-load with actionable errors | `io/schema.py`, `io/serializer.py` | 8–10 |
| B5 | GUI: graph/chain editor (add/edit Frames & Edges, multi-format HTM entry) | `gui/graph_editor/` | 10–14 |
| B6 | GUI: tolerance editor (per-DoF distribution, bound, sigma-level, lock toggle) | `gui/tolerance_editor/` | 6–8 |
| B7 | GUI: run panel (mode select, N trials, seed, distribution, run trigger) | `gui/run_panel/` | 4–6 |
| B8 | GUI: results viewer (envelope tables, histograms, 2D projections of bounding shapes via Matplotlib) | `gui/results_viewer/` | 10–14 |
| B9 | GUI: point-pair analysis panel + saved-analysis persistence in project file | `gui/point_pair_panel/` | 5–8 |
| B10 | Additional/expanded test coverage, especially around allocation engine edge cases (infeasible targets, all-locked edges, etc.) | `tests/` | 8–12 |
| B11 | Integration, bug fixing, end-to-end polish pass | — | 12–15 |

**Milestone B exit criteria:** A double-clickable (or `python main.py`-launchable) desktop application where a user with no Python experience could define a system, set tolerances, run both modes, and extract bounding-shape decisions — backed by the same validated engine from Milestone A.

## 7.3 Deferred / Explicitly Out of Scope for V1.0

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
| Rotation bounding shape (v1) | [TO BE CONFIRMED — pending project owner answer: per-axis box vs. single worst-case tilt-angle cone. Default assumption until confirmed: report both, lead with per-axis box.] |
| Bounding shape visualization | 2D projections (not a true rotatable 3D viewer) |
| GUI framework | PySide6, desktop, local-only, no server/browser split |
| Project file format | JSON via Pydantic schema |

---

# 9. Testing & Validation Philosophy

Because this tool produces engineering decisions about physical hardware tolerances, **correctness validation is non-negotiable and is budgeted explicitly as real project hours, not an afterthought.** The approach:

1. **Every new core math module ships with at least one hand-calculable test case** before being considered done — e.g., a single toleranced HTM perturbed by a known delta, checked against manually computed small-angle math.
2. **Chain-level test cases** (2–3 edges) are checked by composing the hand calculations from step 1 manually and comparing to engine output, for both worst-case and statistical modes.
3. **Point-pair and shared-frame test cases** explicitly verify that two downstream Frames sharing a common upstream edge see *identical* sampled perturbation for that edge within a given trial (Section 2.4) — this is a correctness-critical property, not just a nice-to-have, and deserves a dedicated regression test.
4. **Allocation engine validation** always includes the MC-validation-pass discrepancy check (Section 3.2, step 3) as part of its own test suite, not just as a runtime feature.

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

