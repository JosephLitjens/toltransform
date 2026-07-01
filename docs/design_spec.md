---
title: "TolTransform: Design Specifications & Project Plan"
subtitle: "A Kinematic Error-Budgeting Tool for Precision Machine Design"
author: "Living Engineering Document — Architecture, Module Specifications, and Phased Plan"
date: "Last updated: 2026-06-30 (deferred cleanup items)"
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

**Allocation objective:** `LoosestAllocation` (log-sum NLP) is the sole allocation objective. It maximises each free DoF's tolerance bound independently subject to the linear worst-case output-envelope constraints, guaranteeing no zero bounds even under Jacobian cross-coupling. Earlier `EqualAllocation` (worst-case linear sum) and `RSSAllocation` (statistical RSS) implementations were removed on 2026-06-28 — both collapsed to a single global scale factor that suppressed independent DoFs unnecessarily.

## 3.3 Point-Pair Post-Processing (Available in Both Modes)

After any Monte Carlo run (FK or the validation step of IK), the user may select **any two Frames** in the graph and request the relative pose tolerance/envelope between them, computed from the *already-stored* per-trial pose data (Section 2.3/2.4) — no re-simulation required. This is the mechanism that satisfies the original optical-systems use case: relative alignment tolerance between two components on different chains sharing a common mount.

---

# 4. Third-Party Libraries (Explicit Decisions — Do Not Re-Litigate)

| Library | Used For | Rationale |
|---|---|---|
| **NumPy** | All numerical arrays, vectorized Monte Carlo trial batches | Standard; vectorize trials as batched `(N,4,4)` array ops, never loop over trials in Python |
| **pytransform3d** | HTM <-> euler/quaternion/screw/axis-angle conversions | Battle-tested, avoids hand-rolled convention/edge-case bugs. **Do not use its perturbation/composition conventions for the tolerance model** — our local-frame right-multiply small-angle perturbation (Section 2.2.2) is bespoke and must be hand-implemented/tested |
| **NetworkX** | Frame graph: DAG validation, cycle detection, path-finding between arbitrary Frames | Avoids hand-written graph traversal code; directly supports Section 2.3 |
| **SciPy** | `scipy.stats` for sampling distributions (uniform, normal incl. sigma-level conversion); `scipy.optimize.minimize` (trust-constr) + `LinearConstraint` + `Bounds` for the `LoosestAllocation` log-sum NLP | Standard, reliable |
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
│   └── allocation.py       # Inverse allocation engine: LoosestAllocation NLP + MC validation
├── postprocess/
│   ├── stats.py            # Per-Frame and per-point-pair envelope stats: box, percentiles, sigma
│   ├── bounding_shapes.py  # Bounding box / ellipsoid / sphere (translation), bounding cone (rotation)
│   └── reporting.py        # Plot generation (histograms, 2D projections of bounding shapes)
├── persistence/
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
- **`AllocationEngine`** (`sim/allocation.py`): `solve(frame_graph, frame_a, frame_b, target_tolerance) -> dict[str, ToleranceSpec6]` (single-pair, LoosestAllocation), `solve_multi(frame_graph, targets) -> dict` (multi-pair), `allocate(...)` and `allocate_multi(...)` (with MC damping loop), plus `validate(proposed_set, frame_a, frame_b, n_trials, seed) -> ValidationReport`.
- **`BoundingShapeFitter`** (`postprocess/bounding_shapes.py`): takes an `(n_trials, 3)` translation point cloud or `(n_trials, 3)` rotation-vector cloud, returns box/ellipsoid/cone parameters.

## 5.3 GUI-Engine Decoupling Principle (V1.0)

The GUI **never touches `core`/`sim` objects directly during editing** — it reads and writes the `persistence.schema` Pydantic models exclusively, and only constructs `core`/`sim` objects at "Run" time from the validated schema. This keeps the engine fully scriptable/headless-usable independent of the GUI, and means a future CLI or batch-runner needs zero GUI code.

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
- *Used by:* `core/tolerance.py` (perturbation composition), `core/frame_graph.py` (`HTMEdge.nominal`), `sim/monte_carlo_fk.py`, `sim/allocation.py`, `persistence/schema.py` (serialization round-trip), and eventually `gui/graph_editor/`.
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
- *Used by:* `core/frame_graph.py` indirectly (an `HTMEdge` carries a `ToleranceSpec6`), `sim/monte_carlo_fk.py` (calls `.sample()` and `apply_perturbation_batch()` per edge), `sim/allocation.py` (constructs candidate `ToleranceSpec6` instances during allocation), `persistence/schema.py` (serializes/deserializes `ToleranceSpec`/`ToleranceSpec6`).
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
- *Used by:* `sim/monte_carlo_fk.py` (consumes `topological_edge_order()`, `all_edges()`, `root_frames()`), `sim/allocation.py` (consumes `path_edges_between()`, `adjoint()`, and `compute_sensitivity()` for its inverse solve, per the Mod 2 relocation), `postprocess/stats.py` (consumes `weakly_connected_components()` to validate point-pair queries, **and now also `path_edges_between()`/`compute_sensitivity()` for the Pareto sensitivity breakdown, per Mod 2**), `persistence/schema.py` (constructs a `FrameGraph` from a loaded `Project`), eventually `gui/graph_editor/`.
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

*(Last revised: 2026-06-28 — Claude, multi-pair IK allocation: `solve_multi`, `allocate_multi`, per-pair validation, GUI constraint list, results viewer per-pair display. See changelog 2026-06-28 entries.)*

**Responsibility:** The inverse tolerance allocation engine (Mode 2, Section 3.2): consumes the shared sensitivity primitive from `core/frame_graph.py` (Section 6.3) to solve the allocation objective, then validates and iteratively corrects the proposed tolerance set via Monte Carlo.

**Deliverables:**

- **`LoosestAllocation`** — the sole allocation objective; log-sum NLP (`maximize Σ log(b_ij)`) subject to linear worst-case output-envelope constraints — see Step 8 below for the full mathematical treatment. `EqualAllocation` and `RSSAllocation` were removed on 2026-06-28.
- `_build_result(free_edges, bounds: np.ndarray)` — internal helper that assembles `dict[str, ToleranceSpec6]` from a per-DoF bounds vector of shape `(6*N,)`; replaces the earlier scalar version that could only express a uniform allocation
- `AllocationEngine.allocate(...)` — single-pair entry point, wrapping the baseline linear solve in an **iterative nonlinear damping/correction loop** with binary-search angular refinement
- `AllocationEngine.allocate_multi(targets: list[tuple[str, str, ToleranceSpec6]], ...)` — **multi-pair entry point**: optimises per-DoF bounds subject to ALL specified point-pair constraints simultaneously; see Step 9 for full treatment
- `AllocationEngine.solve(...)` (single-pair linear step), `.solve_multi(...)` (multi-pair linear step), and `.validate(...)` (single MC pass), retained as internal building blocks
- `AllocationResult` output structure preserving both allocations, convergence status, target tolerance, method name, and (for multi-pair) `per_pair_validation` + `per_pair_targets` per-constraint breakdown
- `tests/test_allocation.py`, covering allocation-specific behavior (the Jacobian's own correctness is now tested in `tests/test_frame_graph.py`, Section 6.3) and dedicated tests for the damping loop, `LoosestAllocation`-specific properties, and multi-pair scenarios

**Granular Task List:**

1. **Sensitivity derivation — relocated (Mod 2, cross-review):** `compute_sensitivity()` and its underlying `adjoint()` helper now live in `core/frame_graph.py` (Section 6.3, Step 13). This module's job is to call `frame_graph.path_edges_between(frame_a, frame_b)` to get the relevant edges, then `compute_sensitivity(frame_graph, frame_a, frame_b, free_edge_names)` to get the `(6, 6*len(free_edge_names))` sensitivity matrix — do not re-implement the adjoint transform here.
2. ~~Implement the `AllocationObjective` abstract interface.~~ **(Superseded 2026-06-28)** `AllocationObjective` ABC and `EqualAllocation`/`RSSAllocation` implementations were removed. `LoosestAllocation` is the sole objective and is called directly — no interface indirection.
3. ~~Implement `EqualAllocation(AllocationObjective)`.~~ **(Superseded 2026-06-28)** Removed. Equal allocation collapsed all DoFs to a single scale factor `s = min_k(B_k / Σ|J[k,free]|)`, suppressing structurally independent DoFs unnecessarily.
4. Implement `AllocationEngine.solve(frame_graph, frame_a, frame_b, target_tolerance: ToleranceSpec6) -> dict[edge_name, ToleranceSpec6]` *(no `objective` parameter — LoosestAllocation always used)*:
   - Identify all edges on the path between `frame_a` and `frame_b` via `frame_graph.path_edges_between(frame_a, frame_b)` (Section 6.3, Step 12).
   - Partition into free (unlocked) and locked edges.
   - Compute `compute_sensitivity()` for the free edges, build the constraint matrix `(A, b)`, call `LoosestAllocation._run_nlp(A, b)`.
   - Return the full proposed per-edge `ToleranceSpec6` set (locked edges unchanged, free edges populated with the NLP solution).
5. **Iterative nonlinear damping/correction loop (locked decision, this revision):** Long serial chains with high-leverage joints exhibit meaningful geometric cross-coupling (`dx ≈ L·θ` — a small angular tolerance on an upstream joint sweeps a large positional arc by the time it reaches a distant downstream Frame). The closed-form linear allocation from Steps 1–4 does not account for this, and can produce an allocation that passes the linear sensitivity math but fails nonlinear Monte Carlo validation. `AllocationEngine.allocate()` is the top-level method that wraps `solve()` and `validate()` in a correction loop to address this:
   a. Call `solve(...)` (Steps 1–4) to produce the **baseline linear allocation**. Retain this unmodified — it is never overwritten (Step 5e).
   b. Call `validate(frame_graph, baseline_allocation, frame_a, frame_b, n_trials=1000, seed)` (Step 6) — a deliberately low-overhead `N_validate = 1000`-trial Monte Carlo pass, fast enough to run on every loop iteration without materially slowing down an interactive allocation workflow.
   c. If the validated achieved envelope is within the target bound (per DoF) for every DoF, the baseline allocation is accepted as final with no correction — record `iterations_used = 0`, `converged = True`.
   d. If validation fails (the achieved envelope exceeds the target on at least one DoF), apply a uniform damping factor `gamma` to the **free edges' angular DoF bounds only** (`rx, ry, rz` — translational DoF bounds are not damped by this loop, since the geometric-leverage failure mode this addresses is specifically angular-to-positional coupling): `new_bound = current_bound * gamma`, with `gamma` a tunable parameter, **default range `[0.7, 0.95]`** (locked this revision — document the exact default value chosen within that range once implemented, e.g., a fixed `gamma = 0.9` per iteration, or a configurable parameter exposed to the caller). Re-run `validate()` on the damped allocation and repeat.
   e. **Termination:** `max_iter` defaults to 30 (was 10; raised to reduce non-convergence on tight chains) and is exposed as a parameter to `allocate()` so callers can tune it. If the loop satisfies validation before reaching the cap, return with `converged = True` and the iteration count used. If the cap is reached without satisfying validation, **terminate gracefully (do not raise an exception that crashes the caller)** and return `converged = False` with the status message **exactly**: `"Allocation could not converge to target budget"`. The caller (the GUI's run panel, Section 6.16) is responsible for surfacing this status to the user.
   e2. **Binary-search angular refinement:** After the damping loop finds the first passing iteration, `_bisect_angular()` binary-searches between the last failing and first passing allocations (within 0.5% relative tolerance) to recover the loosest angular bounds that still pass MC. This eliminates the ~10% slack inherent in a fixed `gamma=0.9` step.
   f. **Output structure — `AllocationResult`:** the dataclass returned by `allocate()` must explicitly carry **both** allocations, never overwriting one with the other:
      ```
      AllocationResult
      ├── baseline_linear_allocation: dict[str, ToleranceSpec6]   # uncorrected, from Step 5a
      ├── corrected_allocation: dict[str, ToleranceSpec6]          # final, possibly == baseline if no correction was needed
      ├── converged: bool
      ├── iterations_used: int
      ├── status_message: str | None   # set to the locked non-convergence string on failure, else None
      ├── final_validation_report: ValidationReport                # from the last validate() call in the loop
      ├── target_tolerance: ToleranceSpec6 | None                  # echoed for display in results viewer
      └── method: str                                              # always "LoosestAllocation"
      ```
      The explicit preservation of both allocations is intentional and load-bearing: the *difference* between the baseline and corrected allocations is itself a direct, quantitative diagnostic of how much geometric-leverage coupling exists in the user's chain.
6. Implement `AllocationEngine.validate(frame_graph, proposed_tolerances, frame_a, frame_b, n_trials, seed) -> ValidationReport`:
   - Build a temporary `FrameGraph` (or mutate a copy) with the proposed tolerances applied to the free edges.
   - Run `MonteCarloFKEngine.run(...)` (Section 6.6) on it.
   - Compute the achieved relative-pose envelope between `frame_a` and `frame_b` from the resulting `TrialData` (via `postprocess/stats.py`, Section 6.8).
   - Compare achieved vs. target per DoF; populate a `ValidationReport` with both values and the discrepancy (absolute and percentage), and a per-DoF pass/fail flag (consumed directly by Step 5's loop termination check).
7. Write `tests/test_allocation.py`:
   - **Sensitivity primitive correctness:** not re-derived here — `compute_sensitivity()`'s finite-difference cross-check now lives in `tests/test_frame_graph.py` (Section 6.3, Step 14), since the primitive itself was relocated there this revision. This module's tests assume that primitive is already validated and focus on allocation-specific behavior below.
   - **`solve()` sanity check** (`test_solve_returns_all_free_edges`): simple 3-edge identity chain with symmetric target — confirms `solve()` returns exactly the free edges and, by symmetry, gives equal bounds across edges.
   - **Locked-edge case:** one edge locked to a known value, confirm the solver only adjusts the free edges and correctly accounts for the locked edge's fixed contribution.
   - **All-edges-locked edge case:** confirm the solver detects an infeasible/over-constrained situation (no free edges to solve for) and raises a clear, specific error rather than silently returning a meaningless result.
   - **MC validation discrepancy reporting:** confirm `validate()` correctly flags a case where the linear allocation under- or over-shoots the nonlinear MC-validated result (construct a case with deliberately large nominal rotation offsets between edges to induce a meaningful nonlinearity, since pure small-angle propagation through near-zero nominal offsets may not exercise this path).
   - **Damping loop convergence test (new, locked decision):** using the same high-leverage geometry constructed for the discrepancy test above, confirm `allocate()` converges within `max_iter`, confirm `corrected_allocation`'s angular bounds are strictly tighter than `baseline_linear_allocation`'s, and confirm the final validation pass actually satisfies the target.
   - **Damping loop non-convergence test (new, locked decision):** construct a deliberately infeasible target (e.g., an unreasonably tight target on a chain with most edges locked) and confirm `allocate()` returns `converged=False` with `status_message == "Allocation could not converge to target budget"` after exactly `max_iter` iterations, without raising an uncaught exception.
   - **LoosestAllocation tests (new, this revision):** see Step 8 below.

8. ✅ **Done (2026-06-28)** — `LoosestAllocation`: log-sum NLP for per-DoF loosest allocation. *(Note: `AllocationObjective` ABC and `EqualAllocation`/`RSSAllocation` removed 2026-06-28 — see Step 8 history and changelog.)*

   **Why single-scale-factor methods are suboptimal.**  `EqualAllocation` and `RSSAllocation` (the earlier implementations, now removed) collapsed the allocation problem to a single global scale factor `s` applied uniformly to every free DoF on every edge.  The binding constraint was the output DoF that gave the smallest `s_k`:

   ```
   EqualAllocation:   s_k = B_k / Σ_{free (i,j)} |J[k, 6i+j]|
   RSSAllocation:     s_k = B_k / sqrt(Σ_{free (i,j)} J[k, 6i+j]²)
   s = min_k(s_k),   b_ij = s  for every free (i,j)
   ```

   The binding DoF constrains every other DoF even if they are insensitive to it.  Concretely: a 2-edge identity chain with B_dx = 0.01 (tight) and B_rz = 10.0 (loose) gives `s = 0.005` for all DoFs — rz gets 0.005 when it could have 5.0 (1000× waste).

   **Correct formulation — the allocation LP.**  The proper statement of "loosest possible tolerances" is:

   ```
   maximize  Σ_{free (i,j)} b_ij
   subject to:
     Σ_{free (i,j)} |J[k, 6i+j]| · b_ij  ≤  B_k    for each active output DoF k
     b_ij  ≥  0
   ```

   This is a linear program.  **Critical problem:** nominal rotations between frames create cross-coupling in the Jacobian — both `ry` and `rz` inputs can appear in the same output constraint row (e.g., `J[4, ry_col] ≠ 0` and `J[4, rz_col] ≠ 0` simultaneously when a frame has a non-trivial nominal rotation about `x`).  The LP finds a vertex of the feasible polytope, which in this case assigns all budget to whichever of the two inputs it encounters first and sets the other to exactly zero.  Physically: zero tolerance is unmanufacturable and meaningless as a design specification.

   **Fix — log-sum NLP (locked, this revision).**  Replace the linear-sum objective with the log-sum:

   ```
   maximize  Σ_{free (i,j)} log(b_ij)
   subject to:
     Σ_{free (i,j)} |J[k, 6i+j]| · b_ij  ≤  B_k    for each active output DoF k
     b_ij  ≥  ε   (ε = 10⁻¹⁰, a small positive floor)
   ```

   Key mathematical properties of the log-sum objective:
   - **No zero solutions:** `log(b) → −∞` as `b → 0`, so the optimizer is infinitely penalized from driving any variable to zero.  Every free DoF always receives a positive bound.
   - **Balanced allocation under symmetric coupling:** when `J[k, col1] = J[k, col2]` (equal sensitivities in the same constraint row), the log-sum KKT conditions give `b_col1 = b_col2` — equal shares, not the arbitrary vertex the LP finds.
   - **Proportional allocation under asymmetric coupling:** for a single active constraint with coefficients `a_j` on variables `b_j`, the log-sum KKT gives `b_j* ∝ 1/a_j · (λ⁻¹)` where `λ` is the constraint's Lagrange multiplier — i.e., DoFs with lower sensitivity get proportionally looser bounds.
   - **Correct unbounded fill for zero-sensitivity DoFs:** a DoF with `J[k, col] = 0` for all `k` is genuinely unconstrained; its log-gradient pushes it toward `CAP = 10⁶`, correctly representing "this DoF can be as loose as desired."
   - **Convexity:** the constraints are linear (hence convex); the objective is concave (maximization of a concave function); the feasible set is a convex polytope.  The problem is convex NLP, so any local optimum found by the solver is the global optimum.

   **Numerical conditioning fix.**  The initial (SLSQP) implementation failed on chains with widely-varying target magnitudes (e.g., `B_dx = 0.001 rad`, `B_rz = 10 rad` — a 10 000:1 ratio).  The root cause: the earlier warm start used the global equal-allocation scale `s = min_k(B_k / Σ|J_k|)` for all DoFs, placing rz at `s ≈ 10⁻⁴` when the optimum is `≈ 5`.  At this point the log-gradient is `−1/s ≈ −10 000`, while the gradient for dx (already near its optimum) is `≈ −200`.  SLSQP's linesearch diverged under this 50:1 gradient mismatch.

   Two fixes (both locked):
   1. **Per-DoF warm start.**  For each free variable `j`, find the tightest constraint it appears in and estimate a fair per-DoF share:
      ```
      x0[j] = 0.9 × min_k { B_k / (|J[k,j]| × n_sharing_k) }
      ```
      where `n_sharing_k` = number of free variables contributing to constraint `k`.  This places each variable near its own binding point; gradient magnitudes are uniformly `O(1/b_j*)` with no cross-scale mismatch.
   2. **Switched to `scipy.optimize.minimize(method="trust-constr")` with `LinearConstraint`.**  The trust-region method tolerates gradient scale differences (it rescales implicitly via the Hessian approximation); the linesearch-based SLSQP does not.

   **`_bisect_angular` bug fix (locked, this revision).**  The binary-search refinement had a subtle bug: `lo_scale` (the current lower bound on the absolute angular value) was updated each passing iteration, but `lo` (the allocation object passed to `_scale_angular`) was never updated.  On the second bisection iteration, `ratio = mid_scale / lo_scale` used the *updated* `lo_scale` rather than the original `lo` allocation's bound, so `_scale_angular(lo, ratio)` produced the wrong absolute value:

   ```
   # Before fix (wrong):
   ratio = mid_scale / lo_scale   # lo_scale updated each iter, lo is still original
   mid = _scale_angular(lo, ratio)
   # → mid.b_rz = lo_original.b_rz × (new_mid / prev_passing_mid)  ← wrong
   ```

   Fix: introduce `base_scale = lo_scale` before the loop and always divide by it:

   ```
   # After fix (correct):
   base_scale = lo_scale   # never updated
   ...
   ratio = mid_scale / base_scale   # always relative to original lo
   mid = _scale_angular(lo, ratio)
   # → mid.b_rz = lo_original.b_rz × (new_mid / lo_original.b_rz) = new_mid  ✓
   ```

   This bug was latent because `_bisect_angular` is only called when the damping loop finds a passing iteration, and when all angular bounds are equal the first bisection iteration usually meets the 0.5% tolerance criterion immediately.  With `LoosestAllocation`, heterogeneous bounds mean more bisection iterations are common, making the bug visible.

   **`_build_result` refactor (locked, this revision).**  The helper that assembles per-edge `ToleranceSpec6` dicts was previously `_build_result(free_edges, s: float)` — a single scalar applied uniformly.  Refactored to `_build_result(free_edges, bounds: np.ndarray)` where `bounds` is shape `(6*N,)` with one entry per `(edge, DoF)`.  `LoosestAllocation` passes the NLP solution vector directly.

   **Tests for LoosestAllocation (in `tests/test_allocation.py`):**
   - `test_solve_fills_independent_dof_budgets`: 2-edge identity chain, `B_dx = 0.01` / `B_rz = 10.0`. Asserts the tight dx constraint does not drag rz down — both constraints are filled to their own limits independently.
   - `test_loosest_allocation_mc_validation_passes`: simple 3-edge chain, `solve()` result validated at 1.5× target (50% margin) — confirms the NLP output is feasible under MC.
   - `test_loosest_allocation_no_zero_bounds_on_coupled_chain` **(regression test, critical)**: single-edge chain with `Ry(π/2)` nominal — Jacobian cross-coupling between ry and rz inputs.  Asserts all 6 per-DoF bounds `> 1e-6`.  This test fails with a naive linear-sum LP (vertex degeneracy) and passes with the log-sum NLP.
   - `test_loosest_allocation_lever_arm_converges`: `solve()` + damping loop on the lever-arm chain — confirms the NLP + correction loop pipeline converges end-to-end.

9. ✅ **Done (2026-06-28)** — `AllocationEngine.solve_multi` / `allocate_multi`: simultaneous multi-pair IK allocation.

   **Motivation and use case.**  The single-pair API (`solve` / `allocate`) expresses one constraint: the relative pose between frame A and frame B must stay within a given envelope.  For optical assemblies with three or more elements, the practitioner needs every element-to-element relationship to be within tolerance at the same time — e.g., mirror 1 ↔ mirror 2 AND mirror 1 ↔ mirror 3.  Running `allocate()` independently for each pair and merging results is incorrect when any edge is **shared** between two pairs: the per-pair solutions may assign different bounds to the same edge, and naively merging them (e.g., taking the tighter) discards information about how the other pair's budget depends on that edge.

   **Shared-edge coupling and why stacking is correct.**  Consider a Y-topology: `root → base → {mirror_1, mirror_2}`.  The edge `base → root` (call it `e_rb`) lies on the path for both pairs (root, mirror_1) and (root, mirror_2).  If pair 1 has a tight dx target and pair 2 has a loose dx target, the NLP must find bounds for `e_rb.dx` that satisfy BOTH constraints simultaneously:

   ```
   b_rb_dx + b_m1_dx  ≤  B1_dx    (pair 1, tight)
   b_rb_dx + b_m2_dx  ≤  B2_dx    (pair 2, loose)
   ```

   Running pair 1 independently gives `b_rb_dx ≈ B1_dx / 2` — correct.  Running pair 2 independently gives a looser `b_rb_dx` — also locally correct, but inconsistent with pair 1.  The stacked formulation uses **both rows simultaneously** and automatically finds `b_rb_dx` constrained by the tighter pair 1, while `b_m2_dx` absorbs the remaining pair 2 budget.  This is the only formulation that is globally consistent.

   **Mathematical construction — stacked constraint matrix.**

   Given `P` pairs `{(a_p, b_p, B_p)}_{p=1}^{P}`:

   1. **Union of free edges.** Collect all free edges appearing on any pair's path (preserving insertion order, deduplicating by name): `E_free = ∪_p path_edges_free(a_p, b_p)`. Let `N = |E_free|` and index them `0 … N−1`.

   2. **Padded Jacobians.** For each pair `p`, call `compute_sensitivity(fg, a_p, b_p, pair_p_edge_names)` to get a compact `(6, 6·M_p)` matrix (only the `M_p` edges on pair p's path). Embed this into a full-width matrix by placing each edge's 6-column block at its global index position; columns for edges NOT on pair p's path are zero:
      ```
      J_full_p ∈ ℝ^{6 × (6N)},   J_full_p[:, 6g:6g+6] = compact_block_g  if edge g ∈ path_p
                                                          = 0                otherwise
      ```

   3. **Stacked constraints.** For each pair `p` and each active output DoF `k` (where `|J_full_p[k, free_cols]|.sum() > 0` and `B_p[k] > 0`), append a constraint row:
      ```
      A_rows.append( np.abs(J_full_p[k, free_idx]) )
      b_vals.append( B_p[k] )
      ```
      The final matrix `A ∈ ℝ^{C × n_free}` (where `C = Σ_p n_active_k_p`) has one row per active (pair, DoF) combination.  Non-path edges have zero entries in their rows; active path edges have nonzero entries.

   4. **NLP.** Pass `(A, b)` to `LoosestAllocation._run_nlp(A, b)` — the same log-sum optimizer used for single-pair allocation, now with `C` constraint rows instead of ≤ 6.  The problem remains convex (linear constraints, concave objective) and the solver is unchanged.  The solution vector `x ∈ ℝ^{n_free}` gives per-DoF bounds for all edges in `E_free`.

   **`LoosestAllocation._run_nlp(A, b)` extraction (locked, this revision).**  To allow both `solve()` (single pair) and `solve_multi()` (multiple pairs) to share the NLP without code duplication, the optimization logic was extracted from `solve()` into a new instance method `_run_nlp(A, b) -> np.ndarray`.  `solve()` now builds `(A, b)` from its single Jacobian and calls `_run_nlp`; `solve_multi()` builds the stacked `(A, b)` and calls `_run_nlp` directly.

   **Multi-pair damping loop.**  The nonlinear correction loop (`allocate_multi`) is structurally identical to the single-pair `allocate`:
   - "Failed" = any of the `P` pairs fails MC validation.
   - One MC simulation is run (all pairs' paths share the same perturbed FK trial data, so one call to `MonteCarloFKEngine.run()` suffices).
   - `_mc_validate_multi` then calls `point_pair_envelope_box()` once per pair against the shared trial data, producing one `ValidationReport` per pair.
   - When any pair fails, `gamma` is applied to ALL free angular bounds globally (not per-pair) — this keeps the allocation consistent across pairs and avoids the case where tightening angular bounds for one pair loosens them for another via the shared NLP solution.
   - The bisection refinement (`_bisect_angular_multi`) validates all pairs simultaneously.

   **`AllocationResult` additions.**
   - `per_pair_validation: list[tuple[str, str, ValidationReport]] | None` — one entry per pair; `None` for single-pair `allocate()` results.
   - `per_pair_targets: list[tuple[str, str, ToleranceSpec6]] | None` — the original target for each pair, preserved for display in the results viewer.

   **GUI changes (this revision).**
   - `gui/run_panel/run_panel_widget.py`: the single frame-pair UI is replaced by a dynamic list of `_ConstraintRowWidget` objects, each containing its own Frame A / Frame B combos and 6-DoF target spinboxes.  "Add Constraint" appends a new row; "✕" removes it (disabled when only one row remains).  The worker calls `allocate_multi()` with the full list.
   - `gui/results_viewer/results_viewer_widget.py`: the "Achieved Envelope" section is rebuilt dynamically — one `QGroupBox` per pair, labeled with the frame pair name and pass/fail in its title color.  Each group shows a `DoF | Target ± | Min | Max | Pass?` table, using `per_pair_targets` to populate the Target column so users can directly self-verify achieved vs. requested.

   **Tests for multi-pair allocation (5 new, in `tests/test_allocation.py`):**
   - `test_solve_multi_shared_edge_respects_tighter_constraint` **(key correctness test)**: Y-topology graph, pair 1 tight dx / pair 2 loose dx.  Asserts `e_rb.dx ≈ 0.005` (shared edge constrained by tighter pair 1) and `e_bm2.dx ≈ 0.995` (non-shared edge fills pair 2's full budget).
   - `test_solve_multi_independent_pairs_unaffected`: disjoint paths, asserts each pair's edge gets its own independent budget without cross-interference.
   - `test_allocate_multi_result_structure`: verifies `per_pair_validation` is populated with correct frame labels and that all three edges appear in the allocation.
   - `test_allocate_multi_mc_validation_check_target`: angular-locked single-edge pairs, `solve_multi` output validated at 1.5× budget — confirms multi-pair MC validation reports all pairs passing with margin.
   - `test_allocate_multi_lever_arm_two_pairs`: two disjoint lever-arm chains run as a multi-pair problem; both must converge within `max_iter=20`, confirming the damping loop handles multi-pair scenarios end-to-end.

**Interfaces:**

- *Depends on:* `core/frame_graph.py` (`FrameGraph`, `path_edges_between()`, `adjoint()`, `compute_sensitivity()`), `core/tolerance.py` (`ToleranceSpec6`), `core/transforms.py` (`HTM`), `sim/monte_carlo_fk.py` (`MonteCarloFKEngine`), `postprocess/stats.py` (`point_pair_envelope_box`), `scipy.optimize` (`minimize`, `LinearConstraint`, `Bounds`).
- *Used by:* `gui/run_panel/` and `gui/results_viewer/`.
- *Public API (conceptual):*
  ```
  LoosestAllocation   # sole allocation objective: log-sum NLP, per-DoF bounds

  AllocationEngine.solve(frame_graph, frame_a, frame_b,
                          target_tolerance) -> dict[str, ToleranceSpec6]
  AllocationEngine.solve_multi(frame_graph,
                                targets: list[tuple[str, str, ToleranceSpec6]]) -> dict[str, ToleranceSpec6]
  AllocationEngine.validate(frame_graph, proposed_tolerances, frame_a, frame_b,
                             n_trials, seed) -> ValidationReport
  AllocationEngine.allocate(frame_graph, frame_a, frame_b, target_tolerance,
                             n_validate=1000, gamma=0.9, max_iter=30, seed=...) -> AllocationResult
  AllocationEngine.allocate_multi(frame_graph, targets: list[tuple[str, str, ToleranceSpec6]],
                                   n_validate=1000, gamma=0.9, max_iter=30, seed=...) -> AllocationResult

  AllocationResult(baseline_linear_allocation, corrected_allocation, converged,
                    iterations_used, status_message, final_validation_report,
                    target_tolerance, method,
                    per_pair_validation,   # list[(str, str, ValidationReport)] | None
                    per_pair_targets)      # list[(str, str, ToleranceSpec6)] | None
  # Note: adjoint() and compute_sensitivity() are NOT part of this module's API —
  # they live in core/frame_graph.py (Section 6.3) and are imported from there.
  ```

---

## 6.8 `postprocess/stats.py`

*(Last revised: 2026-06-25 — Claude, B1-3 implementation; **Steps 1–7 implemented in A5, commit `019eb34`, 26 tests passing. Steps 8–9 implemented in B1-3, commit `fd8a08b`, 8 new tests added to `tests/test_stats.py`.** Rotation-error extraction decision: `scipy.spatial.transform.Rotation.from_matrix(R_error).as_rotvec()` — exact matrix logarithm, equivalent to skew-extraction at small angles but more robust at moderate angles; documented in module docstring.)*

**Responsibility:** Computes envelope statistics for any single Frame, and for the relative pose between any two Frames in the same component, directly from a `TrialData` instance — no re-simulation, no graph traversal beyond a same-component validation check. **As of this revision, also computes the Pareto-sorted per-edge sensitivity breakdown** (Mod 2) — answering "which tolerance edge dominates this error budget" as a standalone forward-analysis capability, independent of whether inverse allocation is ever run.

**Deliverables:**

**A5 scope (Steps 1–7) — ✅ Implemented (commit `019eb34`, 26 tests):**
- Per-Frame envelope/percentile/histogram-data functions (`frame_envelope_box`, `frame_percentiles`, `frame_histogram_data`)
- Point-pair relative-pose statistics: `relative_pose_trials`, `relative_pose_nominal`, `point_pair_envelope_box` — exploiting the "absolute pose already stored" property (Section 6 top-level note)
- `pose_error_vector_batch()` — the shared extraction primitive producing `(N,6)` error vectors in `[dx,dy,dz,rx,ry,rz]` order with rotvec `ω=θu` rotation columns compatible with `postprocess/bounding_shapes.py`
- `tests/test_stats.py` — 26 tests, including shared-ancestor cancellation integration check

**B1-3 scope (Steps 8–9) — ✅ Implemented (commit `fd8a08b`, 8 new tests in `tests/test_stats.py`):**
- `compute_tolerance_sensitivities(frame_graph, frame_a, frame_b) -> ParetoSensitivityReport` — Pareto-sorted percentage-contribution breakdown per edge/DoF
- `ParetoSensitivityReport` dataclass + `to_ascii_chart(top_n=10) -> str` rendering hook
- **Key decisions:** `trial_data` omitted (tolerance specs live on `FrameGraph.edges[name].tolerance`, not on `TrialData`); variance formula `uniform → b²/3`, `normal → (b/sigma_level)²`; percentage contribution summed unweighted across all 6 output DoF; first-order caveat mandatory as on-chart annotation in `postprocess/reporting.py` (not just prose)

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
8. ✅ **Done (B1-3, `fd8a08b`)** Implement `compute_tolerance_sensitivities(frame_graph, frame_a, frame_b) -> ParetoSensitivityReport`: computes the first-order percentage contribution of each toleranced edge/DoF on the path between `frame_a` and `frame_b` to the total variance of the target relative pose.
   - Uses `frame_graph.path_edges_between(frame_a, frame_b)` and `compute_sensitivity(...)` from `core/frame_graph.py` — no second Jacobian implementation.
   - **Variance formula (locked):** `uniform → b²/3`, `normal → (b/sigma_level)²` — consistent with the RSS benchmark (Section 9.1.1).
   - **Weighting (locked):** percentage contribution summed **unweighted across all 6 output DoF** — total output variance is the scalar sum of all 6 DoF variance contributions from all edges.
   - Returns `ParetoSensitivityReport` dataclass: `ranked_contributions: list[tuple[str, str, float]]` (edge_name, dof_label, pct), `total_variance: float`.
   - First-order-linear-approximation caveat documented in the function's docstring AND as a mandatory on-chart annotation in `postprocess/reporting.py` (Section 6.10 Step 4 — locked there, cannot be suppressed).
9. ✅ **Done (B1-3, `fd8a08b`)** Implement `ParetoSensitivityReport.to_ascii_chart(top_n=10) -> str`: bar-chart-style text output for quick terminal/script use. Groups contributions beyond `top_n` into a single "(others)" entry.
10. Write `tests/test_stats.py` — **A5 tests ✅ implemented (26 tests); B1-3 Pareto tests deferred:**
   - ✅ Construct a small synthetic `TrialData` with known, hand-computed pose errors (don't run the full MC engine — directly build the `TrialData` fields) and confirm `frame_envelope_box` returns the expected min/max.
   - ✅ Confirm `relative_pose_trials` between a Frame and itself returns identity for every trial (trivial sanity check).
   - ✅ Confirm `point_pair_envelope_box` between two Frames sharing a common upstream edge correctly reflects that the shared edge's contribution cancels out of the *relative* error (a key qualitative check that validates the whole point of Section 2.4's shared-sampling design — relative tolerance between two downstream points should be tighter than either point's absolute tolerance when they share a noisy common ancestor edge).
   - ✅ Confirm the different-component case raises a clear error.
   - ✅ **Done (B1-3, `fd8a08b`) Pareto sensitivity correctness:** construct a simple chain with one dominant-tolerance edge and several much-tighter edges, confirm `compute_tolerance_sensitivities()` ranks the dominant edge first with a percentage contribution that matches a hand-computed variance ratio.
   - ✅ **Done (B1-3, `fd8a08b`) Pareto contributions sum to ~100%:** confirmed the reported percentages across all edge/DoF entries sum to approximately 100% for a representative chain.
   - ✅ **Done (B1-3, `fd8a08b`) Uniform vs. normal consistency:** confirmed the variance formula (`bound²/3` for uniform, `sigma²` for normal) produces self-consistent Pareto rankings when the same chain is run with all-uniform and all-normal tolerances at equivalent variance.

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

*(Last revised: 2026-06-25 — Claude, B1-2 implementation, commit `ef4c89c`, 25 tests passing. Original spec note: design-spec update hardening the rotation-vector type contract per cross-review.)*

**Responsibility:** Fits bounding shapes (box, ellipsoid/sphere for translation; cone or per-axis box for rotation) to the error-vector point clouds produced by `postprocess/stats.py`. This is the module responsible for the "bounding shape the engineer can use to make decisions" deliverable (Section 1.1, Section 3.1).

**Deliverables:**

- Axis-aligned bounding box fitting (translation and rotation, trivial — already substantially covered by `stats.py`'s envelope functions; this module focuses on the non-trivial shapes)
- Bounding ellipsoid/sphere fitting for translation point clouds
- Bounding cone fitting for rotation-vector point clouds — **confirmed lead representation, locked 2026-06-23** (per-axis box still implemented as a cheap secondary cross-check, not as the primary reported value)
- `tests/test_bounding_shapes.py` (new test file — add to the tree)

**Granular Task List:**

1. ✅ **Done (B1-2, `ef4c89c`)** Implement `fit_bounding_sphere(points: np.ndarray) -> dict(center, radius)`. **Implementation decision (locked):** uses centroid + max distance — `center = np.mean(pts, axis=0)`, `radius = max(||p_i - center||)`. Not the minimum enclosing sphere, but a simpler, always-correct conservative bound; appropriate for an error-budgeting tool. Documented in the function's docstring.
2. ✅ **Done (B1-2, `ef4c89c`)** Implement `fit_bounding_ellipsoid(points: np.ndarray, coverage=1.0) -> dict(center, axes_lengths, axes_directions)`. **Critical implementation decision (locked):** for `coverage=1.0`, **per-axis independent max projection does NOT guarantee enclosure** (a point far along two PCA axes simultaneously can escape). Fix: **uniform-scale approach** — compute each point's Mahalanobis distance `r²_i = Σ_j (projection_ij / sigma_j)²`, then scale all PCA axes uniformly by `sqrt(max(r²_i))`. Enclosure is guaranteed by construction (the scaled ellipsoid is the minimum Mahalanobis-distance sphere containing all points). For `coverage<1.0`: scale by `sqrt(chi2.ppf(coverage, df=3))`.
3. ✅ **Done (B1-2, `ef4c89c`)** Rotation bounding shapes — cone (lead) and box (secondary):
   - `fit_rotation_cone(rotvecs: np.ndarray) -> dict(max_angle, mean_axis)` — **primary.** `max_angle = max(||rotvec_i||)`. `mean_axis` = normalized mean direction; falls back to `[0, 0, 1]` if all magnitudes are zero (avoids NaN).
   - `fit_rotation_box(rotvecs: np.ndarray) -> dict(min, max)` — **secondary.** Delegates to `fit_bounding_box()` after the shape check — no separate implementation.
4. ✅ **Done (B1-2, `ef4c89c`)** Interface type-hardening: `_check_rotvec_shape(rotvecs)` private helper raises `ValueError` with a clear message if `rotvecs.shape` is `(N,4,4)` (or anything other than `(N,3)`). Called at entry to both `fit_rotation_cone()` and `fit_rotation_box()`. Convention `ω = θu` documented by name in both docstrings.
5. ✅ **Done (B1-2, `ef4c89c`)** Implement `fit_bounding_box(points: np.ndarray) -> dict(min, max)`: `{"min": points.min(axis=0), "max": points.max(axis=0)}`. Operates on arbitrary 3D point clouds; reused by `fit_rotation_box()`.
6. ✅ **Done (B1-2, `ef4c89c`)** `tests/test_bounding_shapes.py`: 25 tests, 6 test classes:
   - `TestFitBoundingBox`: axis-aligned bounds on known synthetic data
   - `TestFitBoundingSphere`: centroid+max distance enclosure check, single-point degenerate case
   - `TestFitBoundingEllipsoid`: coverage=1.0 uniform-scale enclosure guarantee (the key invariant), coverage<1.0 strictly smaller, near-degenerate cases
   - `TestFitRotationCone`: correct max_angle, mean_axis normalization, zero-magnitude fallback
   - `TestFitRotationBox`: delegates to fit_bounding_box, bounds correct
   - `TestTypeHardening`: confirms `(N,4,4)` input raises `ValueError` for both `fit_rotation_cone` and `fit_rotation_box`

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

*(Last revised: 2026-06-25 — Claude, B1-4 implementation, commit `e258538`, 17 smoke tests. Original spec note: design-spec update adding Pareto sensitivity chart rendering per Mod 2, cross-review.)*

**Responsibility:** Generates Matplotlib plots from the statistics (`postprocess/stats.py`) and fitted shapes (`postprocess/bounding_shapes.py`): per-DoF histograms, 2D projections of bounding shapes, and **as of this revision, the Pareto sensitivity breakdown chart** (Mod 2), per the locked decision to use 2D projections rather than a rotatable 3D viewer (Section 8).

**Deliverables:**

- Per-DoF histogram plotting function
- 2D projection plotting for translation bounding shapes (XY/XZ/YZ)
- 2D plotting for rotation bounding box/cone representations
- Pareto sensitivity bar chart plotting (new, Mod 2) — the graphical counterpart to `ParetoSensitivityReport.to_ascii_chart()` (Section 6.8)
- `tests/test_reporting.py` (smoke tests only — rendering correctness is best verified visually, not asserted numerically)

**Granular Task List:**

1. ✅ **Done (B1-4, `e258538`)** Implement `plot_histogram(counts, bin_edges, dof_label, ax=None) -> matplotlib.axes.Axes`: single-DoF histogram, returns the Axes.
2. ✅ **Done (B1-4, `e258538`)** Implement `plot_translation_projection(points: np.ndarray, bounding_shape: dict, plane: Literal["xy","xz","yz"], ax=None) -> matplotlib.axes.Axes`. **Implementation note:** scatter uses private `_maybe_subsample(pts, rng)` helper with `_SCATTER_MAX_POINTS = 2000`; 2D ellipsoid projection uses covariance-slice + `eigh`:
   ```python
   C_3d = V @ np.diag(axes_lengths**2) @ V.T
   C_2d = C_3d[np.ix_([ai, aj], [ai, aj])]       # 2×2 sub-matrix for the plane
   evals, evecs = np.linalg.eigh(C_2d)             # ascending eigenvalues
   angle = np.degrees(np.arctan2(evecs[1,1], evecs[0,1]))
   patch = Ellipse(center_2d, width=2*np.sqrt(evals[1]), height=2*np.sqrt(evals[0]), angle=angle)
   ```
3. ✅ **Done (B1-4, `e258538`)** Implement `plot_rotation_summary(rotvecs: np.ndarray, cone: dict, box: dict, ax=None) -> matplotlib.axes.Axes`: cone drawn as prominent shaded circle at `max_angle`, mean_axis marked; box rendered as secondary annotation.
4. ✅ **Done (B1-4, `e258538`)** Implement `plot_pareto_sensitivity(report: ParetoSensitivityReport, ax=None, top_n=10) -> matplotlib.axes.Axes`. **Locked decision:** first-order-linear-approximation caveat is **hardcoded as a mandatory `ax.annotate()`** on the chart — cannot be suppressed without editing source. Annotation text: `"* First-order linear approximation via small-angle adjoint Jacobian.\n  Nonlinear contributions may differ for high-leverage chains."` at `(0.01, 0.01)` in axes fraction. Justified by Section 1.4 sourcing-discussion use case.
5. ✅ **Done (B1-4, `e258538`)** Implement `generate_frame_report(trial_data, frame_name) -> matplotlib.figure.Figure`. **Layout (locked):** 4×3 GridSpec:
   - Row 0: `hist(dx)` | `hist(dy)` | `hist(dz)`
   - Row 1: `proj(xy)` | `proj(xz)` | `proj(yz)`
   - Row 2: `hist(rx)` | `hist(ry)` | `hist(rz)`
   - Row 3: rotation summary spanning all 3 columns (`gs[3, :]`)
6. ✅ **Done (B1-4, `e258538`)** Implement `generate_sensitivity_report(report: ParetoSensitivityReport) -> matplotlib.figure.Figure`: single-panel figure wrapping `plot_pareto_sensitivity()`.
7. ✅ **Done (B1-4, `e258538`)** `tests/test_reporting.py`: 17 smoke tests. **Critical:** `import matplotlib; matplotlib.use("Agg")` must be the **first import** in the file — before any other matplotlib import — to prevent headless display errors in CI. All 6 public functions confirmed to return valid `Axes`/`Figure` without raising.

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

## 6.11 `persistence/schema.py`

*(Last revised: 2026-06-29 — Claude, apply-allocation + persist IK parameters feature, commit `ca37d91`. Prior: B1-5 implementation, commit `fda4e5c`, 24 tests. Now 30 tests in `tests/test_schema.py`.)*

**Responsibility:** Pydantic models defining the on-disk project data model — the only interface the GUI is permitted to read from or write to (Section 5.3).

**Deliverables:**

- `FrameModel`, `HTMEdgeModel`, `ToleranceSpecModel`/`ToleranceSpec6Model`, `IKConstraintModel`, `SimSettingsModel`, `SavedAnalysisModel`, `ProjectModel`
- Bidirectional conversion functions between these Pydantic models and the live `core`/`sim` objects (`FrameGraph`, `ToleranceSpec6`, etc.)
- `tests/test_schema.py` (new test file)

**Granular Task List:**

1. ✅ **Done (B1-5, `fda4e5c`)** Implement `ToleranceSpecModel`: `distribution: Literal["uniform","normal"]`, `bound: float`, `sigma_level: float = 3.0`, `locked: bool = False`. `@field_validator("bound")` asserts `bound >= 0`.
2. ✅ **Done (B1-5, `fda4e5c`)** Implement `ToleranceSpec6Model`: six `ToleranceSpecModel` fields named `dx, dy, dz, rx, ry, rz`.
3. ✅ **Done (B1-5, `fda4e5c`)** Implement `HTMInputModel` as a Pydantic v2 discriminated union via `kind: Literal[...]` field:
   ```python
   HTMInputModel = Annotated[
       Union[HTMInputXyzEuler, HTMInputMatrix, HTMInputQuaternion, HTMInputScrew],
       Field(discriminator="kind"),
   ]
   ```
   HTMs without `input_representation` (e.g., composed transforms) fall back to `kind="matrix"`. All numpy arrays stored as Python lists (`.tolist()`) in JSON; reconstructed to numpy arrays on load. `HTMInputScrew` has `point_on_axis: list[float] | None = None`.
4. ✅ **Done (B1-5, `fda4e5c`)** Implement `HTMEdgeModel`: `name`, `parent`, `child`, `nominal: HTMInputModel`, `tolerance: ToleranceSpec6Model`.
5. ✅ **Done (B1-5, `fda4e5c`)** Implement `FrameModel`: `name: str`, `metadata: dict = Field(default_factory=dict)`.
6. ✅ **Done (B1-5, `fda4e5c`)** Implement `SimSettingsModel`: `mode: Literal["fk_verification","ik_allocation"]`, `n_trials: int`, `seed: int`. (`default_distribution` and `default_sigma_level` omitted as they are not used by the engine — YAGNI.)
7. ✅ **Done (B1-5, `fda4e5c`)** Implement `SavedAnalysisModel`: `name: str`, `frame_a: str`, `frame_b: str`.
8. ✅ **Done (B1-5, `fda4e5c`)** Implement `ProjectModel`: `schema_version: int = 1`, `frames: list[FrameModel]`, `edges: list[HTMEdgeModel]`, `sim_settings: SimSettingsModel`, `saved_analyses: list[SavedAnalysisModel] = []`.
9. ✅ **Done (B1-5, `fda4e5c`)** Implement `ProjectModel.validate_references()` using Pydantic v2's `@model_validator(mode="after")` — runs after all field validators pass. Checks every edge's `parent`/`child` and every `SavedAnalysisModel`'s `frame_a`/`frame_b` against the declared frame names. Raises `ValidationError` naming the bad reference.
10. ✅ **Done (B1-5, `fda4e5c`)** Implement `project_model_to_frame_graph(project: ProjectModel) -> FrameGraph`. Private `_model_to_htm()` dispatches on `model.kind` to call the appropriate `HTM.from_*()` constructor; `_model_to_tol6()` reconstructs `ToleranceSpec6`.
11. ✅ **Done (B1-5, `fda4e5c`)** Implement `frame_graph_to_project_model(frame_graph, sim_settings, saved_analyses=None) -> ProjectModel`. Private `_htm_to_model()` reads `htm.input_representation` and dispatches on `kind`; falls back to `HTMInputMatrix` if `input_representation is None`. Private `_tol6_to_model()` converts `ToleranceSpec6`. **`TrialData` is NOT persisted** — only project topology is saved; run outputs are ephemeral.
12. ✅ **Done (B1-5, `fda4e5c`)** `tests/test_schema.py`: 24 tests across 4 classes:
    - `TestToleranceSpecModel` (4 tests): valid construction, negative bound raises, zero bound valid, invalid distribution raises
    - `TestProjectModelValidation` (6 tests): valid project, `schema_version` defaults to 1, dangling edge parent/child raises, dangling saved-analysis frame_a/frame_b raises
    - `TestHTMInputModelRoundTrip` (7 tests): all 4 `kind` variants, screw null/non-null `point_on_axis`, `None` input_representation fallback to matrix
    - `TestConversionFunctions` (7 tests): frame/edge names round-trip, HTM matrix close, tolerance all-fields, no `validate_dag()` required, saved analyses, frame metadata
13. ✅ **Done (feature branch, `ca37d91`)** Implement `IKConstraintModel`: `frame_a: str`, `frame_b: str`, `target: ToleranceSpec6Model`. Represents one point-pair IK constraint — the frame pair and per-DoF target tolerances used by the multi-pair allocation engine.
14. ✅ **Done (feature branch, `ca37d91`)** Extend `SimSettingsModel` with IK persistence fields:
    - `ik_constraints: list[IKConstraintModel] = Field(default_factory=list)` — the saved constraint list; defaults to empty so existing project files load without modification.
    - `ik_max_iter: int = 30` — saved maximum damping iterations; defaults to 30 matching the run panel default.
    Both fields have defaults to ensure full backward compatibility: projects saved before this change load and validate without error, and immediately gain a working IK setup when mode is switched.
15. ✅ **Done (feature branch, `ca37d91`)** Extend `ProjectModel.validate_references()` to check IK constraint frame references: for each `IKConstraintModel` in `sim_settings.ik_constraints`, validates both `frame_a` and `frame_b` are declared frames. Raises `ValidationError` with the index, attribute, bad reference, and sorted list of declared frames.
16. ✅ **Done (feature branch, `ca37d91`)** `tests/test_schema.py` additions — 6 new tests in new class `TestIKConstraintModel`:
    - `test_ik_constraint_valid` — valid round-trip
    - `test_sim_settings_ik_max_iter_default` — default is 30
    - `test_sim_settings_ik_constraints_default_empty` — default is `[]`
    - `test_sim_settings_with_constraints_round_trips` — full project round-trip with constraints present
    - `test_project_model_validates_ik_constraint_frame_a` — dangling `frame_a` raises
    - `test_project_model_validates_ik_constraint_frame_b` — dangling `frame_b` raises

**Interfaces:**

- *Depends on:* `core/transforms.py`, `core/tolerance.py`, `core/frame_graph.py` (for the conversion functions in Steps 10–11), `pydantic`.
- *Used by:* `persistence/serializer.py` (Section 6.12), eventually every GUI panel (Section 5.3 — the GUI reads/writes only these models).
- *Public API (conceptual):*
  ```
  ProjectModel, FrameModel, HTMEdgeModel, ToleranceSpec6Model, IKConstraintModel, SimSettingsModel, SavedAnalysisModel
  ProjectModel.validate_references() -> None  # raises on violation
  project_model_to_frame_graph(project: ProjectModel) -> FrameGraph
  frame_graph_to_project_model(frame_graph, sim_settings, saved_analyses) -> ProjectModel
  ```

---

## 6.12 `persistence/serializer.py`

*(Last revised: 2026-06-27 — renamed from `io/` to `persistence/` to eliminate stdlib name collision; root `conftest.py` deleted. Original B1-5 implementation commit `fda4e5c`.)*

**Responsibility:** JSON save/load for `persistence.schema` models, surfacing clear, actionable validation errors before the engine is ever invoked.

**Deliverables:**

- `save_project(project: ProjectModel, path: str) -> None`
- `load_project(path: str) -> ProjectModel`
- Actionable error wrapping around raw Pydantic/JSON errors
- `tests/test_serializer.py` (new test file)

**Granular Task List:**

1. ✅ **Done (B1-5, `fda4e5c`)** Implement `save_project(project: ProjectModel, path: str) -> None`: `project.model_dump_json(indent=2)` written to disk with UTF-8 encoding.
2. ✅ **Done (B1-5, `fda4e5c`)** Implement `load_project(path: str) -> ProjectModel`. Sequential checks: (1) file open — wraps `FileNotFoundError` as `ProjectLoadError`; (2) `ProjectModel.model_validate_json(raw)` — wraps `pydantic.ValidationError` as `ProjectLoadError` (in Pydantic v2, `model_validate_json` wraps JSON decode errors as `ValidationError` internally, so no separate `json.JSONDecodeError` catch is needed); (3) `schema_version` check — `ProjectLoadError` if `!= EXPECTED_SCHEMA_VERSION`. The `validate_references()` model validator runs automatically during Step 2 (it is a `@model_validator(mode="after")`).
3. ✅ **Done (B1-5, `fda4e5c`)** `ProjectLoadError(Exception)`: all load failures wrapped with a human-readable message; original exception always attached as `__cause__` via `raise ... from exc`.
4. ✅ **Done (B1-5, `fda4e5c`)** `schema_version = 1` in `ProjectModel`. `EXPECTED_SCHEMA_VERSION = 1` constant in `persistence/serializer.py`. Error message includes both found and expected version numbers.
5. ✅ **Done (B1-5, `fda4e5c`)** `tests/test_serializer.py`: 16 tests across 2 classes:
   - `TestSaveLoadProject` (8 tests): `model_dump()` equal after round-trip, frame names, edge connectivity, valid JSON output, indent formatting, saved analysis survives, HTM matrix close to 1e-12, tolerance all-fields
   - `TestLoadProjectErrorHandling` (8 tests): file not found raises `ProjectLoadError`, `__cause__` is `FileNotFoundError`, malformed JSON raises, empty file raises, schema_version mismatch raises, mismatch message contains version numbers, dangling edge reference raises, missing required field raises

**Interfaces:**

- *Depends on:* `persistence/schema.py` (`ProjectModel`), `pydantic`, `json`/standard library file I/O.
- *Used by:* eventually `gui/main_window.py` (save/load/new-project actions), and any future CLI.
- *Public API (conceptual):*
  ```
  save_project(project: ProjectModel, path: str) -> None
  load_project(path: str) -> ProjectModel   # raises ProjectLoadError
  ```

---

## 6.13 `gui/main_window.py`

*(Last revised: 2026-06-29 — Claude, `_on_allocation_applied` handler added (feature branch, commit `ca37d91`). Prior: C-1 through C-7 Milestone C implementation, commit `f1ffa2c` (C-6+C-7). Suite: 373 passed. Original planning note (2026-06-23): GUI modules are specified at a coarser grain than core/sim/postprocess/io, consistent with Section 10's rule that GUI work does not begin until the engine is proven.)*

**Responsibility:** Top-level `QMainWindow`; owns the currently-loaded `ProjectModel` in memory, hosts all five GUI panels as named `QDockWidget` instances, owns File menu actions (New/Open/Save/Save As/Exit), cross-panel signal routing, and session persistence via `QSettings`.

**Deliverables:**

- `QMainWindow` subclass with menu bar, status bar, and five tabbed/docked panels
- File → New / Open / Open Recent (with history capped at 5) / Save / Save As / Exit
- Single in-memory `ProjectModel` as source of truth; `_dirty` flag with title-bar asterisk
- `QSettings("TolTransform", "TolTransform")` persistence of window geometry, dock layout, and recent-file list across sessions

**Implemented Architecture:**

- `_project: ProjectModel` — the single live project instance. Never directly passed to engine code; only child panels and `persistence/serializer.py` touch it.
- `_path: str | None` — current file path (`None` for unsaved projects).
- `_dirty: bool` — set True on any `project_changed` signal; cleared on save or new/open.
- `_recent_files: list[str]` — capped at `_MAX_RECENT = 5`, deduplicated on add, missing files removed gracefully on open attempt.
- `_last_run_result: object` — last FK/IK result; used to re-route to panels on demand (not currently displayed; available for future use).
- Dock layout (all `QDockWidget`): **Left** = Graph Editor; **Right** (tabbed/stacked) = Tolerance Editor, Run Panel, Results Viewer, Point-Pair Analysis.
- Each dock has a stable `setObjectName(...)` so `QMainWindow.restoreState()` can re-identify them after a layout change between sessions.

**Key signal wiring (cross-panel integration):**

| Signal source | Signal | Receiver(s) |
|---|---|---|
| `GraphEditorWidget` | `project_changed` | `_on_graph_editor_changed()` → sets dirty, calls `refresh_view()` on Tolerance Editor, Run Panel, Point-Pair Panel |
| `GraphEditorWidget` | `edge_selected(str)` | `ToleranceEditorWidget.set_selected_edge(str)` — clicking an edge in the graph auto-selects it in the tolerance editor |
| `ToleranceEditorWidget` | `project_changed` | `_on_project_changed()` → sets dirty |
| `RunPanelWidget` | `project_changed` | `_on_project_changed()` → sets dirty |
| `RunPanelWidget` | `run_completed(object)` | `_on_run_completed()` → routes to `ResultsViewerWidget.set_result()` AND `PointPairPanelWidget.set_result()` |
| `RunPanelWidget` | `run_failed(str)` | `_on_run_failed()` → no-op (run panel already shows error in its own status label) |
| `PointPairPanelWidget` | `project_changed` | `_on_project_changed()` → sets dirty |
| `ResultsViewerWidget` | `project_changed` | `_on_allocation_applied()` → sets dirty, calls `refresh_view()` on Tolerance Editor so the newly-written bounds appear immediately |

**QSettings persistence (C-7):**

- `_restore_settings()` called at end of `__init__()` (after `_setup_ui()`): restores window geometry, dock state, and recent-file list from the platform store.
- `_save_settings()` called from `closeEvent()` before accepting: saves geometry, state, recent-file list.
- `closeEvent()` checks `_confirm_discard_changes()` first; ignores the close event if the user clicks Cancel in the "save?" dialog.

**Recent Files (C-7):**

- `_rebuild_recent_menu()`: clears and repopulates `Open &Recent` submenu; shows `"(No recent files)"` (disabled) when list is empty; appends `"Clear Recent"` separator+action when non-empty.
- `_add_recent(path)`: prepend → deduplicate → cap at 5 → rebuild menu. Called from `_open_project()` and `_save_project_as()`.
- `_open_recent(path)`: same body as `_open_project()` but without the file dialog; removes the entry and shows `QMessageBox.critical` if the file no longer exists.

**Interfaces:**

- *Depends on:* `persistence/schema.py`, `persistence/serializer.py`, all `gui/*` panel modules, `PySide6.QtCore.QSettings`, `PySide6`.
- *Used by:* the application entry point (`main.py`).
- *Public API (for tests):*
  ```
  MainWindow._project: ProjectModel
  MainWindow._on_graph_editor_changed() -> None
  MainWindow._on_run_completed(result: object) -> None
  MainWindow._on_allocation_applied() -> None
  MainWindow._new_project() -> None
  MainWindow._open_project() -> None
  _empty_project() -> ProjectModel   # module-level helper
  ```

---

## 6.14 `gui/graph_editor/`

*(Last revised: 2026-06-28 — Claude, C-1 Milestone C implementation. **Implemented C-1, commit `67f154e`. 19 tests in `tests/test_gui_graph_editor.py`.** Original planning note (2026-06-23): Coarse-grained per Section 6.13's note.)*

**Responsibility:** Build/edit the Frame graph — add/remove Frames and Edges, enter each edge's nominal transform in any supported format, emit `project_changed` and `edge_selected` signals to MainWindow.

**Deliverables:**

- `GraphEditorWidget(QWidget)` — top-level panel; emits `project_changed` and `edge_selected(str)`.
- `FrameEdgeTree(QTreeWidget)` — two-section tree ("Frames" / "Edges"), with root frames rendered bold blue (`[ROOT]` prefix) and junction frames orange (`[JUNCTION]` prefix). Refreshed from `ProjectModel` via `refresh(project)`.
- `AddFrameDialog(QDialog)` — single name field; validates uniqueness; OK disabled until valid.
- `AddEdgeDialog(QDialog)` — name field, parent/child combos, `HTMEntryWidget`; validates uniqueness and differing parent/child; OK gated on all fields valid.
- `HTMEntryWidget(QWidget)` — multi-format nominal-transform entry with live validation (emits `validation_changed`).

**Implemented Architecture:**

- **`GraphEditorWidget`**: layout = `FrameEdgeTree` (stretch=1) + button row (`+ Frame`, `+ Edge`, `Delete Selected`). All mutations go to `self._project` (a `ProjectModel` reference) directly — no `FrameGraph` constructed here (Section 5.3). After every mutation, calls `refresh_view()` and emits `project_changed`.
- **Deletion safety**: deleting a Frame checks for referencing edges first; if found, shows `QMessageBox.warning` and aborts. Edges may be deleted without checks.
- **`FrameEdgeTree.selected_item_info()`**: returns `(name, kind)` tuple where `kind` is `"frame"` or `"edge"`, or `None` if no selectable item is highlighted. Root node items ("Frames", "Edges") have no `UserRole` data and return `None`.
- **`HTMEntryWidget`**: `QComboBox` format selector + `QStackedWidget` (4 pages: XYZ+Euler, 4×4 Matrix, Quaternion+XYZ, Screw). Each page delegates validation to `core/transforms.py`'s `HTM` constructors via a `_validate()` call on every `valueChanged` signal. `validation_changed(bool)` emitted only when validity state flips. Bottom row of the matrix page is disabled (always `[0,0,0,1]`). `set_htm_input_model(model)` populates fields from an existing `HTMInputModel` for editing round-trips.
- **New edges default to all-locked-zero tolerance** via `_default_tolerance()` — safe to simulate immediately without contributing error.
- All content in `QScrollArea(widgetResizable=True, frameShape=NoFrame)` so docks can be resized to any height.

**Interfaces:**

- *Depends on:* `persistence/schema.py` (reads/writes `ProjectModel`), `core/transforms.py` (live HTM validation in `HTMEntryWidget._validate()`), `PySide6`.
- *Used by:* `gui/main_window.py`.
- *Public API:*
  ```
  GraphEditorWidget.set_project(project: ProjectModel) -> None
  GraphEditorWidget.refresh_view() -> None
  GraphEditorWidget.project_changed: Signal()
  GraphEditorWidget.edge_selected: Signal(str)    # emits edge name on tree click
  HTMEntryWidget.is_valid() -> bool
  HTMEntryWidget.get_htm_input_model() -> HTMInputModel
  HTMEntryWidget.set_htm_input_model(model: HTMInputModel) -> None
  HTMEntryWidget.validation_changed: Signal(bool)
  AddFrameDialog.result_frame() -> FrameModel      # only after Accepted
  AddEdgeDialog.result_edge() -> HTMEdgeModel      # only after Accepted
  FrameEdgeTree.refresh(project: ProjectModel) -> None
  FrameEdgeTree.selected_item_info() -> tuple[str, str] | None
  ```

---

## 6.15 `gui/tolerance_editor/`

*(Last revised: 2026-06-28 — Claude, C-2 Milestone C implementation. **Implemented C-2, commit `33423f9`. 13 tests in `tests/test_gui_tolerance_editor.py`.**)* 

**Responsibility:** Per-edge, per-DoF tolerance entry — distribution, bound, sigma-level, locked flag. Writes directly to the in-memory `ToleranceSpec6Model` on the selected edge; emits `project_changed` on every valid change.

**Deliverables:**

- `ToleranceEditorWidget(QWidget)` — edge selector combo + stacked widget (placeholder / DoF panel) + bulk-apply group.
- `_DofRow` (internal helper, not a `QWidget`) — 6 rows of controls, one per DoF: distribution combo, bound spinbox, σ-level spinbox (enabled only when `"normal"` selected), locked checkbox, per-row error label.
- "Bulk Apply" group: set distribution, bound, σ-level, apply to all 6 DoF of the current edge or to all edges in the project.

**Implemented Architecture:**

- **Layout**: `QScrollArea` wrapping a container `QWidget` containing: edge-selector row, `QStackedWidget` (page 0 = placeholder, page 1 = DoF grid), bulk-apply `QGroupBox`.
- **Edge selector**: `QComboBox` populated from `project.edges`. Uses `activated` signal (not `currentIndexChanged`) so a re-click on the same edge still triggers a reload. Auto-selects the first edge on `set_project()` so the DoF grid appears immediately without requiring a user click.
- **Live write-back**: every field change calls `_on_field_changed()`, which iterates all 6 rows, calls `_DofRow.is_valid()` on each (reusing `core.tolerance.ToleranceSpec` constructor as validator), writes valid specs back via `setattr(edge.tolerance, dof_name, row.get_model())`, then emits `project_changed`. Per-row error labels surface validation failures inline.
- **Load guard**: `self._loading = True` during `_load_selected_edge()` and `_DofRow.load()` to suppress `_on_field_changed()` during batch widget population — prevents spurious dirty-flag sets.
- **σ-level spinbox**: disabled (greyed out) when distribution is `"uniform"`; enabled only when `"normal"`. This is enforced in `_DofRow._on_dist_changed()` and on initial load.
- **Bulk apply to edge**: applies bulk fields to all 6 DoF of the currently selected edge (preserves each DoF's `locked` flag). Calls `_DofRow.load()` to keep the grid in sync.
- **Bulk apply to all edges**: iterates `project.edges`, updates all 6 DoF on every edge, preserves locked flags, reloads the current edge display.
- **`set_selected_edge(name)`**: called by MainWindow when user clicks an edge in the graph editor tree; syncs the edge combo without re-triggering its `activated` signal.
- All content in `QScrollArea(widgetResizable=True, frameShape=NoFrame)`.

**Interfaces:**

- *Depends on:* `persistence/schema.py` (`ProjectModel`, `ToleranceSpecModel`), `core/tolerance.py` (`ToleranceSpec` — validation reuse only), `PySide6`.
- *Used by:* `gui/main_window.py`.
- *Public API:*
  ```
  ToleranceEditorWidget.set_project(project: ProjectModel) -> None
  ToleranceEditorWidget.set_selected_edge(edge_name: str) -> None
  ToleranceEditorWidget.refresh_view() -> None
  ToleranceEditorWidget.project_changed: Signal()
  ```

---

## 6.16 `gui/run_panel/`

*(Last revised: 2026-06-29 — Claude, IK constraint + max_iter persistence added (feature branch, commit `ca37d91`). Prior: C-3 Milestone C implementation, commit `d0457bd`; IK enhancements (max_iter spinbox) merged to main 2026-06-28. Suite: 373 passed.)*

**Responsibility:** Configure and trigger a simulation run. **The one and only place where `persistence.schema` objects are converted into live `core`/`sim` objects** (Section 5.3). Runs the engine on a `QThread` background worker to keep the UI responsive. Writes SimSettings back to `project.sim_settings` on every field change so they persist with the project file — including the full IK constraint list and `ik_max_iter`.

**Deliverables:**

- `RunPanelWidget(QWidget)` — simulation settings + IK target group + Run button + progress bar + status label.
- `_RunWorker(QThread)` — background FK or IK engine execution; emits `finished(object)` or `failed(str)`.

**Implemented Architecture:**

- **Mode selector** (`QComboBox`): "FK Verification" (`"fk_verification"`) and "IK Allocation" (`"ik_allocation"`). IK group (`self._ik_group`) is hidden when FK mode is active.
- **SimSettings write-back**: mode, n_trials, and seed changes call `_on_mode_changed()` / `_on_n_trials_changed()` / `_on_seed_changed()` which update `project.sim_settings.*` in place and emit `project_changed`.
- **Randomize button**: sets seed to `random.randint(0, 2**31 - 1)`.
- **IK target group**: dynamic list of `_ConstraintRowWidget` instances (Frame A/B combos + 6 DoF spinboxes each) + "Add Constraint" / "✕ Remove" buttons + **Max iterations spinbox** (`_max_iter_spin`, default 30, range 1–500). Frame combos refreshed by `refresh_view()` when the graph changes. No method selector — `LoosestAllocation` is always used.
- **IK constraint persistence**: constraint rows and `ik_max_iter` are saved to and loaded from `project.sim_settings` on every change, so they survive project save/load and are available to the run panel immediately when a project is opened.
  - `_load_ik_constraints_from_project()`: clears existing rows, then rebuilds them from `project.sim_settings.ik_constraints`, calling `row.set_frame_pair(frame_a, frame_b)` and `row.set_target_model(target)` with `blockSignals(True)` during programmatic population to suppress spurious `project_changed` emissions.
  - `_save_ik_constraints_to_project()`: iterates current `_constraint_rows`, writes a `IKConstraintModel` per row to `project.sim_settings.ik_constraints`.
  - `_on_max_iter_changed(value)`: writes to `project.sim_settings.ik_max_iter` and emits `project_changed`.
  - `_on_constraints_changed()`: called when any constraint row's frame pair or target spinbox changes; calls `_save_ik_constraints_to_project()` and emits `project_changed`.
  - `_load_sim_settings()` now also calls `_max_iter_spin.setValue(s.ik_max_iter)` and `_load_ik_constraints_from_project()` so IK state is fully restored on `set_project()`.
- **Run flow** (`_on_run_clicked()`):
  1. Guard: no edges → error status, return.
  2. IK guard: A==B or empty → error status, return.
  3. `project_model_to_frame_graph(self._project)` — the only schema→core conversion in the GUI.
  4. Disable Run button, show indeterminate progress bar, set status "Running…".
  5. Construct `_RunWorker` with mode, graph, n_trials, seed, ik_targets list, max_iter.
  6. `worker.start()`.
- **_RunWorker.run()**: dispatches to `MonteCarloFKEngine.run()` (FK) or `AllocationEngine.allocate_multi()` (IK with `n_validate=1000, seed=seed, max_iter=self._max_iter`). Emits `finished(result)` or `failed(str(exc))`.
- **Completion** (`_on_run_finished()`): re-enables Run button, hides progress bar, emits `run_completed(result)`.
- All content in `QScrollArea(widgetResizable=True, frameShape=NoFrame)`.

**Interfaces:**

- *Depends on:* `persistence/schema.py` (`project_model_to_frame_graph`), `core/tolerance.py` (`ToleranceSpec`, `ToleranceSpec6`), `sim/monte_carlo_fk.py` (`MonteCarloFKEngine`), `sim/allocation.py` (`AllocationEngine`, `AllocationResult`), `PySide6.QtCore.QThread`.
- *Used by:* `gui/main_window.py`.
- *Public API:*
  ```
  RunPanelWidget.set_project(project: ProjectModel) -> None
  RunPanelWidget.refresh_view() -> None
  RunPanelWidget.project_changed: Signal()
  RunPanelWidget.run_completed: Signal(object)   # TrialData | AllocationResult
  RunPanelWidget.run_failed: Signal(str)
  RunPanelWidget._constraint_rows: list[_ConstraintRowWidget]  # for integration tests
  _ConstraintRowWidget._frame_a_combo: QComboBox   # for integration tests
  _ConstraintRowWidget._frame_b_combo: QComboBox   # for integration tests
  _ConstraintRowWidget._spins: list[QDoubleSpinBox]  # for integration tests
  ```

---

## 6.17 `gui/results_viewer/`

*(Last revised: 2026-06-29 — Claude, apply-allocation button + `project_changed` signal added (feature branch, commit `ca37d91`). Prior: C-4, commit `63cd2ea`; IK enhancements (Target ± column, method label) merged to main 2026-06-28. Suite: 373 passed.)*

**Responsibility:** Display FK or IK simulation results. After a successful IK allocation run, the user may optionally apply the `corrected_allocation` bounds back to the project's edge tolerance specs via the "Apply Corrected Allocation to Project…" button. Emits `project_changed` after a successful apply so MainWindow can refresh the tolerance editor and set the dirty flag. MainWindow calls `set_result(result, project)` after each run and `clear()` on New/Open.

**Deliverables:**

- `ResultsViewerWidget(QWidget)` — `QStackedWidget` with three pages: placeholder (page 0), FK page (page 1), IK page (page 2).
- `_FigureWindow(QWidget)` — standalone OS window hosting a Matplotlib `FigureCanvasQTAgg`; closed via `plt.close(fig)` on `closeEvent`.
- FK page: frame selector combo → envelope table (6×3 DoF/Min/Max); "Open Frame Report in New Window" button (disabled until FK result arrives); Pareto sensitivity group (Frame A/B combos + "Compute & Open" button).
- IK page: convergence status label (green/orange) with `[Loosest (LP)]` appended; per-pair `QGroupBox` sections (pass/fail title color) each containing a `DoF | Target ± | Min | Max | Pass?` table; corrected allocation table (edge × DoF, locked DoF as `"—"` in gray, tooltip showing baseline bound on damped cells); **"Apply Corrected Allocation to Project…"** button (enabled only when `result.converged`).

**Implemented Architecture:**

- **`_FigureWindow`**: `QWidget` with `Qt.WindowType.Window` flag (opens as a separate OS window, not a child widget). Layout: `QVBoxLayout` → `FigureCanvasQTAgg`. Title set to e.g. `"Frame Report: sensor"`. `resize(1200, 900)`. `closeEvent` calls `plt.close(self._fig)` then `super()`. References held in `self._open_windows: list[_FigureWindow]` on the parent widget to prevent garbage collection.
- **Key design decision — standalone windows instead of embedded canvases**: The original C-4 design embedded `FigureCanvasQTAgg` directly in the dock widget. This caused plots to be invisible at typical dock sizes (Matplotlib canvas minimum size constraints conflict with dock widget height limits). Fixed by switching to `_FigureWindow` standalone windows for all plots. The button "Open Frame Report in New Window" makes this explicit to the user.
- **FK page flow**:
  1. `_show_fk(trial_data)` populates all combos, switches to page 1, immediately calls `_update_fk_display(frame_names[0])`.
  2. `_update_fk_display(name)` calls `frame_envelope_box()` → fills 6×3 table → enables `_view_report_btn`.
  3. `_on_view_report_clicked()` → `generate_frame_report(trial_data, name)` → `_FigureWindow` → `win.show()`.
  4. `_compute_pareto()` → `compute_tolerance_sensitivities()` + `generate_sensitivity_report()` → `_FigureWindow` → `win.show()`.
- **IK page**: `_show_ik(result)` populates corrected-allocation table (iterating `DOF_LABELS`, checking `locked` and computing baseline diff for tooltip), achieved-envelope table with pass/fail coloring, convergence status label. Stores the result in `self._last_ik_result` and sets `_apply_btn.setEnabled(result.converged)`.
- **Apply button flow** (`_on_apply_clicked()`):
  1. Builds a preview text listing each edge and DoF bound that will change.
  2. Shows `QMessageBox.question(...)` confirmation dialog. Aborts if the user clicks Cancel.
  3. For each non-locked DoF on edges that appear in `corrected_allocation`, writes a `ToleranceSpecModel(distribution="uniform", bound=spec.bound, ...)` to the matching `HTMEdgeModel.tolerance` in `self._project`.
  4. Emits `project_changed`.
- **`project_changed = Signal()`**: class-level signal emitted after `_on_apply_clicked()` writes bounds back to the project. Received by `MainWindow._on_allocation_applied()`.
- **`_last_ik_result: AllocationResult | None`**: stored in `__init__`, set in `_show_ik()`, cleared in `clear()`. Guards `_on_apply_clicked()` against being called without a valid result.
- **`clear()`**: calls `_close_windows()` (closes all open `_FigureWindow` instances, ignoring `RuntimeError` if already destroyed), resets `_stack.currentIndex(0)`, sets `_apply_btn.setEnabled(False)` and `_last_ik_result = None`.
- **`_scrollable(inner)`**: module-level helper creating a `QScrollArea(widgetResizable=True, frameShape=NoFrame)` — used to wrap both FK and IK pages so they scroll rather than clipping at small dock heights.

**Interfaces:**

- *Depends on:* `postprocess/stats.py` (`frame_envelope_box`, `compute_tolerance_sensitivities`), `postprocess/reporting.py` (`generate_frame_report`, `generate_sensitivity_report`), `persistence/schema.py` (`project_model_to_frame_graph`), `sim/monte_carlo_fk.py` (`TrialData`), `sim/allocation.py` (`AllocationResult`), `matplotlib`, `PySide6`.
- *Used by:* `gui/main_window.py`.
- *Public API:*
  ```
  ResultsViewerWidget.set_result(result: object, project: ProjectModel) -> None
  ResultsViewerWidget.clear() -> None
  ResultsViewerWidget.project_changed: Signal()        # emitted after apply writes bounds back
  ResultsViewerWidget._stack: QStackedWidget          # for integration tests (currentIndex)
  ResultsViewerWidget._view_report_btn: QPushButton   # for unit tests (isEnabled)
  ResultsViewerWidget._apply_btn: QPushButton         # for unit tests (isEnabled)
  ResultsViewerWidget._per_pair_layout: QVBoxLayout   # for unit tests (count())
  ```

---

## 6.18 `gui/point_pair_panel/`

*(Last revised: 2026-06-28 — Claude, C-5 Milestone C implementation. **Implemented C-5, commit `63cd2ea`. 11 tests in `tests/test_gui_point_pair_panel.py`.**)* 

**Responsibility:** Read-write panel for named `(frame_a, frame_b)` analyses. Saves/deletes `SavedAnalysisModel` entries from `project.saved_analyses`, emits `project_changed` on each mutation. When a FK `TrialData` result is available and the selected pair is connected, computes and shows the relative-pose envelope via `point_pair_envelope_box()`. `AllocationResult` is silently ignored (FK trial data is required for the envelope).

**Deliverables:**

- `PointPairPanelWidget(QWidget)` — Frame Pair group + Saved Analyses group + Relative-Pose Envelope group, all inside a `QScrollArea`.
- Module-level `_are_connected(project, frame_a, frame_b) -> bool` — builds a `networkx.Graph` directly from `project.frames` and `project.edges` (Section 5.3 compliant: no `FrameGraph` constructed, pure schema objects), then calls `nx.has_path()`.

**Implemented Architecture:**

- **Frame Pair group**: Frame A combo + Frame B combo (horizontal row), connectivity warning label (red), name edit, "Save Analysis" button (max width 110), save-error label. On any combo change, `_on_selection_changed()` runs:
  - Empty combo → clear warning, disable Save.
  - A == B → "Frame A and Frame B must be different.", disable Save.
  - Not connected → "Frames '…' and '…' are not connected.", disable Save.
  - Connected → clear warning, enable Save, auto-populate name as `f"{frame_a} → {frame_b}"`, clear save-error label.
- **Saved Analyses group**: `QListWidget` (max height 130) showing `"{name}  ({frame_a} → {frame_b})"` entries; "Delete Selected" button. `currentRowChanged` → `_on_saved_row_changed(row)` which loads the saved analysis's combos using `combo.blockSignals(True)` to avoid re-triggering selection validation during the load. `_on_delete_clicked()` pops the row from `project.saved_analyses` and calls `takeItem(row)` on the list widget directly (no full re-population needed) then emits `project_changed`.
- **Relative-Pose Envelope group**: placeholder label ("Run FK simulation…", gray italic) and 6×3 `QTableWidget` (DoF/Min/Max, max height 210, read-only items). Placeholder and table are mutually exclusive — `_update_envelope()` shows the table only when `_trial_data` is not None and the current pair is connected. `_fill_envelope_table()` formats values to 6 decimal places.
- **`_update_envelope()`** calls `project_model_to_frame_graph()` and `point_pair_envelope_box()` — this is the one place in this panel that constructs core objects, and it is called only at display time (after a run), not during editing.
- **`set_project(project)`**: resets `_trial_data = None`, repopulates combos (preserving current selection by name), repopulates saved-analysis list, calls `_update_envelope()`.
- **`set_result(result)`**: accepts only `TrialData`; stores it in `_trial_data`; calls `_update_envelope()`. `AllocationResult` passed through without effect.
- **`refresh_view()`**: repopulates frame combos after frames are added/removed (called by MainWindow on `_on_graph_editor_changed()`). Calls `_update_envelope()`.
- **`clear()`**: sets `_trial_data = None`, calls `_update_envelope()` (hides table, shows placeholder). Preserves project reference and combos.
- **`_repopulate_frame_combos()`**: uses `blockSignals(True)` on both combos while clearing/refilling, then restores previous selection by name (`findText()`), then defaults `_frame_b_combo` to index 1 (not index 0) to avoid A==B on first load.
- All content in `QScrollArea(widgetResizable=True, frameShape=NoFrame)`.

**Key design decision — Section 5.3 compliance for connectivity check**: `_are_connected()` uses `networkx.Graph` built directly from `project.frames` and `project.edges` (Pydantic models), never calling `project_model_to_frame_graph()` at selection time. This avoids constructing `core.FrameGraph` / `core.HTMEdge` objects during editing, honoring Section 5.3's rule that core objects are only constructed at run time.

**Interfaces:**

- *Depends on:* `persistence/schema.py` (`ProjectModel`, `SavedAnalysisModel`, `project_model_to_frame_graph`), `postprocess/stats.py` (`point_pair_envelope_box`, `DOF_LABELS`), `sim/monte_carlo_fk.py` (`TrialData`), `networkx`, `PySide6`.
- *Used by:* `gui/main_window.py`.
- *Public API:*
  ```
  PointPairPanelWidget.set_project(project: ProjectModel) -> None
  PointPairPanelWidget.set_result(result: object) -> None   # TrialData accepted; AllocationResult ignored
  PointPairPanelWidget.refresh_view() -> None
  PointPairPanelWidget.clear() -> None
  PointPairPanelWidget.project_changed: Signal()
  PointPairPanelWidget._frame_a_combo: QComboBox    # for integration tests
  PointPairPanelWidget._trial_data: TrialData | None  # for integration tests
  ```

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
3. ✅ **Done (D-0, `f3538af`)** Write `examples/allocation_example.py`: 3-frame wafer-inspection gantry chain (`wafer_chuck → stage → sensor_head`, 300 mm z-lift + 500 mm x lateral arm). Three sections: (1) unconstrained FK baseline showing ±mm envelope; (2) achievable inverse allocation to ±0.1 mm / ±1 mrad — prints `converged=True`, `iterations_used`, and a side-by-side baseline vs corrected allocation table with Δ% column; (3) non-convergence scenario with locked imprecise stage_mount (±2 mm, `locked=True`) — prints `converged=False`, `iterations_used=10`, achieved envelope vs target, and three engineering remedies (replace component, restructure, relax target). Text output only — no GUI or figure dependencies.
4. ✅ Each script runs standalone (`python examples/<script>.py`) with no GUI dependency, prints enough intermediate information that a reader can follow the logic without running it. Verified for tasks 1 and 2 above.

**Interfaces:**

- *Depends on:* the full `core`/`sim`/`postprocess` stack (this is integration-level usage, not a module with its own internal logic).
- *Used by:* the project owner directly (onboarding/reference), and indirectly by `tests/` as a source of representative scenarios worth turning into regression tests.

---

## 6.20 `tests/`

*(Last revised: 2026-06-29 — Claude, apply-allocation + persist IK params feature tests added, commit `ca37d91`. **Suite: 373 passed, 0 skipped.** Prior: Milestone C GUI test files added 2026-06-28, **302 passed**. Root `conftest.py` added (not to be confused with `tests/conftest.py`) to work around stdlib `io` name collision — deleted after rename to `persistence/` (2026-06-27). Original A6 note: Global tolerance convention established: `DEFAULT_ATOL=1e-9` (exact composition checks), `SMALL_ANGLE_ATOL=1e-6` (checks where trig residuals at ~1 mrad apply). Three shared fixtures in `tests/conftest.py`: `two_edge_chain`, `three_edge_chain`, `shared_frame_graph`. `test_shared_edge_sampling_consistency` written as required module-level function per Section 9 Item 3. `test_allocation_mc_validation_discrepancy` placeholder added to `test_allocation.py` per Section 9 Item 4 (`pytest.mark.skip` until B-2). README.md has CI entry point.)*

**Responsibility:** The full unit/integration test suite. Houses every hand-calculable validation case described per-module above, plus the dedicated cross-cutting regression tests called out in Section 9.

**Deliverables (test files, consolidated from the per-module task lists above — listed here as the authoritative index):**

**✅ Implemented (Milestone A):**
- `conftest.py` — ✅ A6 (`83b8ee3`): `DEFAULT_ATOL=1e-9`, `SMALL_ANGLE_ATOL=1e-6`, fixtures `two_edge_chain` / `three_edge_chain` / `shared_frame_graph`, helpers `make_tol` / `make_zero_tol` / `make_htm`
- `test_transforms.py` — ✅ A1 (`a31218e`): 26 tests covering `HTM` and `core/conversions.py`
- `test_tolerance.py` — ✅ A2 (`3ac0eed`): 21 tests covering `ToleranceSpec6`, `apply_perturbation_batch`, `core/sampling.py`
- `test_frame_graph.py` — ✅ A3 (`d81645c`): 24 tests covering `FrameGraph`, `adjoint()`, `compute_sensitivity()`
- `test_monte_carlo_fk.py` — ✅ A4 (`744c562`): 18 tests covering `MonteCarloFKEngine`, `TrialData`, per-edge RNG
- `test_stats.py` — ✅ A5 + B1-3 (`019eb34`, `fd8a08b`): 34 tests total — 26 covering Steps 1–7 (A5), 8 covering Steps 8–9 Pareto sensitivity (B1-3)
- `test_integration.py` — ✅ A6 (`83b8ee3`): 15 end-to-end FK → stats tests; includes `test_shared_edge_sampling_consistency` (Section 9 Item 3 standalone named regression)
- `test_allocation.py` — ✅ A6 placeholder (`83b8ee3`): `test_allocation_mc_validation_discrepancy` named and skipped per Section 9 Item 4 (to be implemented in B-2)

**✅ Implemented (Milestone B-1):**
- `test_bounding_shapes.py` — ✅ B1-2 (`ef4c89c`): 25 tests, 6 classes — `TestFitBoundingBox`, `TestFitBoundingSphere`, `TestFitBoundingEllipsoid` (key: coverage=1.0 uniform-scale enclosure guarantee), `TestFitRotationCone`, `TestFitRotationBox`, `TestTypeHardening` (confirms `(N,4,4)` raises `ValueError`)
- `test_reporting.py` — ✅ B1-4 (`e258538`): 17 smoke tests; `import matplotlib; matplotlib.use("Agg")` required as the **first import** (before any other matplotlib import) to prevent headless display errors in CI
- `test_schema.py` — ✅ B1-5 (`fda4e5c`): 24 tests — `TestToleranceSpecModel`, `TestProjectModelValidation` (dangling-reference validator), `TestHTMInputModelRoundTrip` (all 4 kinds + fallback to matrix), `TestConversionFunctions` (full FrameGraph round-trip)
- `test_serializer.py` — ✅ B1-5 (`fda4e5c`): 16 tests — save/load round-trip (8 checks), error handling (file not found, malformed JSON, empty file, schema_version mismatch, dangling reference, missing required field)

**Note on `conftest.py`:** There is one conftest file:
- `tests/conftest.py` — test fixtures (`two_edge_chain`, `three_edge_chain`, `shared_frame_graph`), numerical tolerances, and an **autouse `_qt_main_window_test_guard` fixture** (added 2026-06-29). The guard monkeypatches `MainWindow._save_settings` and `_restore_settings` to no-ops before every test, preventing two failure modes when consecutive tests each create a `MainWindow` in offscreen Qt: (1) `_set_dirty(True)` → `closeEvent` → `QMessageBox.question()` hang in offscreen mode; (2) `QSettings` dock-state save/restore blocking the next window's `__init__`. Each test gets a clean slate; monkeypatch auto-reverts after the test.

The root-level `conftest.py` (stdlib `io` name-collision workaround) was deleted when `io/` was renamed to `persistence/` (2026-06-27).

**✅ Milestone B-1 complete — all tests implemented:**
- `test_physical_validation.py` (B1-6, commit `aacd210`) — 3 named regression tests (RSS, lever-arm, cancellation)

**✅ Milestone B-2 complete:**
- `test_allocation.py` — ✅ B2-3 (`0c9bd9d`): 7 real tests (placeholder removed)

**✅ Implemented (Milestone C — GUI tests):**

All GUI tests use `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` at module top (before any Qt import) and `qtbot.addWidget(widget)` for lifecycle management. A critical headless-Qt invariant applies: **`isHidden()` must be used instead of `isVisible()` for checking widget visibility in offscreen tests.** `QWidget.isVisible()` checks the entire ancestor chain and returns False for all children when the parent window is not shown (which is always the case in headless/offscreen CI). `QWidget.isHidden()` only checks the widget's own explicit hidden flag — the correct choice for testing widget show/hide state without showing the top-level window.

- **`test_gui_graph_editor.py`** — ✅ C-1 (`67f154e`): **19 tests** covering `FrameEdgeTree`, `AddFrameDialog`, `AddEdgeDialog`, `HTMEntryWidget`, and `MainWindow` (4 basic smoke tests). Key tests: tree refreshes on project change; root/junction coloring; duplicate frame name rejected; duplicate edge name rejected; parent==child rejected; invalid matrix entry keeps OK disabled; format selector switches stack page; selected_item_info returns correct (name, kind) tuple; save/load round-trip via MainWindow.

- **`test_gui_tolerance_editor.py`** — ✅ C-2 (`33423f9`): **13 tests** covering `ToleranceEditorWidget`. Key tests: placeholder shown with no edges; first edge auto-selected on set_project; field changes write through to `project.edges[0].tolerance`; sigma spinbox disabled for uniform, enabled for normal; locked checkbox writes through; project_changed emitted on field change; set_selected_edge syncs combo; bulk-apply-to-edge changes all 6 DoF; bulk-apply-to-all-edges changes all edges; set_project clears selection.

- **`test_gui_run_panel.py`** — ✅ C-3 (`d0457bd`) + feature (`ca37d91`): **19 tests** covering `RunPanelWidget`. Original 11 tests: no-edges run shows error status; mode switches to IK shows IK group; IK group hidden in FK mode; frame combos populated on set_project; frame combos refreshed on refresh_view; n_trials and seed write through to sim_settings; sim_settings loaded on set_project; run_completed signal emitted on successful FK run (direct worker call, not background thread); run_failed signal emitted on engine error. **8 new tests (feature, `ca37d91`):** `test_run_panel_loads_ik_constraints_into_rows` — `set_project()` with saved constraints creates the right number of `_ConstraintRowWidget` rows; `test_run_panel_loaded_row_targets_match_model` — target spinboxes match the saved model; `test_run_panel_max_iter_writes_back_to_project` — spinbox change updates `project.sim_settings.ik_max_iter`; `test_run_panel_max_iter_emits_project_changed` — `project_changed` emitted on max_iter change; `test_run_panel_constraint_row_frame_change_writes_back` — changing Frame A combo updates `project.sim_settings.ik_constraints[0].frame_a`; `test_run_panel_add_constraint_row_updates_project` — "Add Constraint" button adds a new row and a new constraint model entry; `test_run_panel_loads_max_iter_from_project` — `set_project()` populates `_max_iter_spin` from saved value. **1 pre-existing broken test fixed:** `test_run_panel_ik_run_emits_allocation_result` — corrected attribute access from `widget._frame_a_combo` (wrong, never existed on `RunPanelWidget`) to `row = widget._constraint_rows[0]; row._frame_a_combo`.

- **`test_gui_results_viewer.py`** — ✅ C-4 (`63cd2ea`) + feature (`ca37d91`): **19 tests** covering `ResultsViewerWidget`. Original 11 tests: starts on page 0 (placeholder); FK result switches to page 1; IK result switches to page 2; FK frame combo populated with frame names; `_view_report_btn` enabled after FK result; envelope table has 6 rows after FK result; IK alloc table row count matches edge count; IK per-pair section populated (≥1 QGroupBox); clear() resets to page 0; FK then clear leaves view_report_btn disabled; pareto combos populated. **1 pre-existing broken test replaced:** `test_results_viewer_ik_achieved_table_has_dof_rows` (referenced `_achieved_table` which no longer exists) → replaced with `test_results_viewer_ik_per_pair_section_populated` (checks `_per_pair_layout.count() >= 1`). **8 new apply-button tests (feature, `ca37d91`):** `test_results_viewer_apply_btn_present`; `test_results_viewer_apply_btn_disabled_initially`; `test_results_viewer_apply_btn_disabled_on_clear`; `test_results_viewer_apply_btn_state_matches_convergence` — enabled for converged result, disabled for non-converged; `test_results_viewer_apply_btn_disabled_for_fk_result` — FK result leaves button disabled; `test_results_viewer_apply_writes_bounds_to_project` — monkeypatches `QMessageBox.question → Ok`, verifies edge tolerance bounds updated; `test_results_viewer_apply_emits_project_changed` — monkeypatches `→ Ok`, verifies `project_changed` signal emitted; `test_results_viewer_apply_cancel_does_not_modify_project` — monkeypatches `→ Cancel`, verifies project untouched.

- **`test_gui_point_pair_panel.py`** — ✅ C-5 (`63cd2ea`): **11 tests** covering `PointPairPanelWidget`. Key tests (all use `isHidden()`/`not isHidden()` for visibility, not `isVisible()`): placeholder shown with no result; combos populated from project; connected frames show no warning; disjoint frames show warning; name auto-populated as `"A → B"`; save adds to `project.saved_analyses`; save emits `project_changed`; duplicate name rejected with error label; selecting saved row loads combos (with `blockSignals` preventing re-trigger); delete removes from project and emits `project_changed`; FK result shows envelope table (6 rows, correct DoF labels).

- **`test_gui_main_window.py`** — ✅ C-6 (`f1ffa2c`) + feature fix (`ca37d91`): **7 cross-panel integration tests** covering `MainWindow` signal routing. Tests call internal handlers directly without starting the background worker. Key tests: FK result → `results_viewer._stack.currentIndex() == 1`; FK result → `point_pair_panel._trial_data is not None`; graph change → `run_panel._constraint_rows[0]._frame_a_combo.count()` updated (fixed from stale `run_panel._frame_a_combo` reference); graph change → `point_pair_panel._frame_a_combo.count()` updated; `_new_project()` → results_viewer back to page 0; `_new_project()` → point_pair_panel._trial_data is None; `_on_run_failed()` → results_viewer stays on page 0.

**Granular Task List (cross-cutting, beyond what's already specified per-module above):**
1. ✅ **Done (A6, `83b8ee3`)** Set up `pytest` configuration with a shared `conftest.py` providing reusable fixtures: `two_edge_chain` (root→B→C, 5 mm + 10 mm translation nominals), `three_edge_chain` (Rz=π/4 + 50 mm + zero-tol), `shared_frame_graph` (shared-base multi-branch). No `pytest.ini` was needed — auto-discovery works from the project root. Module-level helpers `make_tol` / `make_zero_tol` / `make_htm` duplicated inline in `test_integration.py` rather than imported from conftest (conftest.py is not directly importable as a Python module in pytest's default discovery mode without additional path config).
2. ✅ **Done (A6, `83b8ee3`)** `DEFAULT_ATOL = 1e-9` and `SMALL_ANGLE_ATOL = 1e-6` defined in `conftest.py` and mirrored in `test_integration.py`. Convention: `DEFAULT_ATOL` for near-exact floating-point composition (no trig residual); `SMALL_ANGLE_ATOL` for checks where `sin(δθ) vs δθ` at ~1 mrad introduces ~1.7e-10 second-order residual.
3. ✅ **Done (A6, `83b8ee3`)** `test_shared_edge_sampling_consistency` implemented as a module-level function in `tests/test_integration.py` (not inside any class), findable by `pytest -k test_shared_edge_sampling_consistency`. Complementary class-based coverage exists in `test_monte_carlo_fk.py::TestSharedEdgeConsistency`, but the required standalone named function is in `test_integration.py` per this spec requirement.
4. ✅ **Done (A6, `83b8ee3` — placeholder; B2-3 `0c9bd9d` — real implementation)** `test_allocation_mc_validation_discrepancy` in `tests/test_allocation.py`.
5. ✅ **Done (A6, `83b8ee3`)** `README.md` "Running Tests" section: `source .venv/bin/activate && python -m pytest tests/ -q`. Notes that all tests must pass before GUI work begins (Section 10 rule).
6. ✅ **Done (C-1 through C-6, commits `67f154e` → `f1ffa2c`; feature additions `ca37d91`)** Six GUI test files (`test_gui_graph_editor.py`, `test_gui_tolerance_editor.py`, `test_gui_run_panel.py`, `test_gui_results_viewer.py`, `test_gui_point_pair_panel.py`, `test_gui_main_window.py`) totaling **88 tests** (19 + 13 + 19 + 19 + 11 + 7). All run headlessly with `QT_QPA_PLATFORM=offscreen`. **`isHidden()` convention (not `isVisible()`) used for all widget visibility assertions** — documented as a project-wide rule for all future GUI test additions.
7. ✅ **Done (feature, `ca37d91`)** **`tests/conftest.py` autouse fixture** prevents Qt `MainWindow` hang when consecutive tests each create a `MainWindow` in offscreen mode. `_qt_main_window_test_guard` monkeypatches `_save_settings` and `_restore_settings` to no-ops; `QApplication.processEvents()` called after each test to drain the Qt event loop. Without this fix, the full suite hangs indefinitely after the third `MainWindow` instantiation — root cause: `_set_dirty(True)` → `qtbot closeEvent` → `QMessageBox.question()` → blocks forever in offscreen mode.

**Interfaces:**

- *Depends on:* every module in the codebase, by design.
- *Used by:* the project owner and any future Claude Code session, as the primary mechanism for confirming nothing has silently broken between sessions.

**Planned (Milestone D — not yet implemented):**

- **`test_gui_frame_viewer.py`** — D-1: unit tests for `_compute_world_transforms()` helper only (4 tests). `GLViewWidget` rendering is not tested headlessly — OpenGL context initialization is fragile in offscreen CI. Tests cover: single-edge chain gives correct child origin; serial 3-frame chain accumulates translations; 90° rotation edge produces rotated child axes; disconnected components each get their own root identity. Uses the same `isHidden()` / headless-Qt conventions as all other GUI tests.
- **`test_gui_graph_editor.py` additions** — D-2: 3 new tests for `EditEdgeDialog`: dialog opens pre-populated with the existing edge's name and nominal HTM values; accepting the dialog updates `project.edges[idx]` in-place and emits `project_changed`; cancelling the dialog leaves the project unchanged.

---

## 6.21 `gui/frame_viewer/`

*(Planned — Milestone D-1. Not yet implemented. Feature branch: `feature/d1-d2-viewer-edge-edit`.)*

**Responsibility:** Standalone 3D viewer window for real-time frame pose visualization (during graph editing) and post-simulation Monte Carlo point cloud display. Opens on demand from the View menu — not a permanent dock.

**Deliverables:**

- `gui/frame_viewer/__init__.py` — empty package marker
- `gui/frame_viewer/frame_viewer_window.py` — `FrameViewerWindow` class + `_compute_world_transforms()` helper

**Key class:** `FrameViewerWindow(QWidget)` with `Qt.WindowType.Window` flag.

**Public API:**

```python
def update_graph(self, project: ProjectModel) -> None:
    """Recompute nominal world transforms and redraw frame triads + edge lines."""

def set_result(self, trial_data: TrialData) -> None:
    """Store MC result; if in point-cloud mode, redraw scatter."""

def clear(self) -> None:
    """Remove all scene items; forget trial data."""
```

**Internal architecture:**

- **Viewport:** `pyqtgraph.opengl.GLViewWidget` (OpenGL-based, interactive orbit/zoom/pan via mouse drag, handles N > 500,000 points at 60 fps).
- **Two display modes:**
  - *Frames mode* (default): one `GLLinePlotItem` per axis per frame (X=red, Y=green, Z=blue, length auto-scaled to `max(edge_lengths) × 0.1`), plus grey `GLLinePlotItem` lines connecting each frame origin to its parent origin for every edge.
  - *Point cloud mode:* `GLScatterPlotItem` with `pos=xyz` (shape `(N, 3)`) and `color` mapped through the viridis colormap by z-depth. Frame selector `QComboBox` (visible only in this mode) lets the user choose which frame's distribution to show.
- **Mode toggle button:** "Show Point Cloud" / "Show Frames"; disabled when no `TrialData` is available.
- **Point cloud extraction:** `xyz = trial_data.frames[frame_name][:, :3, 3]` — two lines of numpy on data already in memory.

**Key implementation decision — `_compute_world_transforms(project)`:**

```python
def _compute_world_transforms(project: ProjectModel) -> dict[str, np.ndarray]:
    """Chain 4×4 HTM matrices from HTMEdgeModel.nominal values only.

    Section 5.3 compliant: no FrameGraph, no core.* imports.
    Uses Kahn's topological sort algorithm on project.edges.
    Disconnected components each get their own root (identity transform for root frames).
    Returns {frame_name: np.ndarray shape (4,4)} — world transform for each frame.
    """
```

This is the critical Section 5.3 constraint: the viewer must update in real-time while the user is still editing the graph (before any "Run"), so it cannot build a `FrameGraph` (a core object). Instead, it chains `numpy` 4×4 matrices directly from `HTMEdgeModel.nominal` values (which are `HTMInputXyzEuler` or other `HTMInputModel` subtypes). The helper lives in `gui/frame_viewer/frame_viewer_window.py` and depends only on `numpy` and `persistence.schema`.

**Integration in `MainWindow`:**

- New `View` menu (between File and the application menu): `View → 3D Frame Viewer` shortcut `Ctrl+3`.
- `MainWindow._frame_viewer: FrameViewerWindow | None = None` — lazy-created on first toggle.
- `_toggle_frame_viewer()`: creates if None (connecting `destroyed` signal to clear the reference), calls `update_graph(self._project)`, then `show()` + `raise_()`.
- Hooks added (one line each) to existing handlers:
  - `_on_graph_editor_changed()` → `if self._frame_viewer: self._frame_viewer.update_graph(self._project)`
  - `_on_run_completed()` → `if self._frame_viewer: self._frame_viewer.set_result(result)`
  - `_new_project()` → `if self._frame_viewer: self._frame_viewer.clear()`
  - `closeEvent()` → `if self._frame_viewer: self._frame_viewer.close()` (before `event.accept()`)

**New dependency:** `pyqtgraph` — added to `requirements.txt`. No other changes to the dependency list.

**Tests:** `tests/test_gui_frame_viewer.py` — 4 unit tests covering `_compute_world_transforms()` only. `GLViewWidget` rendering is not tested headlessly (OpenGL context is fragile in offscreen CI). See Section 6.20 Planned entries.

**Interfaces:**

- *Depends on:* `persistence.schema` (read-only), `sim.monte_carlo_fk.TrialData` (read-only), `pyqtgraph.opengl`, `numpy`.
- *Used by:* `gui/main_window.py` exclusively.

---

## 6.22 `gui/graph_editor/edit_edge_dialog.py`

*(Planned — Milestone D-2. Not yet implemented. Feature branch: `feature/d1-d2-viewer-edge-edit`.)*

**Responsibility:** In-place editing of an existing `HTMEdgeModel`'s name and nominal transform. Complements `AddEdgeDialog` (which only creates new edges). The parent→child relationship is deliberately not editable — structural changes (reconnecting a joint to a different parent or child) require delete + recreate to maintain DAG integrity.

**Deliverables:**

- `gui/graph_editor/edit_edge_dialog.py` — `EditEdgeDialog` class

**Key class:** `EditEdgeDialog(QDialog)`

**Constructor:**

```python
def __init__(self, edge: HTMEdgeModel, project: ProjectModel,
             parent: QWidget | None = None) -> None:
```

**Layout (mirrors `AddEdgeDialog` structure):**

- `QLabel` showing `"{edge.parent} → {edge.child}"` as read-only grey text with a note "(parent/child cannot be changed here)"
- `QLineEdit` for edge name, pre-populated with `edge.name`
- `HTMEntryWidget` pre-populated via `self._htm_entry.set_htm_input_model(edge.nominal)` — `set_htm_input_model()` already exists on `HTMEntryWidget` (line 91 of `htm_entry_widget.py`); this is not new
- Red error `QLabel` for validation messages
- Standard OK/Cancel `QDialogButtonBox`

**Validation:**

- Name must be non-empty
- If name was changed: must not collide with any other edge's name (original name is excluded from the collision check — renaming to the same name is allowed as a no-op)
- `HTMEntryWidget.is_valid()` must return True (same as `AddEdgeDialog`)

**Result:**

```python
def result_edge(self) -> HTMEdgeModel:
    """Return the edited HTMEdgeModel. Only valid after Accepted."""
```

Returns a new `HTMEdgeModel` with the updated name + nominal, preserving the original `parent`, `child`, and `tolerance` fields unchanged.

**Trigger in `GraphEditorWidget`:**

Two triggers for the edit action:
1. **Double-click:** `self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)` — calls `_on_edit_edge()` only when the double-clicked item is an edge (not a frame).
2. **"Edit Selected" button:** added to the button row alongside the existing "Delete Selected" button; enabled only when an edge (not a frame) is selected in the tree.

**`_on_edit_edge()` method in `GraphEditorWidget`:**

```python
def _on_edit_edge(self) -> None:
    info = self._tree.selected_item_info()
    if not info or info[1] != "edge":
        return
    edge = next(e for e in self._project.edges if e.name == info[0])
    dlg = EditEdgeDialog(edge, self._project, parent=self)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    updated = dlg.result_edge()
    idx = next(i for i, e in enumerate(self._project.edges) if e.name == edge.name)
    self._project.edges[idx] = updated
    self._tree.refresh(self._project)
    self.project_changed.emit()
```

**Interfaces:**

- *Depends on:* `persistence.schema.HTMEdgeModel`, `gui.graph_editor.htm_entry_widget.HTMEntryWidget`.
- *Used by:* `gui/graph_editor/graph_editor_widget.py` exclusively.

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

| # | Status | Task | Module(s) | Est. Hours | Commit |
|---|---|---|---|---|---|
| B1-1 | ✅ Done | `adjoint()`, `compute_sensitivity()`, and `path_edges_between()` — built in A3, cross-validated with finite-difference tests; formulas corrected (see changelog 2026-06-24) | `core/frame_graph.py` | 6–9 | `d81645c` |
| B1-2 | ✅ Done | Bounding shape fitting: box/ellipsoid/sphere (translation), cone (rotation, locked as lead), rotation-vector type-hardening; uniform-scale coverage=1.0 enclosure guarantee | `postprocess/bounding_shapes.py` | 8–12 | `ef4c89c` |
| B1-3 | ✅ Done | Pareto sensitivity engine: `compute_tolerance_sensitivities()`, `ParetoSensitivityReport`, variance-contribution math (uniform/normal), `to_ascii_chart()` | `postprocess/stats.py` | 8–12 | `fd8a08b` |
| B1-4 | ✅ Done | Plotting/reporting: 6 public functions; first-order caveat annotation locked/mandatory on Pareto chart; 4×3 GridSpec frame report | `postprocess/reporting.py` | 8–10 | `e258538` |
| B1-5 | ✅ Done | Pydantic v2 schema + JSON save/load; `HTMInputModel` discriminated union; `ProjectModel` cross-ref validator; `ProjectLoadError`; `schema_version=1`; originally named `io/` (renamed to `persistence/` 2026-06-27) | `persistence/schema.py`, `persistence/serializer.py` | 8–10 | `fda4e5c` |
| B1-6 | ✅ Done | **Physical Validation Test Suite (Section 9.1) — gating deliverable:** Linear Stack-Up (RSS) Benchmark, Sine-Bar Lever Arm Benchmark, Common-Ancestor Cancellation Benchmark, each as a dedicated named regression test | `tests/` | 8–12 | `aacd210` |
| B1-7 | ✅ Done | Example scripts demonstrating new forward-analysis capabilities (sensitivity Pareto breakdown, component-selection and mitigation-verification use cases from Section 1.4) | `examples/` | 3–5 | `04b5b05` |

**Milestone B-1 ✅ COMPLETE — all 7 tasks done. Suite: 223 passed, 1 skipped. B-2 is unblocked.**

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

**Goal:** The full PySide6 desktop application on top of the proven engine (Milestones A, B-1, B-2). No engine logic is implemented here — this phase only builds the GUI layer described in Sections 6.13–6.18, talking exclusively to the `persistence.schema` data model (Section 5.3).

**Target: ~55–77 hours.**

| # | Status | Task | Module(s) | Est. Hours | Commit |
|---|---|---|---|---|---|
| C1 | ✅ Done | GUI: graph/chain editor (`FrameEdgeTree`, `AddFrameDialog`, `AddEdgeDialog`, `HTMEntryWidget` with 4 formats + live validation; root/junction visual coding; `GraphEditorWidget` shell) | `gui/graph_editor/` | 10–14 | `67f154e` |
| C2 | ✅ Done | GUI: tolerance editor (per-DoF distribution/bound/sigma-level/locked grid; σ-level spinbox conditional enable; bulk-apply-to-edge and bulk-apply-to-all; auto-select first edge; `set_selected_edge` cross-panel wiring) | `gui/tolerance_editor/` | 6–8 | `33423f9` |
| C3 | ✅ Done | GUI: run panel (FK/IK mode selector; N trials/seed/randomize; IK target frame pair + 6 DoF target bound entry; `_RunWorker(QThread)` background execution; sim_settings write-back; status label + indeterminate progress bar) | `gui/run_panel/` | 4–6 | `d0457bd` |
| C4 | ✅ Done | GUI: results viewer (3-page `QStackedWidget`; FK page with frame combo + envelope table + "Open Frame Report" button + Pareto group; IK page with convergence label + allocation table + achieved envelope; Matplotlib figures in `_FigureWindow` standalone windows to avoid dock-height sizing conflicts) | `gui/results_viewer/` | 10–14 | `63cd2ea` |
| C5 | ✅ Done | GUI: point-pair analysis panel (Frame A/B combos; connectivity check via `networkx.Graph` on schema objects; name auto-populate; save/delete `SavedAnalysisModel`; relative-pose envelope table from `point_pair_envelope_box()`; Section 5.3 compliant — no `FrameGraph` at selection time) | `gui/point_pair_panel/` | 5–8 | `63cd2ea` |
| C6 | ✅ Done | Cross-panel integration tests: 7 tests in `test_gui_main_window.py` verifying FK result → results_viewer page, FK result → point_pair_panel trial data, graph change → run_panel / point_pair_panel combo refresh, new_project → viewer reset, run_failure → viewer unchanged | `tests/` | 8–12 | `f1ffa2c` |
| C7 | ✅ Done | Window/dock state persistence (`QSettings` save/restore geometry + dock state); Recent Files menu (capped at 5, dedup, missing-file graceful removal, "Clear Recent" action); `closeEvent` save-before-close dialog; title-bar dirty asterisk | `gui/main_window.py` | 12–15 | `f1ffa2c` |

**✅ Milestone C COMPLETE — all 7 tasks done and pushed to `origin/main`. Suite: 302 passed, 0 skipped.**

**Milestone C exit criteria:** A double-clickable (or `python main.py`-launchable) desktop application where a user with no Python experience could define a system, set tolerances, run both modes, and extract bounding-shape and sensitivity decisions — backed by the same validated engine from Milestones A, B-1, and B-2.

## 7.5 Milestone D — 3D Viewer & Edge Editing

**Goal:** Two focused GUI enhancements that significantly improve the graph-authoring and inspection workflow: a live 3D frame viewer (real-time during editing, point-cloud mode after simulation) and in-place edge editing (eliminating the delete-and-recreate workaround for nominal-transform corrections).

**Feature branch:** `feature/d1-d2-viewer-edge-edit`

**New dependency:** `pyqtgraph` (D-1 only) — added to `requirements.txt`.

**Target: ~15–25 hours.**

| # | Status | Task | Module(s) | Est. Hours |
|---|---|---|---|---|
| D-1 | ⬜ Planned | **Live 3D Frame Viewer** — `FrameViewerWindow(QWidget, Qt.Window)` opened via `View → 3D Frame Viewer` (Ctrl+3); two modes: *Frames* (pyqtgraph `GLViewWidget` with coordinate triads + edge lines, real-time update on `project_changed`) and *Point Cloud* (`GLScatterPlotItem` from `trial_data.frames[frame][:, :3, 3]`, viridis depth color); Section 5.3-compliant world-transform helper `_compute_world_transforms()` chains 4×4 numpy matrices from schema only; lazy window creation stored in `MainWindow._frame_viewer`; 4 unit tests for the helper (no headless OpenGL rendering tests) | `gui/frame_viewer/`, `gui/main_window.py`, `tests/test_gui_frame_viewer.py` | 10–16 |
| D-2 | ⬜ Planned | **Edit Edge Dialog** — `EditEdgeDialog(QDialog)` pre-populated via `HTMEntryWidget.set_htm_input_model(edge.nominal)` (method already exists); parent/child shown read-only; name + nominal HTM editable; duplicate-name check excludes original name; double-click on edge row in `FrameEdgeTree` OR dedicated "Edit Selected" button triggers `GraphEditorWidget._on_edit_edge()` which replaces `project.edges[idx]` in-place + emits `project_changed`; 3 new tests (pre-population, accept updates, cancel no-ops) | `gui/graph_editor/edit_edge_dialog.py`, `gui/graph_editor/graph_editor_widget.py`, `tests/test_gui_graph_editor.py` | 5–9 |

**Milestone D exit criteria:** (1) Opening the 3D viewer window during graph editing shows correct frame coordinate systems in world space, updating live as edges are added or modified. (2) After running FK simulation, the "Show Point Cloud" toggle displays the MC scatter for the selected output frame. (3) Double-clicking any edge in the graph editor tree opens the edit dialog pre-populated; accepting the dialog updates the project and marks it dirty without losing tolerance assignments.

## 7.6 Milestone E-1 — Standalone Packaging

**Goal:** Package TolTransform as a double-clickable desktop application for Mac and Windows — no Python installation required on the target machine. Provide a GitHub Actions CI pipeline that builds and releases both platforms automatically on version tag pushes.

**Tasks completed (2026-06-30):**

| Task | Description | Status |
|---|---|---|
| E1-1 | `toltransform.spec` — PyInstaller one-folder spec; collects pyqtgraph GLSL shaders, matplotlib data, pytransform3d data; hidden imports for PySide6 OpenGL + pydantic internals; Mac BUNDLE block with `Info.plist` | ✅ |
| E1-2 | `packaging/rthook_opengl.py` — runtime hook that sets `QT_OPENGL=desktop` on Windows before Qt loads (prevents ANGLE from intercepting OpenGL) | ✅ |
| E1-3 | `main.py` — set `QSurfaceFormat` CompatibilityProfile + depth buffer before `QApplication` construction | ✅ |
| E1-4 | `assets/` — app icons: `icon.icns` (Mac, 2.1 MB generated via `sips`/`iconutil`), `icon.ico` (Windows, generated via Pillow), `toltransform_icon.png` (1024×1024 source PNG) | ✅ |
| E1-5 | `.github/workflows/build.yml` — matrix CI (macOS-latest + windows-latest); triggered by `v*.*.*` tag; zips artifact; auto-attaches to GitHub Release via `gh release upload` | ✅ |
| E1-6 | Windows HiDPI 3D viewer bug: `_HiDPIGLView` subclass in `gui/frame_viewer/frame_viewer_window.py`; `_reset_camera()` pan-center fix | ✅ |
| E1-7 | `requirements-dev.txt` — added `pyinstaller>=6.0,<7` | ✅ |
| E1-8 | GitHub Release v1.0.0 created with `TolTransform-macOS.zip` and `TolTransform-Windows.zip` attached | ✅ |

**✅ Milestone E-1 COMPLETE — v1.0.0 released 2026-06-30. Mac and Windows builds available at https://github.com/JosephLitjens/toltransform/releases/tag/v1.0.0**

**Milestone E-1 exit criteria:** A user with no Python installation can download a zip from GitHub Releases, unzip it, and launch TolTransform on either Mac or Windows. All GUI features — including the 3D frame viewer — work correctly on both platforms.

## 7.7 Deferred / Explicitly Out of Scope for V1.0

Recorded here so they are not forgotten, but also not accidentally started early:

- Convex-hull (non-axis-aligned) bounding shapes for translation point clouds (v1.x nice-to-have)
- Fast analytical (non-Monte-Carlo) linear worst-case estimate as a quick sanity-check shortcut (v1.x nice-to-have)
- Alternative allocation objectives (e.g., cost-weighted, min-sensitivity) — `LoosestAllocation` is the sole current objective; a cost-weighted variant would be the most natural addition
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
| Inverse allocation objective | `LoosestAllocation` — log-sum NLP, per-DoF maximization. `EqualAllocation` and `RSSAllocation` removed 2026-06-28. |
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
| 2026-06-25 | Claude (B1-2 implementation) | **Implemented `postprocess/bounding_shapes.py`** (commit `ef4c89c`). 5 public functions + 2 private helpers. **Key decisions:** (1) `fit_bounding_sphere`: uses centroid + max distance (conservative, always-correct bound — not the geometric minimum enclosing sphere, which would add implementation complexity for minimal benefit in an error-budgeting tool). (2) `fit_bounding_ellipsoid` with `coverage=1.0`: per-axis independent max projection does NOT guarantee enclosure when points have off-axis components (a point far along two PCA axes simultaneously can escape). Fix: **uniform-scale approach** — scale all PCA axes by the same factor `sqrt(max_Mahalanobis²)`, which guarantees enclosure by construction. This invariant is explicitly tested. For `coverage<1.0`: `chi2.ppf(coverage, df=3)` scaling. (3) `_check_rotvec_shape()` private helper raises `ValueError` with a clear message on `(N,4,4)` input, enforcing the `(N,3)` rotation-vector contract; `fit_rotation_cone` and `fit_rotation_box` both call it at entry. (4) `fit_rotation_cone`: `mean_axis` falls back to `[0,0,1]` when all rotvec magnitudes are zero, avoiding NaN. (5) `fit_rotation_box` delegates to `fit_bounding_box()` after the shape check — no separate implementation. 25 tests in `tests/test_bounding_shapes.py`, 6 test classes. |
| 2026-06-25 | Claude (B1-3 implementation) | **Implemented Pareto sensitivity engine** (commit `fd8a08b`) — Steps 8–9 of `postprocess/stats.py`. **Key decisions:** (1) `trial_data` deliberately omitted from `compute_tolerance_sensitivities(frame_graph, frame_a, frame_b)` signature — tolerance specs (distribution, bound, sigma_level) live on `FrameGraph.edges[name].tolerance`, not on `TrialData`; `TrialData` is run outputs and has no role here. (2) Variance formula (locked): `uniform → b²/3`, `normal → (b/sigma_level)²`. Consistent with the RSS benchmark (Section 9.1.1). (3) Percentage contributions summed **unweighted across all 6 output DoF** — total output variance is the scalar sum of all 6 DoF variance contributions from all edges. (4) First-order-linear-approximation caveat required as **on-chart annotation** in `postprocess/reporting.py` — documented in function docstring and enforced as a hardcoded `ax.annotate()` in reporting.py (locked, cannot be suppressed). `ParetoSensitivityReport` dataclass: `ranked_contributions: list[tuple[str, str, float]]`, `total_variance: float`, `to_ascii_chart(top_n=10) -> str`. 8 new tests in `tests/test_stats.py`. |
| 2026-06-25 | Claude (B1-4 implementation) | **Implemented `postprocess/reporting.py`** (commit `e258538`). 6 public functions + 3 private helpers (`_ensure_ax`, `_maybe_subsample` with `_SCATTER_MAX_POINTS = 2000`, `_bounding_shape_to_ellipse`). **Key decisions:** (1) First-order caveat annotation on the Pareto chart is **locked/mandatory** — hardcoded as `ax.annotate(...)` with `xycoords="axes fraction"` at `(0.01, 0.01)`. Cannot be suppressed without editing source. Justified by Section 1.4 sourcing-discussion use case (chart is shared standalone; caveat must travel with it). (2) `generate_frame_report` uses a 4×3 `GridSpec`: Row 0 = translation histograms, Row 1 = translation projections (xy/xz/yz), Row 2 = rotation histograms, Row 3 = rotation summary spanning all columns (`gs[3, :]`). (3) `plot_translation_projection` 2D ellipsoid: covariance-slice approach — extract the 2×2 submatrix of the 3D covariance for the requested plane, run `np.linalg.eigh`, render as `matplotlib.patches.Ellipse`. (4) All 6 public functions return `Axes` or `Figure`; callers own `.show()`/`.savefig()`. This module never calls either. 17 smoke tests in `tests/test_reporting.py`; `import matplotlib; matplotlib.use("Agg")` must be the first import. |
| 2026-06-25 | Claude (B1-5 implementation) | **Implemented `persistence/schema.py` and `persistence/serializer.py`** (commit `fda4e5c`). **Key decisions:** (1) `HTMInputModel` discriminated union (4 variants: `xyz_euler`, `matrix`, `quaternion`, `screw`; `kind` field as Pydantic discriminator). HTMs without `input_representation` (e.g., composed transforms) fall back to `kind="matrix"`. (2) All numpy arrays stored as Python lists in JSON (`.tolist()`); reconstructed to numpy arrays on load. (3) `TrialData` NOT persisted — only project topology saved; run outputs are ephemeral by design. (4) `schema_version = 1` checked on load; mismatch raises `ProjectLoadError` with a friendly message including both found and expected versions. (5) `@model_validator(mode="after")` (Pydantic v2 idiom) used for cross-reference validation — runs after all field-level validators. (6) **Stdlib name collision (now resolved):** package was originally named `io/`, which collided with Python's frozen stdlib `io` module; a root-level `conftest.py` importlib workaround was used. The package was renamed to `persistence/` on 2026-06-27 (see that changelog entry), eliminating the collision and the workaround. 24 tests in `tests/test_schema.py`, 16 tests in `tests/test_serializer.py`. Suite: **220 passed, 1 skipped.** |
| 2026-06-26 | Claude (B1-6 implementation) | **Implemented `tests/test_physical_validation.py`** (commit `aacd210`) — 3 module-level named regression tests constituting the Milestone B-1 gating deliverable (Section 9.1). (1) `test_rss_linear_stack_up`: 5-link normal-distribution translation chain, independent edges; validates output variance equals classical RSS sum within the statistically-derived 5-SE sampling bound `5 × σ² × √(2/N)`, not an arbitrary tolerance. (2) `test_sine_bar_lever_arm`: single rz-only pivot (1 mrad) + 100 mm arm; validates `var(dy) = L² × var(rz)` within 1% rtol — the lever-arm cross-coupling that motivates B-2's damping loop. (3) `test_common_ancestor_cancellation`: 1 m shared structural tolerance + 1 mm per-instrument tolerances; validates relative camera↔sample envelope stays < 3 mm (cancellation), while absolute camera envelope > 500 mm (shared error is genuinely large). Private helpers `_normal_tol`, `_uniform_tol`, `_zero_tol`, `_rz_only_tol` defined inline (not imported from conftest — conftest is not a regular importable module in pytest's default discovery mode). Suite: **223 passed, 1 skipped.** |
| 2026-06-26 | Claude (B1-7 implementation) | **Implemented `examples/pareto_sensitivity_example.py`** (commit `04b5b05`) — standalone example script demonstrating all three Section 1.4 engineering-decision use cases using the B1 capabilities for the first time. Scenario: 4-edge serial inspection robot arm (`base → shoulder → elbow → wrist → probe_tip`, 500 mm reach). Section 1: baseline MC run (50,000 trials, seed=42) + envelope and percentile tables for `probe_tip`. Section 2: Sensitivity Pinpointing — `compute_tolerance_sensitivities(fg, "base", "probe_tip")` produces Pareto breakdown; `shoulder` edge (cheap universal joint, uniform ±3 mrad) dominates at ~32% per angular DoF via lever-arm amplification. Section 3: Component Selection — rebuild with `shoulder` upgraded to ±1 mrad, re-run MC, compare Pareto rankings; shows probe dx range reduces 56%, `elbow` identified as next bottleneck (8.4%). Section 4: Reporting — saves `probe_tip_frame_report.png`, `probe_tip_pareto_baseline.png`, and `probe_tip_pareto_upgraded.png` to `examples/output/` using `generate_frame_report` / `generate_sensitivity_report`. `matplotlib.use("Agg")` must be first matplotlib import (headless, no display). **Milestone B-1 fully complete. B-2 unblocked.** Suite: 223 passed, 1 skipped. |
| 2026-06-26 | Project Owner (decision), Claude (B2-1 implementation) | **Resolved EqualAllocation multi-DoF reconciliation (Section 6.7 Step 3):** when multiple active target DoF constraints are present, `EqualAllocation.solve()` uses the most-restrictive scale factor: `s = min_k(B_k / Σ_{free (i,j)} |J[k, 6*i+j]|)` — Option A. Rationale: guarantees all active constraints are satisfied simultaneously at the linear-approximation step, minimising the number of damping-loop iterations needed. Option B (least-squares across DoF) was rejected because it may violate individual constraints at the linear step, increasing reliance on the damping loop's fixed `max_iter=10` cap. **Implemented B2-1:** `AllocationObjective` (ABC), `EqualAllocation`, and `AllocationEngine.solve()` in `sim/allocation.py`. Free-edge definition: an edge where not all 6 DoF have `locked=True`. Within free edges, only non-locked DoF columns of the Jacobian contribute to the scale-factor computation; locked DoF on free edges keep their existing specs unchanged. Raises `ValueError("No free edges to allocate — all path edges are locked")` when the path has no free edges. Suite: 223 passed, 1 skipped. |
| 2026-06-26 | Claude (B2-2 implementation) | **Implemented `AllocationEngine.allocate()`, `AllocationEngine.validate()`, `AllocationResult`, `ValidationReport`, `_copy_frame_graph_with_tolerances()`, and `_damp_angular()` in `sim/allocation.py`** (commit `b005eaf`). Key implementation details: (1) `_copy_frame_graph_with_tolerances(fg, new_tolerances)` builds a fresh `FrameGraph` with the same frames and edges but swaps in proposed tolerances for any edge whose name appears in `new_tolerances`; used by `validate()` so the original graph is never mutated. (2) `_damp_angular(allocation, gamma)` scales ONLY angular DoF (indices 3,4,5 = rx,ry,rz) by `gamma` per call — translation bounds (dx,dy,dz) are intentionally left unchanged because the failure mode being corrected is angular-to-positional lever-arm coupling, not translational error. Locked DoF are not damped. (3) `validate()` runs `MonteCarloFKEngine.run()` on a copied graph, then calls `point_pair_envelope_box()` for the relative pose between the two measurement frames; per-DoF pass/fail determined by `max(|achieved["min"]|, |achieved["max"]|) ≤ target_bound`. (4) `allocate()` runs `solve()` first, validates the baseline, and immediately returns with `iterations_used=0` if it passes. If it fails, a `deepcopy` of the baseline is damped iteratively up to `max_iter=10` times; the baseline dict is never modified so `AllocationResult.baseline_linear_allocation` always reflects the original linear solution. Non-convergence returns `converged=False` with status message exactly `"Allocation could not converge to target budget"`. Locked constants: `gamma=0.9`, `max_iter=10`, `n_validate=1000`, `seed=42`. Suite: 223 passed, 1 skipped (B2-3 placeholder still present). |
| 2026-06-27 | Project Owner (request), Claude (implementation) | **Renamed `io/` package to `persistence/`** (commit TBD). Motivation: `io` is a Python frozen stdlib module; `FrozenImporter` in `sys.meta_path` sits ahead of `PathFinder`, so `from io.schema import X` always resolved to the stdlib, not our package. The previous fix was a root-level `conftest.py` (39 lines) using `importlib.util.spec_from_file_location` to pre-load our modules into `sys.modules` before pytest collection. The rename eliminates the collision entirely. Changes: (1) `io/` directory renamed to `persistence/`; (2) one intra-package import updated in `persistence/serializer.py` (`from io.schema` → `from persistence.schema`); (3) all `from io.schema`/`from io.serializer` imports updated in `tests/test_schema.py` and `tests/test_serializer.py`; (4) root `conftest.py` deleted; (5) `docs/design_spec.md` and `CLAUDE.md` updated throughout. Suite remains **230 passed, 0 skipped** — no test count change. |
| 2026-06-26 | Claude (B2-3 implementation) | **Implemented 7 real tests in `tests/test_allocation.py`** (commit `0c9bd9d`), replacing the `@pytest.mark.skip` placeholder. **Test geometry for lever-arm tests (4, 5, 6, 7):** three-frame chain `base→pivot` (identity nominal, rz FREE, all other DoF locked with bound=0) → `pivot→arm` (Tx(1 m) locked, bound=0) → `arm→exit` (Ry(π/2) locked, bound=0). This geometry was chosen because: the linear Jacobian block for `pivot_edge` is `Ad_{T_{base→pivot,nom}} = Ad_I = I₆`, giving column 5 (rz input) as `[0,0,0,0,0,1]` — zero dy coupling at first order. The MC, however, composes the perturbed pivot rotation through the locked Tx(L) arm: `dy = L·sin(δrz) ≈ L·δrz` (first-order, missed entirely by the Jacobian). With L=1 m and target B_rz=0.10, EqualAllocation assigns rz-bound=0.10; MC produces dy_mc≈0.10 > B_dy=0.05. Damping converges after k=7 iterations (0.9⁷·0.10≈0.0478 ≤ 0.05). For the non-convergence test, B_dy=0.001 requires k≥44 iterations, far beyond max_iter=10. The downstream Ry(π/2) on `arm→exit` also swings the rz perturbation into an rx output error (confirmed: `rx ≈ δrz` via conjugation `Ry(-π/2)·Rz(δrz)·Ry(π/2) ≈ Rx(δrz)`), verifying correct behaviour with non-trivial downstream nominal rotations. **Tests:** (1) `test_equal_allocation_sanity` — 3-edge identity chain, all free, all edges get same bound per DoF. (2) `test_locked_edge_excluded` — locked middle edge absent from result dict. (3) `test_all_edges_locked_raises` — ValueError with "No free edges" message. (4) `test_allocation_mc_validation_discrepancy` — lever-arm chain, `solve()`→`validate()` returns `passed=False`, `per_dof_pass["dy"]=False`. (5) `test_damping_loop_convergence` — `allocate()` returns `converged=True`, `iterations_used≥1`, corrected angular bounds < baseline angular bounds. (6) `test_damping_loop_nonconvergence` — `converged=False`, `status_message=="Allocation could not converge to target budget"`, `iterations_used==10`. (7) `test_allocation_result_preserves_both_allocations` — baseline and corrected are distinct objects, baseline angular bounds > corrected. **Milestone B-2 fully complete.** Suite: **230 passed, 0 skipped.** |
| 2026-06-28 | Claude (C-1 implementation) | **Implemented `gui/graph_editor/`** (commit `67f154e`). Five source files: `__init__.py`, `graph_editor_widget.py`, `frame_edge_tree.py`, `add_frame_dialog.py`, `add_edge_dialog.py`, `htm_entry_widget.py`. **Key decisions:** (1) `FrameEdgeTree` uses `QTreeWidget` with two fixed top-level items ("Frames", "Edges"); items store `name` in `Qt.ItemDataRole.UserRole` and the root/section items store no data — `selected_item_info()` returns `None` for them. Root frames rendered bold blue (`[ROOT] name`); junction frames (2+ incoming edges) orange (`[JUNCTION] name`). (2) `HTMEntryWidget` uses a `QComboBox` format selector + `QStackedWidget` (4 pages: XYZ+Euler, 4×4 Matrix, Quaternion+XYZ, Screw). Every `valueChanged` signal calls `_validate()` which calls the appropriate `HTM.from_*()` constructor and catches the resulting `ValueError`; sets `_valid` and emits `validation_changed(bool)` only when validity flips. The matrix page's bottom row spinboxes are disabled (always `[0,0,0,1]`). (3) `AddEdgeDialog` gates OK on `_validate_fields()` returning `""` (no error). Edge names validated for uniqueness across `project.edges`. Parent == Child validation runs same-frame check. (4) New edges created via `_default_tolerance()` — all 6 DoF set to `uniform, bound=0, locked=True` so they simulate safely from day 1 without contributing error. (5) All content in `QScrollArea(widgetResizable=True, frameShape=NoFrame)`. **Section 5.3 compliance:** `GraphEditorWidget` mutates `self._project` (a `ProjectModel`) directly; never constructs a `FrameGraph`. 19 tests in `tests/test_gui_graph_editor.py`. Suite: **249 passed, 0 skipped.** |
| 2026-06-28 | Claude (C-2 implementation) | **Implemented `gui/tolerance_editor/tolerance_editor_widget.py`** (commit `33423f9`). **Key decisions:** (1) `_DofRow` is a helper class, not a `QWidget` — widgets are added directly to the parent `QGridLayout`, avoiding an extra layout nesting level. Fields: distribution `QComboBox`, bound `QDoubleSpinBox`, σ-level `QDoubleSpinBox` (disabled/grayed when `uniform`), locked `QCheckBox`, error `QLabel`. (2) Edge selector uses `.activated` (not `.currentIndexChanged`) so re-clicking the same edge re-loads it — important if the project was mutated externally. (3) Load guard: `self._loading = True` during `_load_selected_edge()` prevents `_on_field_changed()` from emitting `project_changed` during widget population. (4) `_on_field_changed()` calls `_DofRow.is_valid()` on each row (which constructs a `ToleranceSpec` to validate) and only writes back rows that pass; per-row error labels show inline. (5) Bulk apply preserves existing `locked` flags — distribution, bound, and sigma change but locked state is user-controlled. (6) Auto-selects the first edge in `_refresh_edge_combo()` so the DoF panel appears immediately on `set_project()`. 13 tests in `tests/test_gui_tolerance_editor.py`. Suite: **262 passed, 0 skipped.** |
| 2026-06-28 | Claude (C-3 implementation) | **Implemented `gui/run_panel/run_panel_widget.py`** (commit `d0457bd`). **Key decisions:** (1) `_RunWorker(QThread)` owns `run()`: dispatches to `MonteCarloFKEngine.run()` (FK) or `AllocationEngine.allocate()` (IK with `n_validate=1000, seed=seed`). Emits `finished(object)` or `failed(str)`. `run()` never touches the `ProjectModel` — it receives a `FrameGraph` already constructed by the main thread immediately before `.start()`. (2) `project_model_to_frame_graph()` is called in `_on_run_clicked()` on the main thread, immediately before the worker is started. If conversion fails (e.g., cycle in graph), an error status is shown and the run is aborted before a worker is ever launched. (3) Worker is connected to its own `deleteLater()` on both `finished` and `failed` to prevent dangling worker references in tests. (4) IK target bounds default to `0.001` per DoF with step `0.0001`. (5) SimSettings written back on every field change so mode/n_trials/seed persist in the project file. (6) All content in `QScrollArea(widgetResizable=True, frameShape=NoFrame)`. **This is the canonical single location where Section 5.3's `persistence.schema` → `core/sim` conversion happens in the GUI.** 11 tests in `tests/test_gui_run_panel.py`. Suite: **273 passed, 0 skipped.** |
| 2026-06-28 | Claude (C-4 implementation, including plot visibility fix) | **Implemented `gui/results_viewer/results_viewer_widget.py`** (commit `63cd2ea`, along with C-5). **Key decisions:** (1) **Standalone `_FigureWindow` instead of embedded canvas** (critical fix to original plan): the original spec (Section 6.17) called for embedding `FigureCanvasQTAgg` directly in the dock widget. This was implemented and found to produce invisible/inaccessible plots at typical dock heights — Matplotlib's canvas minimum size constraints prevent it from rendering within a dock that is shorter than the canvas minimum. Fix: `_FigureWindow(QWidget)` with `Qt.WindowType.Window` flag opens as a separate OS window sized 1200×900. `closeEvent` calls `plt.close(self._fig)`. References held in `self._open_windows: list[_FigureWindow]` to prevent garbage collection. (2) `_scrollable(inner)` module-level helper wraps inner pages in `QScrollArea(widgetResizable=True, frameShape=NoFrame)` — standard dock-size-independence pattern. (3) IK alloc table tooltips: when a corrected bound differs from the baseline by `> 1e-9`, a tooltip on the cell shows `"baseline: {b_spec.bound:.6f}"` — visually communicates damping-loop correction. (4) Locked DoF rendered as `"—"` in gray; non-locked rendered as 6-decimal float. (5) "Open Frame Report in New Window" button disabled at construction; enabled in `_update_fk_display()` when a frame's data is loaded; disabled again in `clear()`. 11 tests in `tests/test_gui_results_viewer.py`. |
| 2026-06-28 | Claude (C-5 implementation) | **Implemented `gui/point_pair_panel/point_pair_panel_widget.py`** and `gui/point_pair_panel/__init__.py` (commit `63cd2ea`, same as C-4). **Key decisions:** (1) **Section 5.3 compliance for connectivity check**: `_are_connected(project, frame_a, frame_b)` builds a `networkx.Graph` from `project.frames`/`project.edges` (Pydantic `FrameModel`/`HTMEdgeModel` instances) directly — never calls `project_model_to_frame_graph()`. Core objects are only constructed in `_update_envelope()`, which runs at display time (after a run), not at selection time. (2) `_repopulate_frame_combos()` uses `blockSignals(True)` on both combos during refill and defaults `_frame_b_combo` to index 1 (not 0) to avoid A==B default. (3) `_on_saved_row_changed(row)` also uses `blockSignals(True)` during combo updates to prevent re-triggering `_on_selection_changed()` during a programmatic load. (4) `_on_selection_changed()` auto-populates name as `f"{frame_a} → {frame_b}"` only when both frames are different AND connected — avoids overwriting a user-typed name when they're tweaking an already-saved analysis. (5) `clear()` only resets `_trial_data`; preserves project and combo state — avoids flicker when New is invoked (the graph/frames would also be reset via `set_project()` called separately). (6) `_fill_envelope_table()` renders values to 6 decimal places; all cells are read-only (`Qt.ItemFlag.ItemIsEnabled` only). **`isHidden()` headless-Qt invariant** first documented in this module's tests — all future GUI tests must use `isHidden()` not `isVisible()` for visibility assertions in offscreen mode (see Section 6.20). 11 tests in `tests/test_gui_point_pair_panel.py`. Suite after C-4 and C-5 combined commit: **295 passed, 0 skipped.** |
| 2026-06-28 | Claude (C-6 + C-7 implementation) | **Implemented 7 cross-panel integration tests in `tests/test_gui_main_window.py` (C-6)** and **`QSettings` window/dock persistence + Recent Files menu in `gui/main_window.py` (C-7)** (commit `f1ffa2c`). **C-6 key decisions:** (1) All 7 tests call `MainWindow` internal handlers directly (`_on_run_completed`, `_on_graph_editor_changed`, `_new_project`, `_on_run_failed`) — no background threads, no `qtbot.waitSignal`. (2) `_make_trial_data(project)` helper calls `project_model_to_frame_graph()` + `MonteCarloFKEngine.run()` on the main thread — acceptable in tests since it runs in <0.5 s for 50 trials. (3) Tests import `_empty_project` (module-level helper) for clean new-project state. **C-7 key decisions:** (1) `QSettings("TolTransform", "TolTransform")` — both args are the same string (org name and app name); the platform store location is OS-dependent (macOS: `~/Library/Preferences/com.TolTransform.TolTransform.plist`; Linux: `~/.config/TolTransform/TolTransform.conf`; Windows: Registry). (2) `_restore_settings()` called at end of `__init__()` after `_setup_ui()` — dock object names must already be registered (via `setObjectName()` in `_setup_docks()`) before `restoreState()` is called, otherwise the dock positions are ignored. (3) `_save_settings()` called from `closeEvent()` before `event.accept()` — dock state saved at close, not on every layout change. (4) Recent Files: `_add_recent(path)` uses list comprehension `[path] + [p for p in self._recent_files if p != path]` — prepend + dedup in one line, then cap with slice `[:_MAX_RECENT]`. `_open_recent(path)` checks `os.path.exists()` first; on miss, shows `QMessageBox.critical` and calls `_remove_recent()` before returning. (5) "Clear Recent" is a separator + action at the bottom of the recent menu, so it is never shown when the list is empty (the menu shows only the disabled "(No recent files)" entry in that case). **Milestone C complete.** Suite: **302 passed, 0 skipped.** |
| 2026-06-28 | Claude (D-1 + D-2 planning) | Planned Milestone D on feature branch `feature/d1-d2-viewer-edge-edit`. **D-1 `gui/frame_viewer/`:** `FrameViewerWindow(QWidget, Qt.Window)` opened via `View → 3D Frame Viewer` (Ctrl+3); two modes: *Frames* (pyqtgraph `GLViewWidget` with per-frame coordinate triads via `GLLinePlotItem` in RGB = XYZ, grey edge-connection lines) and *Point Cloud* (`GLScatterPlotItem` from `trial_data.frames[frame][:, :3, 3]`, viridis depth colormap); Section 5.3-compliant world-transform helper `_compute_world_transforms(project)` chains 4×4 numpy matrices from `HTMEdgeModel.nominal` only (no `FrameGraph` or core objects); lazy window stored in `MainWindow._frame_viewer`; 4 unit tests for the helper function (no headless OpenGL rendering tests). New dependency: `pyqtgraph` added to `requirements.txt`. **D-2 `gui/graph_editor/edit_edge_dialog.py`:** `EditEdgeDialog(QDialog)` pre-populated via `HTMEntryWidget.set_htm_input_model(edge.nominal)` (method already existed at line 91); parent/child shown as read-only label; name + nominal HTM editable; duplicate-name check excludes the original name; triggered by double-click on an edge row in `FrameEdgeTree` OR a new "Edit Selected" button; `GraphEditorWidget._on_edit_edge()` replaces `project.edges[idx]` in-place + emits `project_changed`; 3 new tests (pre-population, accept updates project, cancel is a no-op). Spec updated: Sections 6.21, 6.22, 7.5 (Milestone D), 7.6 (formerly 7.5 Deferred), allocation_example.py task marked ✅. |
| 2026-06-28 | Claude (allocation engine enhancements, feature branch → main) | **IK allocation improvements implemented and merged to main.** (1) **`RSSAllocation`** class added to `sim/allocation.py` — statistical RSS formula (`s = min_k { B_k / sqrt(Σ J[k,col]²) }`), sqrt(N) times less conservative than `EqualAllocation` for N independent contributors; becomes the default objective in the GUI. (2) **Binary-search angular refinement** (`_bisect_angular`): after the fixed-step damping loop finds the first passing iteration, binary-searches between the last failing and first passing allocations to within 0.5% relative tolerance — eliminates the ~10% slack inherent in a fixed `gamma=0.9` step. `_mc_validate()` extracted as a shared helper used by both `validate()` and the bisection. (3) **`AllocationResult`** gains `target_tolerance: ToleranceSpec6 \| None` and `method: str` fields so the results viewer can display what was asked for and which objective was used. (4) **IK results viewer** (`gui/results_viewer/`): achieved-bounds table gains a "Target ±" column (5 columns total: DoF / Target ± / Min / Max / Pass?); status label now appends `[Statistical (RSS)]` or `[Worst-Case]`. (5) **Run panel** (`gui/run_panel/`): Method `QComboBox` added to IK group ("Statistical (RSS)" default / "Worst-Case (Linear Sum)"); **Max iterations `QSpinBox`** added (default 30, range 1–500), wired to `AllocationEngine.allocate(max_iter=...)` — lets users increase iterations when the solver is close but not converging. (6) **`SplitAllocation`** explored but removed — a two-step solver that computed separate bounds for translational vs. angular DoF hit fundamental limitations (single damping loop only tightens angular DoF; combined RSS of ang+trans contributions could exceed budget). Suite: **311 passed** on merge. |
| 2026-06-28 | Project Owner (request), Claude (implementation) | **`LoosestAllocation` (log-sum NLP) added to `sim/allocation.py`; becomes the new default IK objective in `allocate()` and the GUI.** Motivated by a practical failure of the prior single-scale-factor approach: `EqualAllocation` and `RSSAllocation` both compute a single global `s = min_k(s_k)` and apply it uniformly — so a tight dx target (e.g., 0.001 rad) suppresses rz to the same tight value even when the rz budget is 10 rad and the two DoFs are structurally independent. The correct formulation maximises all free-DoF bounds individually subject to the output-envelope constraints. **Phase 1 — LP (superseded).** The initial implementation used a linear-sum LP (`maximize Σ b_ij`, `|J| @ b ≤ B`). Nominal rotations between frames create Jacobian cross-coupling: when J[4, ry_col] and J[4, rz_col] are both nonzero (ry and rz inputs both contribute to the Ry output), the LP finds a polytope vertex and assigns all budget to one DoF while setting the other to exactly zero. Tested: a single `Ry(π/2)` edge produced b_ry or b_rz = 0.0000000 — unmanufacturable and useless. **Phase 2 — Log-sum NLP (current, locked).** Replace the linear-sum objective with `maximize Σ log(b_ij)`. Because `log(b) → −∞` as `b → 0`, every variable is infinitely penalized from reaching zero, and the optimizer distributes budget proportionally rather than concentrating it at a vertex. Under symmetric coupling (J coefficients equal), the log-sum KKT conditions produce equal shares; under asymmetric coupling, shares are proportional to 1/J[k,col] for the binding constraint's Lagrange multiplier. The problem remains convex NLP (concave objective, linear constraints) so any local optimum is global. Implemented via `scipy.optimize.minimize(method="trust-constr")` with `LinearConstraint` and `Bounds`. **Numerical conditioning:** the original SLSQP warm-start used the global equal-allocation `s` for all DoFs; with a 10 000:1 target ratio this created log-gradients differing by 4 orders of magnitude and destabilised the linesearch. Fix: per-DoF warm start `x0[j] = 0.9 × min_k { B_k / (|J[k,j]| × n_sharing_k) }` (each variable starts near its own binding point); switched to trust-constr (trust-region, not linesearch, handles gradient scale mismatch). **`_bisect_angular` bug fix (same commit):** `lo_scale` was updated each bisection iteration but `lo` (the allocation object) was not, so `ratio = mid_scale / lo_scale` used a stale denominator and `_scale_angular(lo, ratio)` produced wrong absolute values on iterations ≥ 2. Fix: `base_scale = lo_scale` frozen before the loop; `ratio = mid_scale / base_scale` always. **`_build_result` refactored** from `(free_edges, s: float)` to `(free_edges, bounds: np.ndarray)` — required to express per-DoF heterogeneous bounds from the NLP; EqualAllocation/RSSAllocation pass `np.full(6*N, s)`. **GUI:** "Loosest (LP)" added as first entry in the IK Method combo (`gui/run_panel/`). **4 new tests in `tests/test_allocation.py`** including a regression test (`test_loosest_allocation_no_zero_bounds_on_coupled_chain`) that fails with the LP and passes with the log-sum NLP — locked as a permanent regression guard. Suite: **321 passed**. |
| 2026-06-28 | Project Owner (request), Claude (implementation) | **`LoosestAllocation` made the sole allocation objective; `EqualAllocation`, `RSSAllocation`, and `AllocationObjective` ABC removed from `sim/allocation.py`.** Motivation: both prior single-scale-factor methods collapse the allocation to `s = min_k(s_k)` applied uniformly — a tight constraint on one output DoF (e.g. dx = 0.001) suppresses every other DoF to that same scale even if the other budgets (e.g. rz = 10.0) are structurally independent. `LoosestAllocation` already strictly dominates both in every case, so the other implementations were dead code with no valid use. **Changes:** `sim/allocation.py` — `ABC`/`abstractmethod` import removed; `AllocationObjective`, `EqualAllocation`, `RSSAllocation` classes deleted; `AllocationEngine.solve`, `allocate`, `solve_multi`, `allocate_multi` no longer accept an `objective` parameter; `_run_nlp` called directly; `method` field hardcoded to `"LoosestAllocation"`. **GUI run panel** — Method `QComboBox` removed (was "Loosest / Statistical RSS / Worst-Case"); `_RunWorker` no longer holds an `objective` field. **GUI results viewer** — method label hardcoded to `"Loosest (LP)"`. **Tests** — `EqualAllocation` import removed; `test_loosest_allocation_beats_equal_on_mixed_target` rewritten as `test_solve_fills_independent_dof_budgets` (asserts LP fills each independent DoF budget to its own limit without needing an EqualAllocation comparison); all `objective=LoosestAllocation()` kwargs stripped. Suite: **239 passed** (non-GUI). |
| 2026-06-28 | Project Owner (request), Claude (implementation) | **Multi-pair IK allocation: `AllocationEngine.solve_multi` / `allocate_multi`.** Extends the IK engine from a single point-pair constraint to simultaneous multiple constraints — e.g., mirror A ↔ B AND mirror A ↔ C — so that tolerance bounds for all edges are optimised jointly rather than independently. **Core algorithm:** for P pairs, build one padded Jacobian per pair (`J_full_p ∈ ℝ^{6 × 6N}` with zero columns for edges not on that pair's path), stack all active (pair, DoF) constraint rows into `A ∈ ℝ^{C × n_free}`, then pass to `LoosestAllocation._run_nlp(A, b)` — the same log-sum NLP that powers single-pair allocation, now with C rows. Shared edges appear in multiple constraint rows, so the optimizer automatically finds the tightest globally-consistent bound without any special handling. **`_run_nlp` extraction:** the NLP core was extracted from `solve()` into a new `LoosestAllocation._run_nlp(A, b)` method shared by both `solve()` and `solve_multi()`. **Multi-pair damping loop (`allocate_multi`):** "failed" = any pair fails MC validation; `gamma` applied uniformly to all free angular bounds; bisection (`_bisect_angular_multi`) validates all pairs simultaneously. **`AllocationResult` additions:** `per_pair_validation: list[tuple[str, str, ValidationReport]] | None` and `per_pair_targets: list[tuple[str, str, ToleranceSpec6]] | None`. **GUI — run panel:** single frame-pair UI replaced by a dynamic `_ConstraintRowWidget` list with "Add Constraint" / "✕" buttons; worker calls `allocate_multi()`. **GUI — results viewer:** per-pair `QGroupBox` sections with pass/fail color in title, and a `DoF | Target ± | Min | Max | Pass?` table per pair so users can directly self-verify achieved vs. requested. **5 new tests** (16 total in `tests/test_allocation.py`): shared-edge correctness, independent-pair isolation, result structure, MC validation with margin, lever-arm multi-pair convergence. Suite: **239 passed** (non-GUI tests). |
| 2026-06-28 | Project Owner (request), Claude (implementation) | **Asymmetric tolerance bounds (FK mode only): `lower`/`upper` format as an alternative to `±bound`.** Motivated by real-world manufacturing scenarios where a feature has a tighter constraint in one direction than the other (e.g., a clearance hole that must be at least 0.002 mm oversize but can be at most 0.006 mm oversize). IK allocation remains symmetric-only. **`core/tolerance.py`** — `ToleranceSpec` dataclass gains two optional fields `lower: float | None` and `upper: float | None`. In asymmetric mode (both set): `bound` is auto-derived as `max(abs(lower), abs(upper))` in `__post_init__`; the new `is_asymmetric` property returns True; `variance` property extended to compute `E[X²] = Var(X) + E[X]²` for off-centre intervals. `sample()` dispatches to asymmetric samplers when `is_asymmetric`. Mutual-exclusivity enforced: setting only one of lower/upper raises `ValueError`. **`core/sampling.py`** — three new functions: `sample_uniform_asymmetric(lower, upper, n_trials, rng)`, `sample_normal_asymmetric(lower, upper, sigma_level, n_trials, rng)` (mean = midpoint, sigma = half-width / sigma_level), and `sample_asymmetric(distribution, lower, upper, sigma_level, n_trials, rng)` dispatcher. **`persistence/schema.py`** — `ToleranceSpecModel` gains `lower: float | None` and `upper: float | None` with `@model_validator(mode="after")` cross-field validation; `_tol_to_model` / `_model_to_tol` conversion helpers updated to preserve and restore asymmetric fields. **`postprocess/stats.py`** — `compute_tolerance_sensitivities` updated to use `spec.variance` (handles both symmetric and asymmetric correctly) instead of the inline symmetric-only formula. **`gui/tolerance_editor/tolerance_editor_widget.py`** — `_DofRow` gains a mode toggle button (`±` / `↔`), a `QStackedWidget` in the bound column (page 0: single `bound_spin`; page 1: `lower_spin` + `upper_spin` with grey "min"/"max" labels). Mode switch pre-populates the opposite representation from current values. **49 new tests** across `tests/test_tolerance.py` (17 new), `tests/test_schema.py` (7 new), `tests/test_gui_tolerance_editor.py` (8 new). Suite: **282 passed** (non-GUI), **302 passed** (full). Committed as `aac8cef`. |
| 2026-06-28 | Claude (bug fix) | **NameError in `allocate_multi` non-convergence path.** The non-convergence `return` branch of `AllocationEngine.allocate_multi()` referenced the undefined variable `method_name` (a stale remnant from the multi-objective era) instead of the hardcoded string `"LoosestAllocation"`. This caused a `NameError` whenever `allocate_multi` exhausted `max_iter` without converging. Fix: replaced `method=method_name` with `method="LoosestAllocation"` in the non-convergence return. Regression test `test_allocate_multi_non_convergence_returns_method_field` added to `tests/test_allocation.py` — forces the non-convergence branch with an impossibly tight target and verifies `result.method == "LoosestAllocation"` (was `NameError` before the fix). Committed as `805fec1`. |
| 2026-06-29 | Project Owner (request), Claude (implementation + tests) | **"Apply Corrected Allocation to Project" button + IK simulation parameter persistence.** Two features implemented on feature branch `feature/apply-allocation-persist-ik-params` (commit `ca37d91`), merged to `main`. **(1) Apply Allocation button** (`gui/results_viewer/`): after a converged IK run, a new `QPushButton("Apply Corrected Allocation to Project…")` (enabled only when `result.converged`) lets the user write `corrected_allocation` bounds back to the project's edge tolerance specs. `_on_apply_clicked()` builds a preview of all changes, shows a `QMessageBox.question()` confirmation dialog, then writes `ToleranceSpecModel(distribution="uniform", bound=spec.bound)` for each non-locked DoF on matching edges. Emits the new `project_changed = Signal()` on `ResultsViewerWidget` after writing; `MainWindow._on_allocation_applied()` sets the dirty flag and calls `_tolerance_editor.refresh_view()` so the written bounds appear immediately. **(2) IK parameter persistence** (`persistence/schema.py`, `gui/run_panel/`): added `IKConstraintModel(frame_a, frame_b, target: ToleranceSpec6Model)` and extended `SimSettingsModel` with `ik_constraints: list[IKConstraintModel] = []` and `ik_max_iter: int = 30`. Both fields default safely so existing project files load unchanged. `_load_ik_constraints_from_project()` / `_save_ik_constraints_to_project()` in `RunPanelWidget` rebuild constraint rows from or write them to `project.sim_settings`; `_on_constraints_changed()` and `_on_max_iter_changed()` keep the project in sync on every UI edit. `validate_references()` in `ProjectModel` extended to check IK constraint frame references. **(Test suite):** 71 new tests (6 in `test_schema.py` for `IKConstraintModel`; 8 in `test_gui_run_panel.py` for constraint persistence; 8 in `test_gui_results_viewer.py` for apply-button; 1 fixed in `test_gui_main_window.py`; plus 2 pre-existing broken tests fixed). **`tests/conftest.py` autouse fixture** added to prevent Qt `MainWindow` hang in consecutive tests (root cause: `_set_dirty(True)` → `closeEvent` → `QMessageBox.question()` blocking forever in offscreen mode). **Suite: 373 passed, 0 skipped.** |
| 2026-06-28 | Project Owner (report), Claude (root-cause and fix) | **IK allocation convergence failure when locked DoFs have asymmetric (min/max) tolerances.** **Root cause:** `LoosestAllocation.solve()` built the NLP constraint as `Σ_free |J[k,col]| · b[col] ≤ B[k]` where `B[k] = target[k].bound` — the full target budget. This ignored the contribution of locked DoFs to the MC output envelope entirely. For symmetric locked DoFs with small bounds, the damping loop compensated iteratively. For asymmetric locked DoFs, `spec.bound = max(|lower|, |upper|)` can be significantly larger than a naive symmetric bound, so the NLP over-allocated free DoFs by a wider margin and the damping loop (which only scales free angular DoFs) could not recover — the locked contribution is fixed and unaffected by angular damping. **Fix — two changes to `sim/allocation.py`:** (1) New private helper `_compute_locked_budget(frame_graph, frame_a, frame_b, path, free_edges, J_free) → np.ndarray (6,)` computes `Σ_locked |J[k,col]| · spec.bound` for every locked DoF on the path. Handles two classes: locked DoF within partially-free edges (columns present in `J_free` but outside `free_mask`) and fully-locked path edges (absent from `J_free` entirely, handled by a second `compute_sensitivity` call). (2) `LoosestAllocation.solve()` gains a `locked_budget: np.ndarray | None = None` parameter; the NLP budget is now `b_eff[k] = max(0, B[k] - locked_budget[k])` rather than `B[k]`. Both `AllocationEngine.solve()` and `AllocationEngine.solve_multi()` compute `locked_budget` via the helper and pass it through. `solve_multi()` also now stores the path per pair (new `pair_paths` list) so `_compute_locked_budget` can find fully-locked edges without recomputing the path. **Backwards compatibility:** existing setups where all locked DoFs have `bound=0` see `locked_budget = [0,…,0]` and behaviour is unchanged — all 17 existing allocation tests pass unmodified. **Practical effect:** the baseline NLP allocation is now accurate enough that MC validation passes on the first attempt (0 damping iterations) for the common case where locked DoF contributions are properly budgeted. Regression test `test_ik_convergence_with_asymmetric_locked_dofs` added (18th test in `tests/test_allocation.py`): verifies both the exact linear allocation (`e1.dx.bound == target - locked_contrib`) and that `allocate()` converges with `iterations_used=0`. Suite: **270 passed** (non-GUI). |
| 2026-06-30 | Project Owner (request), Claude (implementation) | **Deferred cleanup items (4 items) — committed directly to `main`.** **(1) `gui/_table_helpers.py`:** extracted `_ro_item()` and `_fill_envelope_table()` into a shared module; both `results_viewer_widget.py` and `point_pair_panel_widget.py` import from it (was byte-for-byte duplicate code). **(2) `tests/helpers.py` + `pytest.ini`:** consolidated `_FixedToleranceSpec6`, `_uniform_spec`, `_tol6` from 5+ independent test-file definitions into a single shared module; `pytest.ini` adds `pythonpath = tests` so `helpers` and `conftest` are importable from test files; `test_integration.py` now imports `make_tol`/`make_zero_tol`/`make_htm`/`DEFAULT_ATOL`/`SMALL_ANGLE_ATOL` from `conftest` instead of redefining locally. **(3) `_ConstraintRowWidget.get_target_model()` (Section 5.3 fix):** renamed from `get_target()`; now returns `ToleranceSpec6Model` (schema type) instead of `ToleranceSpec6` (core type); conversion to core objects now happens only in `_on_run_clicked()` at the run boundary (the designated schema→core conversion point); `_save_ik_constraints_to_project()` simplified — no more core→schema→core round-trip; `_collect_ik_targets()` return type changed to `list[tuple[str, str, ToleranceSpec6Model]]`. **(4) `pytest.skip` antipattern fix:** three apply-button tests in `test_gui_results_viewer.py` replaced `if not result.converged: pytest.skip(...)` with `assert result.converged` — non-convergence on a trivial single-edge test case is a genuine failure, not a reason to skip. Suite: **373 passed, 0 skipped**. |
| 2026-06-30 | Project Owner (request), Claude (implementation) | **Milestone E-1: PyInstaller standalone packaging for Mac + Windows.** **(1) `toltransform.spec`:** PyInstaller one-folder spec; `collect_data_files` for `pyqtgraph` (GLSL shaders required by `GLViewWidget`), `matplotlib`, and `pytransform3d`; hidden imports for `PySide6.QtOpenGL`, `PySide6.QtOpenGLWidgets`, `pyqtgraph.opengl`, `matplotlib.backends.backend_qtagg`, and pydantic v2 internals; excludes `tkinter`/`PyQt5`/`PyQt6`; Mac-only `BUNDLE` block with `CFBundleShortVersionString=1.0.0`, `NSHighResolutionCapable=True`, `bundle_identifier=com.joeylitjens.toltransform`. **(2) `packaging/rthook_opengl.py`:** runtime hook (runs before any Python imports) that sets `QT_OPENGL=desktop` on Windows so Qt uses native desktop OpenGL instead of ANGLE (its DirectX-based wrapper, which breaks pyqtgraph's `GLViewWidget`). **(3) `main.py`:** sets `QSurfaceFormat` with `CompatibilityProfile` + 24-bit depth buffer before `QApplication` construction; `os.environ.setdefault("QT_OPENGL", "desktop")` guards for Windows. **(4) `assets/`:** `toltransform_icon.png` (1024×1024 engineering-themed TT logo); `icon.icns` (Mac, generated via `sips` + `iconutil` with 16/32/64/128/256/512 px + @2x); `icon.ico` (Windows, generated via Pillow with 16/32/48/64/128/256 px). **(5) `.github/workflows/build.yml`:** CI matrix (`macos-latest` + `windows-latest`), triggered by `v*.*.*` tag pushes; checkout → Python 3.12 → `pip install -r requirements.txt -r requirements-dev.txt` → `pyinstaller toltransform.spec` → zip artifact (`zip -r` on Mac, `Compress-Archive` on Windows) → `actions/upload-artifact` (30-day retention) → `gh release upload` to attach to the matching GitHub Release. **(6) `requirements-dev.txt`:** added `pyinstaller>=6.0,<7`. **(7) GitHub Release v1.0.0** created with both `TolTransform-macOS.zip` and `TolTransform-Windows.zip` attached. Confirmed working: Mac (.app) double-click launches cleanly; Windows build confirmed functional by project owner. |
| 2026-06-30 | Project Owner (report), Claude (root-cause and fix) | **Windows HiDPI bug: 3D viewer rendered with distorted perspective ("4D" appearance) at 125% display scaling.** **Root cause:** Windows display scaling at 125% means `QWidget.width()`/`height()` return logical pixels (80% of physical). The OpenGL framebuffer is in physical pixels. pyqtgraph's `GLViewWidget.paintGL()` calls `self.resizeGL(self.width(), self.height())` with logical pixels, which sets `glViewport(0, 0, logical_w, logical_h)` — only 64% of the framebuffer area — and computes the projection matrix for the wrong aspect ratio. macOS is unaffected because its Metal/OpenGL layer handles Retina scaling transparently before pyqtgraph sees the dimensions. **Fix:** `_HiDPIGLView(GLViewWidget)` subclass added in `gui/frame_viewer/frame_viewer_window.py`; overrides `resizeGL(w, h)` to call `super().resizeGL(int(self.width() * dpr), int(self.height() * dpr))` where `dpr = self.devicePixelRatioF()`. Uses `self.width()`/`self.height()` (not the passed logical `w`/`h`) so the physical dimensions are always up-to-date. At `dpr=1.0` (100% scaling) the override is a no-op, so Mac and 100%-scaled Windows displays are unaffected. `_setup_ui()` updated to instantiate `_HiDPIGLView()` instead of `GLViewWidget()`. **Second fix:** `_reset_camera()` now passes `pos=QVector3D(0, 0, 0)` explicitly to `setCameraPosition` so the pan center is reset along with distance/elevation/azimuth — previously, panning would shift the camera target and Reset View left it at the shifted position. |
| 2026-06-29 | Project Owner (request), Claude (implementation) | **Full codebase cleanup pass** — branch `fix/codebase-cleanup`, merged to `main`. Three-agent audit found and fixed the following items across 7 commits. **(1) Bug fix:** `gui/frame_viewer/frame_viewer_window.py` — `model.screw_axis` → `model.axis` (the `HTMInputScrew` Pydantic model names the field `axis`; the wrong attribute would raise `AttributeError` whenever the 3D viewer tried to render a project whose nominal HTM was in screw-parameter format). **(2) Dead/stale code:** removed `objective = None` leftover from removed `AllocationObjective` ABC (`run_panel_widget.py`); removed `if TYPE_CHECKING: pass` placeholder (`frame_graph.py`); removed duplicate `ax.set_xlabel()` call in `plot_pareto_sensitivity` (`reporting.py`); removed `f.metadata or None` semantic no-op (`schema.py`); eliminated double `np.asarray()` call in `HTM.from_matrix` (`transforms.py`); removed dead infinity-guards in `_combine_validation_reports` and an inaccurate comment (`allocation.py`); fixed `TrialData` docstring that suggested `np.linalg.inv` instead of the closed-form inverse (`monte_carlo_fk.py`); simplified `_installed[1]` to `_installed_major` scalar (`conversions.py`). **(3) Test cleanup:** removed five unused imports across four test files (`make_edge_rng`, `_RNG`, duplicate `HTMEdgeModel`, `QtBot`, `LoosestAllocation`). **(4) `DOF_LABELS` consolidation:** removed duplicate definition from `sim/allocation.py`; now imported from the canonical source `postprocess/stats.py`. **(5) SHA-256 dedup:** `make_edge_rng()` delegates to `_spawn_key()` internally; `MonteCarloFKEngine.run()` no longer calls both per edge (was computing SHA-256 twice per edge per simulation run). **(6) `skew()` relocation:** moved from `core/tolerance.py` to `core/transforms.py` where it belongs as a pure math primitive; `core/tolerance.py` and `core/frame_graph.py` updated to import from the new location; `test_tolerance.py` updated accordingly. **(7) `FrameGraph` public API:** added `get_edge(name) -> HTMEdge` and `are_connected(frame_a, frame_b) -> bool`; all external `frame_graph._edges[name]` accesses replaced with `get_edge()`; `_assert_same_component` now delegates to `are_connected`. **(8) `main_window.py`:** File menu now appears before View menu (standard convention). Extracted `_apply_project_to_ui()` helper eliminating the identical 4-line block repeated verbatim in `_new_project`, `_open_project`, and `_open_recent`; the helper also calls `_frame_viewer.update_graph(project)` which was previously missing from the open/recent paths (the 3D viewer was left showing the old project after opening a new file). **(9) Minor GUI polish:** `import math` moved to module level in `htm_entry_widget.py`; dead `_DofRow._loading` attribute removed; `enumerate(zip(DOF_LABELS, range(6)))` → `enumerate(DOF_LABELS)` in `results_viewer_widget.py`; unused `col` → `_col` rename in `graph_editor_widget.py`; `queue.pop(0)` → `collections.deque.popleft()` in `frame_viewer_window.py`; `destroyed` signal connected on `_FigureWindow` so closed windows self-remove from `_open_windows` list. **(10) Test quality:** replaced `__import__()` antipattern with regular import in `test_gui_graph_editor.py`; renamed misleading `test_error_is_zero_at_intermediate_frame_B` (it asserts dx=0.003, not zero); removed dead `b_normal`/`b_normal_equiv` variables and fixed docstring in `test_stats.py`; standardised `coverage=0.997` → `0.9973` in `test_bounding_shapes.py`. **(11) Core API:** removed `convention` parameter from `euler_to_rotation_matrix`, `rotation_matrix_to_euler`, `HTM.from_xyz_euler`, and `HTM.to_xyz_euler` (only `"intrinsic_zyx"` was ever valid; the parameter implied flexibility that didn't exist). **(12) Named constants:** `_DEGENERATE_AXIS_TOL = 1e-10` and `_DEGENERATE_SIGMA_TOL = 1e-15` replace inline magic literals in `bounding_shapes.py`. **(13) `ToleranceSpec6.__iter__`:** added explicit `__iter__` method; previously iteration fell through to the `__getitem__` fallback (accidental, not idiomatic). **(14) `requirements.txt` pinning:** pinned all 9 runtime dependencies to `>=current,<next-major` bounds; `pytest` and `pytest-qt` moved to a new `requirements-dev.txt`. Suite throughout: **373 passed, 0 skipped**. |

