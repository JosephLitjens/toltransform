# CLAUDE.md — Standing Instructions for TolTransform

This file is read automatically by Claude Code at the start of every session in this repo. Its job is to make sure no session starts from zero — the design has already been thought through in detail; your job is to implement it faithfully, not re-derive it.

## Before doing anything else

1. **Read `docs/design_spec.md` in full.** This is the authoritative design specification, not just a plan — it contains locked architectural decisions, exact module-by-module task breakdowns, interface contracts, and a full changelog of why things are the way they are. Section 6 is the most important section for implementation work: every module has its own subsection with a granular, ordered task list, explicit deliverables, and a depends-on/used-by/public-API interface contract.
2. **Activate the virtual environment before running any Python command:**
   ```bash
   source .venv/bin/activate   # macOS/Linux — NOTE: venv is at .venv/, NOT venv/
   ```
   If `.venv/` doesn't exist yet (e.g., fresh clone), create it first: `python3 -m venv .venv`, activate, then `pip install -r requirements.txt`.
3. **Check `docs/design_spec.md` Section 11 (Changelog) for the most recent entries** before starting work, in case a decision changed since your last session.

## Non-negotiable conventions (do not silently deviate from these)

These are locked decisions from the design spec. If you think one of them is wrong, say so and ask — do not just implement something different.

- **Small-angle approximation only.** No large-angle rotation handling anywhere in the tolerance/perturbation model (Section 1.2, 2.2).
- **Perturbation convention: local-frame, right-multiplication** — `T_perturbed = T_nominal @ T_delta(delta)` (Section 2.2.2). This must match the adjoint convention used in `core/frame_graph.py`'s sensitivity primitives exactly.
- **No correlated tolerances.** Every DoF and every edge is sampled independently. Do not add covariance/correlation modeling without an explicit new discussion (Section 1.2).
- **FK chains are strictly serial/open — no closed loops, no parallel mechanisms.** `FrameGraph.validate_dag()` enforces at most one incoming edge per Frame (Section 2.3).
- **Disjoint graph components are never silently auto-connected.** If two Frames have no path between them, raise `DisjointFramesError` with the exact locked message text in Section 2.3.1/6.3 — never fabricate a connection.
- **Euler convention: intrinsic ZYX.** Documented in `core/conversions.py`.
- **`TrialData` stores full 4x4 poses per Frame per trial**, not reduced 6-vectors (Section 6, top-level note).
- **Rotation-vector convention: ω = θu**, used consistently between `postprocess/stats.py` and `postprocess/bounding_shapes.py`. `fit_rotation_cone()`/`fit_rotation_box()` must never accept raw `(N,4,4)` pose arrays.
- **The cone is the lead representation for angular error**; the per-axis box is a secondary/expandable cross-check, not co-equal.
- **`adjoint()` and `compute_sensitivity()` live in `core/frame_graph.py` — nowhere else.** Both `sim/allocation.py` and `postprocess/stats.py` consume this single shared implementation. Do not write a second copy.
- **`locked` on a tolerance does not mean zero error.** A locked DoF is still sampled in FK mode — `locked` only excludes it from the free-variable set in inverse allocation (Section 6.2/6.7).
- **GUI code talks only to `persistence.schema` models, never directly to `core`/`sim` objects** (Section 5.3). This rule applies starting Milestone C — not relevant yet in Milestones A/B-1/B-2.

## Milestone discipline

Build order is **A → B-1 → B-2 → C**, per Section 7. Each phase has explicit exit criteria:

- **Milestone A** must be fully complete, tested, and hand-verified before B-1 starts.
- **Milestone B-1 is not done until all three Section 9.1 physical validation benchmarks pass**: the Linear Stack-Up (RSS) Benchmark, the Sine-Bar Lever Arm Benchmark, and the Common-Ancestor Cancellation Benchmark. Do not start B-2 (the inverse allocator) before these pass — this gating is intentional, not a suggestion.
- **Milestone C (GUI) does not start until A, B-1, and B-2 are all complete and tested.**

If asked to jump ahead of this order, flag it rather than silently complying.

## Repository

This repo's remote is **https://github.com/JosephLitjens/toltransform**. Confirm `git remote -v` points here before pushing if you're ever unsure.

- **Commit after each completed task** (already stated above) **and push to `origin` after every commit** — don't let work sit unpushed across a session boundary. If a session ends (or is interrupted) before pushing, the next session should check `git status` and `git log origin/main..HEAD` first and push anything outstanding before starting new work.
- Use clear, conventional commit messages referencing the task ID from Section 7's tables where applicable, e.g. `git commit -m "A1: implement HTM class and conversions (core/transforms.py, core/conversions.py)"`.
- Do not force-push to `main`. If a mistake needs correcting, fix it forward with a new commit rather than rewriting pushed history.
- If `git push` fails (auth prompt, conflict, etc.), stop and surface the exact error to the user rather than retrying blindly or working around it — pushing is the one step that shouldn't fail silently.

## Working style

- Work through one numbered task at a time from the relevant module's Granular Task List in Section 6. Don't skip ahead to later tasks in the same module before earlier ones are done and tested.
- Every module's spec includes its own test requirements — write the tests specified, not just whatever you'd default to.
- Prefer starting a session in Plan Mode (the user can toggle this) for anything beyond a small, obvious edit — propose the approach before writing code, especially for anything touching the sensitivity/Jacobian math, the perturbation model, or the frame graph's validation logic, since these are the modules where a subtle error would be easy to miss and hard to debug later.
- Commit after each completed task (not just at the end of a module, and not in one giant end-of-session commit) so there are clean rollback points — see the Repository section below for commit/push specifics.
- If you hit an ambiguity the design spec doesn't resolve, make the simplest reasonable choice, implement it, and explicitly flag it to the user as a decision made — don't silently invent a convention. Significant decisions should be proposed as a `docs/design_spec.md` Section 11 changelog entry for the user to confirm, not made unilaterally in code with no record.

## Where things stand

*(Update this section at the end of each session so the next session — yours or a fresh one — knows exactly where to pick up.)*

- **Current milestone:** Post-milestone. All milestones A → B-1 → B-2 → C → D complete. No planned work outstanding.
- **Suite: 373 passed, 0 skipped** (full suite including GUI).
- **Next task:** None — awaiting new feature requests or bug reports.

**Last completed work (merged to main 2026-06-29):**
- `fix/codebase-cleanup`: Full codebase audit and cleanup pass (14 categories, 7 commits). See Section 11 changelog for details.
- `feature/apply-allocation-persist-ik-params`: Apply Allocation button + persist IK params.
- `feature/asymmetric-tolerances`: Asymmetric tolerance bounds + NameError fix + locked-DoF budget fix.

**✅ Multi-pair IK allocation complete (merged to main 2026-06-28):**
- `AllocationEngine.solve_multi(fg, targets)` — builds stacked padded Jacobian `A ∈ ℝ^{C × n_free}` from all P pairs' paths; calls `LoosestAllocation._run_nlp(A, b)` to find globally-consistent per-DoF bounds. Shared edges appear in multiple constraint rows and are automatically constrained by the tightest binding pair.
- `AllocationEngine.allocate_multi(fg, targets, ...)` — full pipeline: `solve_multi` + MC validation for ALL pairs simultaneously + `_bisect_angular_multi` + damping loop. "Failed" = any pair fails; `gamma` applied uniformly to all free angular bounds.
- `LoosestAllocation._run_nlp(A, b)` extracted from `solve()` — shared by single-pair and multi-pair paths with no duplication.
- `AllocationResult` gains `per_pair_validation` and `per_pair_targets` fields; `None` for single-pair `allocate()` results.
- GUI run panel: single frame-pair UI replaced by dynamic `_ConstraintRowWidget` list with "Add Constraint" / "✕" buttons.
- GUI results viewer: per-pair `QGroupBox` with pass/fail title color and `DoF | Target ± | Min | Max | Pass?` table per pair.
- 5 new tests (16 total in `tests/test_allocation.py`): shared-edge correctness (key correctness test), independent-pair isolation, result structure, MC validation with margin, lever-arm multi-pair convergence.

**✅ EqualAllocation/RSSAllocation removed (2026-06-28):**
- `AllocationObjective` ABC, `EqualAllocation`, and `RSSAllocation` deleted from `sim/allocation.py`. `LoosestAllocation` is now the sole objective, called directly — no interface indirection, no `objective` parameter on `solve` / `allocate` / `solve_multi` / `allocate_multi`.
- GUI Method combo removed from run panel; method label hardcoded to "Loosest (LP)" in results viewer.
- `test_loosest_allocation_beats_equal_on_mixed_target` rewritten as `test_solve_fills_independent_dof_budgets` (no longer needs EqualAllocation to compare against).

**✅ LoosestAllocation complete (merged to main 2026-06-28):**
- `LoosestAllocation` in `sim/allocation.py` — log-sum NLP (`maximize Σ log(b_ij)`) with linear worst-case constraints; the sole allocation objective.
- **Why log-sum, not LP:** the linear-sum LP (`maximize Σ b_ij`) finds a polytope vertex and assigns zero bounds to DoFs that compete in the same Jacobian row — unmanufacturable. The log-sum forces every DoF positive via `log(0) = −∞` penalty.
- `_build_result` refactored from scalar to `np.ndarray` bounds vector — required for per-DoF heterogeneous bounds.
- `_bisect_angular` bug fixed: `ratio` was computed against an updated `lo_scale` rather than the original base, producing wrong absolute bounds on bisection iterations ≥ 2.
- 4 tests including `test_loosest_allocation_no_zero_bounds_on_coupled_chain` — permanent regression guard against LP degeneracy with cross-coupled Jacobians (e.g., `Ry(π/2)` nominal).

**✅ D-2 complete:** `gui/graph_editor/edit_edge_dialog.py` — `EditEdgeDialog(QDialog)` pre-populated via `HTMEntryWidget.set_htm_input_model(edge.nominal)`; parent/child shown as read-only labels; duplicate-name check excludes original name; triggered by double-click on edge row OR "Edit Selected" button in `GraphEditorWidget`; replaces `project.edges[idx]` in-place + emits `project_changed`. 3 new tests.

**✅ D-1 complete:** `gui/frame_viewer/frame_viewer_window.py` — `FrameViewerWindow(QWidget, Qt.Window)` with pyqtgraph `GLViewWidget`; two modes: Frames (per-frame RGB coordinate triads via `GLLinePlotItem`) and Point Cloud (`GLScatterPlotItem` from MC trial data, viridis depth colormap). `_compute_world_transforms(project)` is Section 5.3-compliant (schema types + numpy only, no core objects). Opened via `View → 3D Frame Viewer` (Ctrl+3); live-updates on graph changes and after each run. 4 unit tests (no headless OpenGL rendering tests).

**✅ IK allocation enhancements complete (merged to main 2026-06-28):**
- `_bisect_angular()` binary-search refinement — recovers ~10% slack from fixed gamma=0.9 step after damping loop converges.
- `AllocationResult` gains `target_tolerance` and `method` fields.
- Run panel: Max iterations spinbox (default 30, range 1–500).
- Results viewer: Target ± column in achieved table; method label in convergence status.
- `RSSAllocation` was added in this milestone then removed on 2026-06-28 (see above); `SplitAllocation` also explored and removed.

**✅ C-7 complete:** `gui/main_window.py` — `QSettings("TolTransform", "TolTransform")` saves/restores window geometry, dock layout, and Recent Files list (capped at 5) between sessions. `closeEvent()` saves settings before accepting; `_restore_settings()` called after `_setup_ui()`. Recent Files submenu under `File > Open Recent` with per-entry `_open_recent()` (handles missing file gracefully), `_add_recent()` (prepend+dedup+cap), `_remove_recent()`, `_clear_recent_files()`. `_save_project_as()` also calls `_add_recent()`.

**✅ C-6 complete:** `tests/test_gui_main_window.py` — 7 cross-panel integration tests: FK result routes to results_viewer (page 1) and point_pair_panel (_trial_data set); graph change refreshes run_panel and point_pair_panel frame combos; new_project resets results_viewer to placeholder and clears point_pair_panel trial data; run failure leaves results_viewer on placeholder. Suite: **302 passed**.

**✅ C-5 complete:** `gui/point_pair_panel/` — named (frame_a, frame_b) analysis persistence; connectivity check via networkx; relative-pose envelope via `point_pair_envelope_box()`; saves/deletes from `project.saved_analyses`; `project_changed` signal. Results viewer plots now open in `_FigureWindow` (standalone Qt window) rather than embedded in dock. 11 headless tests. Suite: **295 passed**.

**✅ C-4 complete:** `gui/results_viewer/results_viewer_widget.py` — QStackedWidget with placeholder / FK page / IK page. FK: frame selector combo, envelope table (6 rows, DoF|Min|Max from `frame_envelope_box`), `FigureCanvasQTAgg` embedding `generate_frame_report()` (recreated on frame switch, `plt.close` called on old), Pareto section with frame A/B combos + Compute button → `compute_tolerance_sensitivities` + `generate_sensitivity_report`. IK: convergence status label (✓ green / ✗ orange), corrected allocation table (edge × DoF, locked shown as "—" with tooltip for baseline diff), achieved envelope table (DoF|Min|Max|✓/✗). All three panels (tolerance editor, run panel, results viewer) now wrap content in `QScrollArea(widgetResizable=True)` so docks can be made arbitrarily small. 11 headless tests. Suite: **284 passed**.

**✅ C-3 complete:** `gui/run_panel/run_panel_widget.py` — `_RunWorker(QThread)` runs FK (`MonteCarloFKEngine.run`) or IK (`AllocationEngine.allocate`) on a background thread. `RunPanelWidget` has a mode selector (FK/IK), N-trials and seed spinboxes with Randomize button, and an IK-only target group (hidden in FK mode) with frame A/B combos and 6 target-bound spinboxes. `project_model_to_frame_graph()` called once on the main thread before starting the worker — the only place core objects are constructed (Section 5.3). mode/n_trials/seed written back to `project.sim_settings` on every change; IK target is ephemeral. 11 headless tests. Suite: **273 passed**.

**✅ C-2 complete:** `gui/tolerance_editor/tolerance_editor_widget.py` — per-edge, per-DoF tolerance editing panel. `_DofRow` helper embeds 6 rows (dx→rz) directly in a QGridLayout, each with distribution combo, bound spin, σ-level spin (enabled only when normal), locked checkbox, and inline error label. `ToleranceEditorWidget` uses a QStackedWidget (placeholder vs. DoF panel), edge selector combo, and bulk-apply group. Write-on-change: validates via `ToleranceSpec` constructor, writes to `setattr(edge.tolerance, dof_name, ...)`, emits `project_changed`. `GraphEditorWidget` gained `edge_selected = Signal(str)` + `currentItemChanged` handler to auto-select edges in the tolerance panel. 13 new tests in `test_gui_tolerance_editor.py`. Suite: **262 passed**.

**✅ C-1 complete (commit `e8d36b2`):** `gui/main_window.py` + `gui/graph_editor/` (5 files: `__init__.py`, `htm_entry_widget.py`, `frame_edge_tree.py`, `add_frame_dialog.py`, `add_edge_dialog.py`, `graph_editor_widget.py`). Multi-format HTM entry with live validation, frame/edge tree with root/junction labels, Add Frame/Edge dialogs, delete with referential guard. MainWindow hosts all panels as dock widgets with File > New/Open/Save/Save As menu. 19 tests in `test_gui_graph_editor.py`. Suite: **249 passed**.

**✅ B2-3 complete (commit `0c9bd9d`):** `tests/test_allocation.py` — 7 real tests replacing the `@pytest.mark.skip` placeholder. Lever-arm geometry: `base→pivot` (identity nominal, rz free) → `pivot→arm` (Tx(1m) locked) → `arm→exit` (Ry(π/2) locked). The downstream Ry(π/2) node exposes the linear/MC discrepancy: linear Jacobian at exit_node=pivot (T=I) gives J[:,5]=[0,0,0,0,0,1] (zero dy coupling), but MC sees dy≈L·δrz (first-order, large). Convergence at k=7 iterations (0.9^7·0.10=0.0478≤0.05); non-convergence demonstrated with B_dy=0.001 target. **Milestone B-2 fully complete.**

**✅ B2-2 complete (commit `b005eaf`):** `AllocationEngine.allocate()` + `AllocationResult` + `AllocationEngine.validate()` + `_copy_frame_graph_with_tolerances()` + `_damp_angular()`. Angular damping targets only indices (3,4,5); translation bounds unchanged. Locked constants: gamma=0.9, max_iter=10, n_validate=1000. Non-convergence message exactly: "Allocation could not converge to target budget".

**✅ B2-1 complete (commit `22f7a86`):** `AllocationEngine.solve()` initial implementation. *(Note: `AllocationObjective` ABC and `EqualAllocation` added here, then superseded and removed on 2026-06-28 by `LoosestAllocation`.)*

**✅ B1-7 complete (commit `04b5b05`):** `examples/pareto_sensitivity_example.py` — standalone example demonstrating all three Section 1.4 use cases: Sensitivity Pinpointing (Pareto breakdown via `compute_tolerance_sensitivities` + `to_ascii_chart()`), Component Selection (upgrade shoulder joint from ±3 mrad to ±1 mrad, compare Pareto rankings), Reporting (save frame report + 2 Pareto charts as PNG via `generate_frame_report` / `generate_sensitivity_report`). No new tests — example scripts only. **Milestone B-1 fully complete.**

**✅ B1-6 complete (commit `aacd210`):** `tests/test_physical_validation.py` — 3 module-level named regression tests:
- `test_rss_linear_stack_up`: 5-link normal-distribution translation chain; output variance matches classical RSS formula within quantified 5-SE sampling bound. 
- `test_sine_bar_lever_arm`: 1-mrad pivot + 100mm arm; `var(dy) = L²×var(rz)` within 1% rtol — validates the lever-arm cross-coupling that motivates the B-2 damping loop.
- `test_common_ancestor_cancellation`: 1m shared structural tolerance cancels completely from camera↔sample relative measurement; absolute frame confirms the shared error is non-trivially large.

**✅ B1-5 complete (commit `fda4e5c`):** `persistence/schema.py` (7 Pydantic v2 models + discriminated HTMInputModel union + ProjectModel cross-ref validator + frame_graph_to_project_model/project_model_to_frame_graph), `persistence/serializer.py` (ProjectLoadError + save_project + load_project). 40 new tests (test_schema.py + test_serializer.py). *(Originally named `io/`; renamed to `persistence/` 2026-06-27 to eliminate stdlib name collision — root `conftest.py` workaround deleted at that time.)*

**✅ B1-4 complete (commit `e258538`):** `postprocess/reporting.py` — 6 public functions (plot_histogram, plot_translation_projection, plot_rotation_summary, plot_pareto_sensitivity, generate_frame_report, generate_sensitivity_report). All return Axes/Figure; callers own show()/savefig(). 2D ellipsoid projection uses covariance-slice + eigh approach. First-order caveat annotation on Pareto chart is mandatory (locked). 17 new smoke tests.

**✅ B1-3 complete (commit `fd8a08b`):** `ParetoSensitivityReport` dataclass + `compute_tolerance_sensitivities(frame_graph, frame_a, frame_b)` appended to `postprocess/stats.py`. Uses `compute_sensitivity()` from `core/frame_graph.py` (no re-implementation). Variance formula: uniform→b²/3, normal→(b/k)². `trial_data` omitted from signature (tolerance specs live on `FrameGraph` edges, not `TrialData`). 8 new tests.

**✅ B1-2 complete (commit `ef4c89c`):** `postprocess/bounding_shapes.py` — 5 public functions (fit_bounding_box, fit_bounding_sphere, fit_bounding_ellipsoid, fit_rotation_cone, fit_rotation_box). Key implementation notes: coverage=1.0 uses uniform-scale approach (scales the covariance-shape ellipsoid uniformly — guarantees enclosure); coverage<1.0 uses chi2.ppf(df=3). Both rotation functions reject (N,4,4) pose arrays. 25 new tests.

**✅ Milestone A complete — all tasks pushed to origin/main:**
- A1: `core/transforms.py` + `core/conversions.py` — `a31218e` (26 tests)
- A2: `core/tolerance.py` + `core/sampling.py` — `3ac0eed` (21 tests)
- A3: `core/frame_graph.py` — `d81645c` (24 tests); also completes B1-1
- A4: `sim/monte_carlo_fk.py` — `744c562` (18 tests)
- A5: `postprocess/stats.py` — `019eb34` (26 tests)
- A6: `tests/conftest.py` + integration tests + allocation placeholder + README — `83b8ee3` (15 tests)
- A7: `examples/single_chain_fk_example.py` + `examples/multi_chain_shared_frame_example.py` — `3d7936d`, `8232c12` (0 new tests — example scripts only)

**Two math corrections made in A3 (documented in Section 11 changelog and Section 6.3):**
- Adjoint formula: `[[R, skew(t)@R],[0,R]]` — NOT `[[R,0],[skew(t)@R,R]]`
- Sensitivity formula: `J_i = Ad_{T_{frame_a→exit_i}}` — NOT `Ad_{T_{exit→frame_b}}`

