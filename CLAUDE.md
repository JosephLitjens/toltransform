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
- **GUI code talks only to `io.schema` models, never directly to `core`/`sim` objects** (Section 5.3). This rule applies starting Milestone C — not relevant yet in Milestones A/B-1/B-2.

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

- **Current milestone:** B-1.
- **Last completed task:** B1-3 — `ParetoSensitivityReport` + `compute_tolerance_sensitivities()` appended to `postprocess/stats.py` (commit `fd8a08b`). Suite: **163 passed, 1 skipped**.
- **Next task:** B1-4 — `postprocess/reporting.py` (plotting layer). Read Section 6.10 of the design spec before starting.

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

**Key facts for the next session (B-1):**
- `postprocess/stats.py` Steps 8–9 (Pareto sensitivity, `ParetoSensitivityReport`) are NOT yet implemented — deferred to B1-3.
- `postprocess/reporting.py` exists as an empty stub — to be implemented in B1-4.
- `postprocess/bounding_shapes.py` exists as an empty stub — to be implemented in B1-2.
- `sim/allocation.py` does NOT exist yet — B-2 scope.
- B1-1 (adjoint + sensitivity primitives in `core/frame_graph.py`) was completed early during A3. Do not re-implement.
