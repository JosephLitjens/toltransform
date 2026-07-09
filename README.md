# TolTransform

A kinematic error-budgeting tool for precision machine design.

TolTransform models an assembly as a graph of coordinate frames connected by toleranced rigid transforms, then answers two questions:

- **Forward verification (FK):** given the tolerances everywhere, how much positional and angular error shows up at any point, worst-case or statistically?
- **Inverse allocation (IK):** given a target error budget, how loose can each interface's tolerance be while still guaranteeing that budget?

Built for mechanical, optomechanical, and precision engineers working on instrument alignment, optical benches, machine frames, and similar rigid-body stacks.

## Features

- **Monte Carlo FK engine**: exact (non-linearized) propagation of sampled perturbations, reproducible per-edge random streams, shared upstream interfaces sampled once per trial so common-base errors cancel correctly in relative measurements.
- **Inverse tolerance allocation**: log-sum convex optimization (never returns a zero tolerance), multi-constraint targets, and an iterative Monte Carlo correction loop with binary-search angular refinement.
- **Pareto sensitivity ranking**: closed-form variance decomposition showing which edge/DoF dominates the error between any two frames.
- **Bounding shapes**: worst-case envelopes, percentile tables, Mahalanobis confidence ellipsoids, rotation cones.
- **Point-pair analysis**: saved relative-alignment queries between any two connected frames, re-evaluated on every run.
- **Desktop GUI** (PySide6): graph editor, per-DoF tolerance editor, run panel, results viewer, and a live 3D frame viewer with Monte Carlo point-cloud display.
- **Fully scriptable**: everything the GUI does is available headless; see `examples/`.

## Download

Prebuilt standalone apps for macOS and Windows are attached to [GitHub Releases](../../releases). Unzip and launch; no Python installation required.

## Documentation

- `docs/TolTransform_Manual.pdf` (also `.docx`): theory and user manual. Part 1 covers the math (screw-theory perturbation model, adjoint sensitivity, convex allocation) at a grad-course level with full derivations in the appendices; Part 3 is the GUI guide; Part 4 walks the examples.
- `docs/design_spec.md`: the living engineering document. Architecture, binding conventions, module specifications, changelog. Start here if you are working on the code.

## Setup from Source

Requires Python 3.12.

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

Launch the GUI:

```bash
python main.py
```

## Examples

Four self-contained scripts demonstrate the engine without the GUI:

```bash
python examples/single_chain_fk_example.py          # basic FK, lever-arm amplification
python examples/multi_chain_shared_frame_example.py # shared-base cancellation, 4 scenarios
python examples/allocation_example.py               # inverse allocation end-to-end
python examples/pareto_sensitivity_example.py       # sensitivity ranking and reporting
```

## Running Tests

```bash
python -m pytest tests/ -q
```

Run this at the start of every working session to confirm nothing has silently broken.

## Status

Milestones A through D complete, plus E-1 (standalone packaging, v1.0.0 released for macOS and Windows). 373 tests passing.

## Scope

TolTransform models rigid bodies with small (sub-degree) perturbations in open kinematic chains. It does not model compliance, thermal effects, correlated errors, closed loops, or GD&T datum semantics; see the manual's scope section (§1.8) before applying it to a problem outside that envelope.
