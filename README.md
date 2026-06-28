# TolTransform

A kinematic error-budgeting tool for precision machine design.

**Start here:** `docs/design_spec.md` — full architecture, conventions, and project plan.

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
\`\`\`

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

All tests must pass before beginning GUI work (Section 10 of the design spec). Run this at the start of every session to confirm nothing has silently broken.

## Status
Milestones A–D complete. 311 tests passing.

**IK Allocation features:** Statistical (RSS) and Worst-Case allocation methods; user-configurable max iterations; Target ± column in results; binary-search angular refinement for maximum possible bounds.
