"""
Tests for sim/allocation.py (Milestone B-2).

This file exists now to house the required named placeholder for Section 9 Item 4
(per Section 6.20 Granular Task List Item 4). Real tests will be added when
sim/allocation.py is implemented in Milestone B-2.
"""

import pytest


@pytest.mark.skip(reason="sim/allocation.py not yet implemented — Milestone B-2 task")
def test_allocation_mc_validation_discrepancy():
    """Section 9, Item 4 — allocation MC-validation-pass discrepancy check.

    A module-level function (not inside any class) so it is findable by name
    via `pytest -k test_allocation_mc_validation_discrepancy`.

    Property under test: AllocationEngine.allocate() must include a forward MC
    validation pass that measures the true achieved envelope and reports the gap
    between the linear estimate and the MC-measured result. This discrepancy is
    the primary diagnostic that tells the user whether the small-angle adjoint
    linearization is accurate enough for their chain geometry.

    To be implemented in Milestone B-2 when sim/allocation.py exists.
    See Section 3.2 Step 3 and Section 6.7 for the full specification.
    """
    raise NotImplementedError
