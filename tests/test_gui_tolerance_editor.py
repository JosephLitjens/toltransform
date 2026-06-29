"""
tests/test_gui_tolerance_editor.py — pytest-qt tests for gui/tolerance_editor/.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_tolerance_editor.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.tolerance_editor.tolerance_editor_widget import ToleranceEditorWidget
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _locked_zero() -> ToleranceSpecModel:
    return ToleranceSpecModel(distribution="uniform", bound=0.0, locked=True)


def _default_tol6() -> ToleranceSpec6Model:
    z = _locked_zero()
    return ToleranceSpec6Model(dx=z, dy=z, dz=z, rx=z, ry=z, rz=z)


def _make_tol6(bound: float, dist: str = "uniform",
               sigma: float = 3.0, locked: bool = False) -> ToleranceSpec6Model:
    s = ToleranceSpecModel(distribution=dist, bound=bound, sigma_level=sigma, locked=locked)
    return ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)


def _make_edge(name: str, parent: str, child: str,
               tol: ToleranceSpec6Model | None = None) -> HTMEdgeModel:
    return HTMEdgeModel(
        name=name, parent=parent, child=child,
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=tol or _default_tol6(),
    )


def _empty_project() -> ProjectModel:
    return ProjectModel(
        sim_settings=SimSettingsModel(mode="fk_verification", n_trials=100, seed=0)
    )


def _project_with_edge(edge_name: str = "e1",
                       tol: ToleranceSpec6Model | None = None) -> ProjectModel:
    p = _empty_project()
    p.frames.append(FrameModel(name="A"))
    p.frames.append(FrameModel(name="B"))
    p.edges.append(_make_edge(edge_name, "A", "B", tol))
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_tolerance_editor_shows_placeholder_with_empty_project(qtbot):
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(_empty_project())
    assert widget._stack.currentIndex() == 0


def test_tolerance_editor_auto_selects_first_edge_on_set_project(qtbot):
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    project = _project_with_edge()
    widget.set_project(project)
    # auto-selects first edge and shows the DoF panel immediately
    assert widget._stack.currentIndex() == 1
    assert widget._selected_edge_name == "e1"


def test_tolerance_editor_populates_fields_from_model(qtbot):
    tol6 = _make_tol6(0.005, dist="normal", sigma=2.0)
    project = _project_with_edge(tol=tol6)
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    assert widget._stack.currentIndex() == 1
    row = widget._rows[0]  # dx
    assert row.dist_combo.currentText() == "normal"
    assert row.bound_spin.value() == pytest.approx(0.005)
    assert row.sigma_spin.value() == pytest.approx(2.0)


def test_tolerance_editor_set_selected_edge_updates_combo(qtbot):
    project = _project_with_edge()
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")
    assert widget._edge_combo.currentText() == "e1"


def test_tolerance_editor_bound_change_updates_project(qtbot):
    project = _project_with_edge(tol=_make_tol6(0.001))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    widget._rows[0].bound_spin.setValue(0.005)  # dx bound

    assert project.edges[0].tolerance.dx.bound == pytest.approx(0.005)


def test_tolerance_editor_distribution_normal_enables_sigma(qtbot):
    project = _project_with_edge()
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    row = widget._rows[0]
    row.dist_combo.setCurrentText("normal")
    assert row.sigma_spin.isEnabled()


def test_tolerance_editor_distribution_uniform_disables_sigma(qtbot):
    project = _project_with_edge(tol=_make_tol6(0.001, dist="normal"))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    row = widget._rows[0]
    assert row.sigma_spin.isEnabled()  # starts as normal
    row.dist_combo.setCurrentText("uniform")
    assert not row.sigma_spin.isEnabled()


def test_tolerance_editor_locked_toggle_updates_model(qtbot):
    project = _project_with_edge(tol=_make_tol6(0.001, locked=False))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    row = widget._rows[3]  # rx
    assert not project.edges[0].tolerance.rx.locked
    row.locked_check.setChecked(True)
    assert project.edges[0].tolerance.rx.locked


def test_tolerance_editor_sigma_level_change_updates_model(qtbot):
    project = _project_with_edge(tol=_make_tol6(0.001, dist="normal", sigma=3.0))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    row = widget._rows[0]  # dx
    row.sigma_spin.setValue(2.5)

    assert project.edges[0].tolerance.dx.sigma_level == pytest.approx(2.5)


def test_tolerance_editor_emits_project_changed(qtbot):
    project = _project_with_edge(tol=_make_tol6(0.001))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    with qtbot.waitSignal(widget.project_changed, timeout=500):
        widget._rows[0].bound_spin.setValue(0.01)


def test_tolerance_editor_bulk_apply_to_edge(qtbot):
    project = _project_with_edge(tol=_make_tol6(0.001))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    widget._bulk_bound_spin.setValue(0.01)
    widget._bulk_dist_combo.setCurrentText("uniform")
    widget._apply_edge_btn.click()

    tol6 = project.edges[0].tolerance
    for dof in ("dx", "dy", "dz", "rx", "ry", "rz"):
        assert getattr(tol6, dof).bound == pytest.approx(0.01), f"{dof} bound mismatch"
        assert getattr(tol6, dof).distribution == "uniform"


def test_tolerance_editor_bulk_apply_to_all_edges(qtbot):
    project = _empty_project()
    project.frames.extend([FrameModel(name=n) for n in ("A", "B", "C")])
    project.edges.append(_make_edge("e1", "A", "B", _make_tol6(0.001)))
    project.edges.append(_make_edge("e2", "B", "C", _make_tol6(0.002)))

    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")

    widget._bulk_bound_spin.setValue(0.02)
    widget._apply_all_btn.click()

    for edge in project.edges:
        for dof in ("dx", "dy", "dz", "rx", "ry", "rz"):
            assert getattr(edge.tolerance, dof).bound == pytest.approx(0.02), \
                f"edge {edge.name} {dof} bound mismatch"


def test_tolerance_editor_set_project_clears_selection(qtbot):
    project = _project_with_edge()
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.set_selected_edge("e1")
    assert widget._stack.currentIndex() == 1

    # Replacing the project clears selection → placeholder
    widget.set_project(_empty_project())
    assert widget._stack.currentIndex() == 0
    assert widget._selected_edge_name is None


# ── Asymmetric tolerance mode toggle ─────────────────────────────────────────

def _asym_tol6(lo: float, hi: float, dist: str = "uniform") -> ToleranceSpec6Model:
    """Build a ToleranceSpec6Model where all 6 DoF are asymmetric lo..hi."""
    s = ToleranceSpecModel(
        distribution=dist,
        bound=max(abs(lo), abs(hi)),
        lower=lo,
        upper=hi,
    )
    return ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)


def test_toggle_to_asymmetric_mode_shows_lower_upper(qtbot):
    """Clicking mode button switches _bound_stack to page 1 (lower/upper widgets)."""
    project = _project_with_edge()
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    assert not row.is_asymmetric
    assert row._bound_stack.currentIndex() == 0

    row.mode_btn.click()  # toggle to asymmetric

    assert row.is_asymmetric
    assert row._bound_stack.currentIndex() == 1
    assert row.mode_btn.text() == "↔"


def test_toggle_to_asymmetric_pre_populates_lower_upper(qtbot):
    """When switching to asymmetric mode, lower/upper are pre-filled as ±bound."""
    bound = 0.005
    project = _project_with_edge(tol=_make_tol6(bound))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    row.mode_btn.click()  # switch to asymmetric

    import pytest
    assert row.lower_spin.value() == pytest.approx(-bound)
    assert row.upper_spin.value() == pytest.approx(bound)


def test_toggle_back_to_symmetric_updates_bound(qtbot):
    """Switching back from asymmetric sets bound = max(|lower|, |upper|)."""
    project = _project_with_edge()
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    row.mode_btn.click()           # → asymmetric
    row.lower_spin.setValue(-0.003)
    row.upper_spin.setValue(0.008)
    row.mode_btn.click()           # → symmetric

    import pytest
    assert not row.is_asymmetric
    assert row._bound_stack.currentIndex() == 0
    assert row.bound_spin.value() == pytest.approx(0.008)  # max(0.003, 0.008)
    assert row.mode_btn.text() == "±"


def test_load_asymmetric_spec_enters_asymmetric_mode(qtbot):
    """Loading a ToleranceSpec6Model with lower/upper sets the row to ↔ mode."""
    lo, hi = -0.002, 0.006
    project = _project_with_edge(tol=_asym_tol6(lo, hi))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    assert row.is_asymmetric
    assert row._bound_stack.currentIndex() == 1

    import pytest
    assert row.lower_spin.value() == pytest.approx(lo)
    assert row.upper_spin.value() == pytest.approx(hi)


def test_get_model_returns_asymmetric_spec(qtbot):
    """get_model() in asymmetric mode returns model with lower/upper set."""
    lo, hi = -0.001, 0.004
    project = _project_with_edge(tol=_asym_tol6(lo, hi))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    model = row.get_model()

    import pytest
    assert model.lower == pytest.approx(lo)
    assert model.upper == pytest.approx(hi)
    assert model.bound == pytest.approx(max(abs(lo), abs(hi)))


def test_asymmetric_field_change_writes_back_to_project(qtbot):
    """Changing lower/upper spinboxes in asymmetric mode updates the ProjectModel."""
    lo, hi = -0.002, 0.005
    project = _project_with_edge(tol=_asym_tol6(lo, hi))
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    new_hi = 0.009
    row.upper_spin.setValue(new_hi)
    # Trigger write-back (simulate valueChanged → _on_field_changed)
    widget._on_field_changed()

    import pytest
    stored = project.edges[0].tolerance.dx
    assert stored.upper == pytest.approx(new_hi)
    assert stored.lower == pytest.approx(lo)


def test_is_valid_returns_false_for_equal_lower_upper(qtbot):
    """lower == upper is invalid — is_valid() should return False."""
    project = _project_with_edge()
    widget = ToleranceEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    row = widget._rows[0]
    row.mode_btn.click()  # → asymmetric
    row.lower_spin.setValue(0.005)
    row.upper_spin.setValue(0.005)

    assert not row.is_valid()
