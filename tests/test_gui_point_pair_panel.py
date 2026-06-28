"""
tests/test_gui_point_pair_panel.py — headless tests for gui/point_pair_panel/.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_point_pair_panel.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.point_pair_panel.point_pair_panel_widget import PointPairPanelWidget
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SavedAnalysisModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
    project_model_to_frame_graph,
)
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uniform_spec(bound: float = 0.001) -> ToleranceSpecModel:
    return ToleranceSpecModel(distribution="uniform", bound=bound)


def _tol6(bound: float = 0.001) -> ToleranceSpec6Model:
    s = _uniform_spec(bound)
    return ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)


def _make_project() -> ProjectModel:
    """2-frame (A, B) project with one edge A→B."""
    p = ProjectModel(sim_settings=SimSettingsModel(mode="fk_verification", n_trials=50, seed=0))
    p.frames.append(FrameModel(name="A"))
    p.frames.append(FrameModel(name="B"))
    p.edges.append(HTMEdgeModel(
        name="e1", parent="A", child="B",
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_tol6(0.001),
    ))
    return p


def _make_disconnected_project() -> ProjectModel:
    """4-frame project: A-B connected, C-D isolated (no edge between components)."""
    p = ProjectModel(sim_settings=SimSettingsModel(mode="fk_verification", n_trials=50, seed=0))
    for name in ("A", "B", "C", "D"):
        p.frames.append(FrameModel(name=name))
    p.edges.append(HTMEdgeModel(
        name="e1", parent="A", child="B",
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_tol6(0.001),
    ))
    p.edges.append(HTMEdgeModel(
        name="e2", parent="C", child="D",
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_tol6(0.001),
    ))
    return p


def _make_trial_data(project: ProjectModel) -> TrialData:
    fg = project_model_to_frame_graph(project)
    return MonteCarloFKEngine.run(fg, n_trials=50, seed=0)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pp_no_result_shows_placeholder(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    widget.set_project(_make_project())

    # isHidden() checks the widget's own flag, not the ancestor chain (offscreen safe)
    assert widget._envelope_table.isHidden()
    assert not widget._envelope_placeholder.isHidden()


def test_pp_set_project_populates_combos(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    widget.set_project(project)

    assert widget._frame_a_combo.count() == len(project.frames)
    assert widget._frame_b_combo.count() == len(project.frames)


def test_pp_connected_frames_no_warning(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    widget.set_project(_make_project())

    # A and B are connected (default selection after set_project)
    assert widget._connectivity_label.text() == ""


def test_pp_disjoint_frames_shows_warning(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_disconnected_project()
    widget.set_project(project)

    # Select A and C (different components)
    idx_a = widget._frame_a_combo.findText("A")
    idx_c = widget._frame_b_combo.findText("C")
    widget._frame_a_combo.setCurrentIndex(idx_a)
    widget._frame_b_combo.setCurrentIndex(idx_c)

    assert widget._connectivity_label.text() != ""


def test_pp_name_autopopulated(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    widget.set_project(_make_project())

    frame_a = widget._frame_a_combo.currentText()
    frame_b = widget._frame_b_combo.currentText()
    assert widget._name_edit.text() == f"{frame_a} → {frame_b}"


def test_pp_save_adds_to_project(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    widget.set_project(project)

    widget._name_edit.setText("My Analysis")
    widget._save_btn.click()

    assert len(project.saved_analyses) == 1
    assert project.saved_analyses[0].name == "My Analysis"
    assert project.saved_analyses[0].frame_a == widget._frame_a_combo.currentText()
    assert project.saved_analyses[0].frame_b == widget._frame_b_combo.currentText()


def test_pp_save_emits_project_changed(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    widget.set_project(_make_project())

    widget._name_edit.setText("My Analysis")
    with qtbot.waitSignal(widget.project_changed, timeout=1000):
        widget._save_btn.click()


def test_pp_duplicate_name_rejected(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    widget.set_project(project)

    widget._name_edit.setText("Dup")
    widget._save_btn.click()
    assert len(project.saved_analyses) == 1

    widget._name_edit.setText("Dup")
    widget._save_btn.click()
    assert len(project.saved_analyses) == 1  # not added again
    assert widget._save_error_label.text() != ""


def test_pp_select_saved_loads_combos(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    project.saved_analyses.append(SavedAnalysisModel(name="P1", frame_a="B", frame_b="A"))
    widget.set_project(project)

    # Click the first saved analysis row
    widget._saved_list.setCurrentRow(0)

    assert widget._frame_a_combo.currentText() == "B"
    assert widget._frame_b_combo.currentText() == "A"


def test_pp_delete_removes_from_project(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    widget.set_project(project)

    widget._name_edit.setText("ToDelete")
    widget._save_btn.click()
    assert len(project.saved_analyses) == 1

    widget._saved_list.setCurrentRow(0)
    signals = []
    widget.project_changed.connect(lambda: signals.append(1))
    widget._delete_btn.click()

    assert len(project.saved_analyses) == 0
    assert len(signals) >= 1


def test_pp_fk_result_shows_envelope(qtbot):
    widget = PointPairPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    widget.set_project(project)

    td = _make_trial_data(project)
    widget.set_result(td)

    assert not widget._envelope_table.isHidden()
    assert widget._envelope_table.rowCount() == 6
    dof_labels = [widget._envelope_table.item(r, 0).text() for r in range(6)]
    assert dof_labels == ["dx", "dy", "dz", "rx", "ry", "rz"]
