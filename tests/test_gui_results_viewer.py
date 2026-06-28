"""
tests/test_gui_results_viewer.py — pytest-qt tests for gui/results_viewer/.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_results_viewer.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.results_viewer.results_viewer_widget import ResultsViewerWidget
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
)
from sim.allocation import AllocationEngine, AllocationResult
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData
from core.tolerance import ToleranceSpec, ToleranceSpec6
from persistence.schema import project_model_to_frame_graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uniform_spec(bound: float = 0.001) -> ToleranceSpecModel:
    return ToleranceSpecModel(distribution="uniform", bound=bound)


def _tol6(bound: float = 0.001) -> ToleranceSpec6Model:
    s = _uniform_spec(bound)
    return ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)


def _make_project() -> ProjectModel:
    p = ProjectModel(sim_settings=SimSettingsModel(mode="fk_verification", n_trials=50, seed=0))
    p.frames.append(FrameModel(name="A"))
    p.frames.append(FrameModel(name="B"))
    p.edges.append(HTMEdgeModel(
        name="e1", parent="A", child="B",
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_tol6(0.001),
    ))
    return p


def _make_trial_data() -> TrialData:
    project = _make_project()
    fg = project_model_to_frame_graph(project)
    return MonteCarloFKEngine.run(fg, n_trials=50, seed=0)


def _make_alloc_result() -> AllocationResult:
    project = _make_project()
    fg = project_model_to_frame_graph(project)
    target = ToleranceSpec6(
        ToleranceSpec("uniform", 0.1),
        ToleranceSpec("uniform", 0.1),
        ToleranceSpec("uniform", 0.1),
        ToleranceSpec("uniform", 0.1),
        ToleranceSpec("uniform", 0.1),
        ToleranceSpec("uniform", 0.1),
    )
    return AllocationEngine.allocate(fg, "A", "B", target, seed=0, n_validate=50)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_results_viewer_shows_placeholder_initially(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)

    assert widget._stack.currentIndex() == 0


def test_results_viewer_clear_resets_to_placeholder(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    td = _make_trial_data()
    widget.set_result(td, project)

    assert widget._stack.currentIndex() == 1  # FK page
    widget.clear()
    assert widget._stack.currentIndex() == 0


def test_results_viewer_fk_shows_fk_page(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    td = _make_trial_data()
    widget.set_result(td, _make_project())

    assert widget._stack.currentIndex() == 1


def test_results_viewer_fk_frame_combo_populated(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    td = _make_trial_data()
    widget.set_result(td, _make_project())

    frame_names = list(td.frame_poses.keys())
    combo_names = [widget._frame_combo.itemText(i) for i in range(widget._frame_combo.count())]
    assert combo_names == frame_names


def test_results_viewer_fk_envelope_table_has_dof_rows(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_trial_data(), _make_project())

    table = widget._envelope_table
    assert table.rowCount() == 6
    dof_labels = [table.item(r, 0).text() for r in range(6)]
    assert dof_labels == ["dx", "dy", "dz", "rx", "ry", "rz"]


def test_results_viewer_fk_envelope_values_match_stats(qtbot):
    from postprocess.stats import frame_envelope_box

    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    td = _make_trial_data()
    widget.set_result(td, _make_project())

    frame_name = widget._frame_combo.currentText()
    expected = frame_envelope_box(td, frame_name)
    table = widget._envelope_table

    for row, dof in enumerate(["dx", "dy", "dz", "rx", "ry", "rz"]):
        min_val = float(table.item(row, 1).text())
        max_val = float(table.item(row, 2).text())
        assert abs(min_val - expected[dof]["min"]) < 1e-9
        assert abs(max_val - expected[dof]["max"]) < 1e-9


def test_results_viewer_fk_view_button_enabled(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_trial_data(), _make_project())

    assert widget._view_report_btn.isEnabled()


def test_results_viewer_ik_shows_ik_page(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_alloc_result(), _make_project())

    assert widget._stack.currentIndex() == 2


def test_results_viewer_ik_status_label_converged(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    result = _make_alloc_result()
    widget.set_result(result, _make_project())

    text = widget._ik_status_label.text()
    if result.converged:
        assert "✓" in text or "Converged" in text or "passed" in text
    else:
        assert "✗" in text


def test_results_viewer_ik_alloc_table_has_edge_rows(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    result = _make_alloc_result()
    widget.set_result(result, _make_project())

    assert widget._alloc_table.rowCount() == len(result.corrected_allocation)


def test_results_viewer_ik_achieved_table_has_dof_rows(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_alloc_result(), _make_project())

    assert widget._achieved_table.rowCount() == 6
