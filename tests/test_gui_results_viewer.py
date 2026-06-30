"""
tests/test_gui_results_viewer.py — pytest-qt tests for gui/results_viewer/.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_results_viewer.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.results_viewer.results_viewer_widget import ResultsViewerWidget
from helpers import _tol6
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SimSettingsModel,
    project_model_to_frame_graph,
)
from sim.allocation import AllocationEngine, AllocationResult
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData
from core.tolerance import ToleranceSpec, ToleranceSpec6


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


def test_results_viewer_ik_per_pair_section_populated(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_alloc_result(), _make_project())

    # At least one per-pair group box should be added to the per-pair container.
    assert widget._per_pair_layout.count() >= 1


# ── Apply Allocation tests ─────────────────────────────────────────────────────

def test_results_viewer_apply_btn_present(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    assert widget._apply_btn is not None


def test_results_viewer_apply_btn_disabled_initially(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    assert not widget._apply_btn.isEnabled()


def test_results_viewer_apply_btn_disabled_on_clear(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_alloc_result(), _make_project())
    widget.clear()
    assert not widget._apply_btn.isEnabled()


def test_results_viewer_apply_btn_state_matches_convergence(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    result = _make_alloc_result()
    widget.set_result(result, _make_project())
    assert widget._apply_btn.isEnabled() == result.converged


def test_results_viewer_apply_btn_disabled_for_fk_result(qtbot):
    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    widget.set_result(_make_trial_data(), _make_project())
    assert not widget._apply_btn.isEnabled()


def test_results_viewer_apply_writes_bounds_to_project(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
    )

    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    result = _make_alloc_result()
    widget.set_result(result, project)

    assert result.converged, "Allocation did not converge; apply is disabled"

    widget._apply_btn.click()

    # Verify corrected bounds were written to the matching project edges.
    changed = False
    for e in project.edges:
        if e.name not in result.corrected_allocation:
            continue
        tol6 = result.corrected_allocation[e.name]
        for dof in ("dx", "dy", "dz", "rx", "ry", "rz"):
            spec = getattr(tol6, dof)
            if not spec.locked:
                new_bound = getattr(e.tolerance, dof).bound
                if abs(new_bound - spec.bound) < 1e-9:
                    changed = True
    assert changed, "No edge bounds were updated by Apply"


def test_results_viewer_apply_emits_project_changed(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
    )

    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    result = _make_alloc_result()
    widget.set_result(result, _make_project())

    assert result.converged, "Allocation did not converge; apply is disabled"

    with qtbot.waitSignal(widget.project_changed, timeout=500):
        widget._apply_btn.click()


def test_results_viewer_apply_cancel_does_not_modify_project(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Cancel),
    )

    widget = ResultsViewerWidget()
    qtbot.addWidget(widget)
    project = _make_project()
    result = _make_alloc_result()
    widget.set_result(result, project)

    assert result.converged, "Allocation did not converge; apply is disabled"

    original_bounds = {
        e.name: {dof: getattr(e.tolerance, dof).bound for dof in ("dx", "dy", "dz", "rx", "ry", "rz")}
        for e in project.edges
    }

    fired = []
    widget.project_changed.connect(lambda: fired.append(True))
    widget._apply_btn.click()

    assert len(fired) == 0
    for e in project.edges:
        for dof in ("dx", "dy", "dz", "rx", "ry", "rz"):
            assert getattr(e.tolerance, dof).bound == pytest.approx(
                original_bounds[e.name][dof]
            )
