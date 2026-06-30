"""
tests/test_gui_main_window.py — MainWindow cross-panel integration tests (C-6).

Tests that signals emitted by one panel (or by MainWindow itself) reach all
downstream panels correctly. None of these tests start the background worker —
they call MainWindow's internal handlers directly.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_main_window.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.main_window import MainWindow, _empty_project
from helpers import _tol6
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SimSettingsModel,
    project_model_to_frame_graph,
)
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData


def _make_project_with_edge() -> ProjectModel:
    p = ProjectModel(sim_settings=SimSettingsModel(mode="fk_verification", n_trials=50, seed=0))
    p.frames.append(FrameModel(name="A"))
    p.frames.append(FrameModel(name="B"))
    p.edges.append(HTMEdgeModel(
        name="e1", parent="A", child="B",
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_tol6(0.001),
    ))
    return p


def _make_trial_data(project: ProjectModel) -> TrialData:
    fg = project_model_to_frame_graph(project)
    return MonteCarloFKEngine.run(fg, n_trials=50, seed=0)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_fk_result_reaches_results_viewer(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = _make_project_with_edge()
    window._project = project
    window._results_viewer.clear()

    td = _make_trial_data(project)
    window._on_run_completed(td)

    assert window._results_viewer._stack.currentIndex() == 1  # FK page


def test_run_fk_result_reaches_point_pair_panel(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = _make_project_with_edge()
    window._project = project

    td = _make_trial_data(project)
    window._on_run_completed(td)

    assert window._point_pair_panel._trial_data is not None


def test_graph_change_refreshes_run_panel_combos(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    # Frame combos live inside the constraint rows, not directly on RunPanelWidget.
    row = window._run_panel._constraint_rows[0]
    initial_count = row._frame_a_combo.count()
    window._project.frames.append(FrameModel(name="NewFrame"))
    window._on_graph_editor_changed()

    assert row._frame_a_combo.count() == initial_count + 1


def test_graph_change_refreshes_point_pair_combos(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    initial_count = window._point_pair_panel._frame_a_combo.count()
    window._project.frames.append(FrameModel(name="NewFrame"))
    window._on_graph_editor_changed()

    assert window._point_pair_panel._frame_a_combo.count() == initial_count + 1


def test_new_project_resets_results_viewer(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = _make_project_with_edge()
    window._project = project
    td = _make_trial_data(project)
    window._on_run_completed(td)
    assert window._results_viewer._stack.currentIndex() == 1  # sanity check

    window._new_project()

    assert window._results_viewer._stack.currentIndex() == 0  # back to placeholder


def test_new_project_clears_point_pair_trial_data(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = _make_project_with_edge()
    window._project = project
    td = _make_trial_data(project)
    window._on_run_completed(td)
    assert window._point_pair_panel._trial_data is not None  # sanity check

    window._new_project()

    assert window._point_pair_panel._trial_data is None


def test_run_failure_does_not_change_results_viewer(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._results_viewer._stack.currentIndex() == 0  # starts on placeholder

    window._on_run_failed("Simulated engine error")

    assert window._results_viewer._stack.currentIndex() == 0  # still on placeholder
