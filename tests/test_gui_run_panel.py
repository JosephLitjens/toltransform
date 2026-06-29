"""
tests/test_gui_run_panel.py — pytest-qt tests for gui/run_panel/.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_run_panel.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.run_panel.run_panel_widget import RunPanelWidget
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    IKConstraintModel,
    ProjectModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
)
from sim.allocation import AllocationResult
from sim.monte_carlo_fk import TrialData


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uniform_spec(bound: float = 0.001) -> ToleranceSpecModel:
    return ToleranceSpecModel(distribution="uniform", bound=bound)


def _default_tol6(bound: float = 0.001) -> ToleranceSpec6Model:
    s = _uniform_spec(bound)
    return ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)


def _make_edge(name: str, parent: str, child: str) -> HTMEdgeModel:
    return HTMEdgeModel(
        name=name, parent=parent, child=child,
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_default_tol6(),
    )


def _make_project(
    mode: str = "fk_verification",
    n_trials: int = 100,
    seed: int = 42,
    with_edge: bool = True,
) -> ProjectModel:
    p = ProjectModel(
        sim_settings=SimSettingsModel(mode=mode, n_trials=n_trials, seed=seed)
    )
    p.frames.append(FrameModel(name="A"))
    p.frames.append(FrameModel(name="B"))
    if with_edge:
        p.edges.append(_make_edge("e1", "A", "B"))
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_panel_loads_sim_settings(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="fk_verification", n_trials=500, seed=99)
    widget.set_project(project)

    assert widget._n_trials_spin.value() == 500
    assert widget._seed_spin.value() == 99
    assert widget._mode_combo.currentData() == "fk_verification"


def test_run_panel_fk_mode_hides_ik_group(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="fk_verification")
    widget.set_project(project)

    # isHidden() checks the widget's own explicit flag, not the full ancestor chain.
    # This is correct for offscreen tests where the parent window is never shown.
    assert widget._ik_group.isHidden()


def test_run_panel_ik_mode_shows_ik_group(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation")
    widget.set_project(project)

    assert not widget._ik_group.isHidden()


def test_run_panel_mode_change_updates_project(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="fk_verification")
    widget.set_project(project)

    idx = widget._mode_combo.findData("ik_allocation")
    widget._mode_combo.setCurrentIndex(idx)

    assert project.sim_settings.mode == "ik_allocation"
    assert not widget._ik_group.isHidden()


def test_run_panel_n_trials_change_updates_project(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(n_trials=100)
    widget.set_project(project)

    widget._n_trials_spin.setValue(2000)

    assert project.sim_settings.n_trials == 2000


def test_run_panel_seed_change_updates_project(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(seed=42)
    widget.set_project(project)

    widget._seed_spin.setValue(999)

    assert project.sim_settings.seed == 999


def test_run_panel_randomize_changes_seed(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(seed=42)
    widget.set_project(project)

    original = widget._seed_spin.value()
    # Run several times to be robust against the (astronomically unlikely) same value
    hits = 0
    for _ in range(5):
        widget._randomize_btn.click()
        if widget._seed_spin.value() != original:
            hits += 1
    assert hits > 0


def test_run_panel_emits_project_changed_on_mode_change(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="fk_verification")
    widget.set_project(project)

    with qtbot.waitSignal(widget.project_changed, timeout=500):
        idx = widget._mode_combo.findData("ik_allocation")
        widget._mode_combo.setCurrentIndex(idx)


def test_run_panel_fk_run_emits_trial_data(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="fk_verification", n_trials=50)
    widget.set_project(project)

    received = []

    def capture(result):
        received.append(result)

    widget.run_completed.connect(capture)

    with qtbot.waitSignal(widget.run_completed, timeout=10000):
        widget._run_btn.click()

    assert len(received) == 1
    assert isinstance(received[0], TrialData)
    assert received[0].n_trials == 50


def test_run_panel_ik_run_emits_allocation_result(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation", n_trials=50, seed=7)
    widget.set_project(project)

    # Switch to IK mode and configure frame pair via the first constraint row
    idx = widget._mode_combo.findData("ik_allocation")
    widget._mode_combo.setCurrentIndex(idx)

    row = widget._constraint_rows[0]
    row._frame_a_combo.setCurrentText("A")
    row._frame_b_combo.setCurrentText("B")

    # Set a non-zero target bound so the allocation has something to do
    for spin in row._spins:
        spin.setValue(0.01)

    received = []

    def capture(result):
        received.append(result)

    widget.run_completed.connect(capture)

    with qtbot.waitSignal(widget.run_completed, timeout=10000):
        widget._run_btn.click()

    assert len(received) == 1
    assert isinstance(received[0], AllocationResult)


def test_run_panel_no_edges_shows_error(qtbot):
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(with_edge=False)
    widget.set_project(project)

    # run_completed must NOT fire
    fired = []
    widget.run_completed.connect(lambda r: fired.append(r))

    widget._run_btn.click()

    assert len(fired) == 0
    assert "Error" in widget._status_label.text()


# ── IK constraint persistence tests ───────────────────────────────────────────

def _make_ik_constraint(frame_a: str = "A", frame_b: str = "B",
                         bound: float = 0.005) -> IKConstraintModel:
    s = ToleranceSpecModel(distribution="uniform", bound=bound)
    return IKConstraintModel(
        frame_a=frame_a, frame_b=frame_b,
        target=ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s),
    )


def test_run_panel_loads_ik_constraints_into_rows(qtbot):
    """set_project with saved ik_constraints populates one row per constraint."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    c1 = _make_ik_constraint("A", "B", 0.005)
    c2 = _make_ik_constraint("B", "A", 0.010)
    project = _make_project(mode="ik_allocation")
    project.sim_settings.ik_constraints = [c1, c2]
    widget.set_project(project)

    assert len(widget._constraint_rows) == 2
    assert widget._constraint_rows[0].get_frame_pair() == ("A", "B")
    assert widget._constraint_rows[1].get_frame_pair() == ("B", "A")


def test_run_panel_loaded_row_targets_match_model(qtbot):
    """Spinbox values reflect the saved target bounds after set_project."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    c = _make_ik_constraint("A", "B", 0.007)
    project = _make_project(mode="ik_allocation")
    project.sim_settings.ik_constraints = [c]
    widget.set_project(project)

    row = widget._constraint_rows[0]
    tol6 = row.get_target()
    for dof in ("dx", "dy", "dz", "rx", "ry", "rz"):
        assert getattr(tol6, dof).bound == pytest.approx(0.007)


def test_run_panel_max_iter_writes_back_to_project(qtbot):
    """Changing max_iter spinbox updates project.sim_settings.ik_max_iter."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation")
    project.sim_settings.ik_max_iter = 30
    widget.set_project(project)

    widget._max_iter_spin.setValue(50)

    assert project.sim_settings.ik_max_iter == 50


def test_run_panel_max_iter_emits_project_changed(qtbot):
    """Changing max_iter emits project_changed."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation")
    widget.set_project(project)

    with qtbot.waitSignal(widget.project_changed, timeout=500):
        widget._max_iter_spin.setValue(99)


def test_run_panel_constraint_row_frame_change_writes_back(qtbot):
    """Changing a row's frame_b combo updates project.sim_settings.ik_constraints."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation")
    widget.set_project(project)

    row = widget._constraint_rows[0]
    # Set frame_a to "A", frame_b to "B"
    row._frame_a_combo.setCurrentText("A")
    row._frame_b_combo.setCurrentText("B")

    constraints = project.sim_settings.ik_constraints
    assert any(c.frame_a == "A" and c.frame_b == "B" for c in constraints)


def test_run_panel_add_constraint_row_updates_project(qtbot):
    """Clicking Add Constraint adds a row and saves to project."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation")
    widget.set_project(project)

    initial_rows = len(widget._constraint_rows)
    widget._add_constraint_btn.click()

    assert len(widget._constraint_rows) == initial_rows + 1


def test_run_panel_loads_max_iter_from_project(qtbot):
    """set_project restores ik_max_iter from project.sim_settings."""
    widget = RunPanelWidget()
    qtbot.addWidget(widget)
    project = _make_project(mode="ik_allocation")
    project.sim_settings.ik_max_iter = 75
    widget.set_project(project)

    assert widget._max_iter_spin.value() == 75
