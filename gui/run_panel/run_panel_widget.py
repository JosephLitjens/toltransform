"""
gui/run_panel/run_panel_widget.py — Simulation execution panel (Section 6.16).

This is the ONE place where persistence.schema objects are converted into live
core/sim objects (Section 5.3): project_model_to_frame_graph() is called on the
main thread immediately before starting the background worker. The worker owns
the FrameGraph and never touches the ProjectModel.

Modes:
  FK Verification — MonteCarloFKEngine.run() → TrialData
  IK Allocation   — AllocationEngine.allocate() → AllocationResult

SimSettings (mode, n_trials, seed) are written back to project.sim_settings on
every change so they persist with the project file.
"""

from __future__ import annotations

import random

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.tolerance import ToleranceSpec, ToleranceSpec6
from persistence.schema import ProjectModel, project_model_to_frame_graph
from sim.allocation import AllocationEngine, AllocationResult, EqualAllocation, RSSAllocation
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData

_DOF_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz")


# ── Background worker ─────────────────────────────────────────────────────────

class _RunWorker(QThread):
    """Runs FK or IK engine on a background thread."""

    finished = Signal(object)  # TrialData | AllocationResult
    failed = Signal(str)

    def __init__(
        self,
        mode: str,
        frame_graph,
        n_trials: int,
        seed: int,
        frame_a: str = "",
        frame_b: str = "",
        target_tol: ToleranceSpec6 | None = None,
        objective=None,
        max_iter: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._frame_graph = frame_graph
        self._n_trials = n_trials
        self._seed = seed
        self._frame_a = frame_a
        self._frame_b = frame_b
        self._target_tol = target_tol
        self._objective = objective
        self._max_iter = max_iter

    def run(self) -> None:
        try:
            if self._mode == "fk_verification":
                result = MonteCarloFKEngine.run(
                    self._frame_graph, self._n_trials, self._seed
                )
            else:
                result = AllocationEngine.allocate(
                    self._frame_graph,
                    self._frame_a,
                    self._frame_b,
                    self._target_tol,
                    objective=self._objective,
                    seed=self._seed,
                    n_validate=1000,
                    max_iter=self._max_iter,
                )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# ── Main widget ───────────────────────────────────────────────────────────────

class RunPanelWidget(QWidget):
    """Simulation configuration and execution panel."""

    project_changed = Signal()      # mode/n_trials/seed written back to project
    run_completed = Signal(object)  # TrialData (FK) or AllocationResult (IK)
    run_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel | None = None
        self._worker: _RunWorker | None = None
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_project(self, project: ProjectModel) -> None:
        self._project = project
        self._load_sim_settings()
        self._refresh_frame_combos()

    def refresh_view(self) -> None:
        """Re-populate IK frame combos — call when frames are added/removed."""
        self._refresh_frame_combos()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        container = QWidget()
        outer = QVBoxLayout(container)

        outer.addWidget(self._build_sim_settings_group())
        outer.addWidget(self._build_ik_group())

        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setMinimumHeight(32)
        outer.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setVisible(False)
        outer.addWidget(self._progress_bar)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        outer.addWidget(self._status_label)

        outer.addStretch()

        self._run_btn.clicked.connect(self._on_run_clicked)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._n_trials_spin.valueChanged.connect(self._on_n_trials_changed)
        self._seed_spin.valueChanged.connect(self._on_seed_changed)
        self._randomize_btn.clicked.connect(self._on_randomize_seed)

        # Initialise IK group visibility
        self._update_ik_group_visibility()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(container)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)

    def _build_sim_settings_group(self) -> QGroupBox:
        group = QGroupBox("Simulation Settings")
        layout = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("FK Verification", "fk_verification")
        self._mode_combo.addItem("IK Allocation", "ik_allocation")
        mode_row.addWidget(self._mode_combo, stretch=1)
        layout.addLayout(mode_row)

        trials_row = QHBoxLayout()
        trials_row.addWidget(QLabel("N trials:"))
        self._n_trials_spin = QSpinBox()
        self._n_trials_spin.setRange(1, 10_000_000)
        self._n_trials_spin.setSingleStep(1000)
        self._n_trials_spin.setValue(10000)
        trials_row.addWidget(self._n_trials_spin, stretch=1)
        layout.addLayout(trials_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(42)
        seed_row.addWidget(self._seed_spin, stretch=1)
        self._randomize_btn = QPushButton("Randomize")
        self._randomize_btn.setMaximumWidth(90)
        seed_row.addWidget(self._randomize_btn)
        layout.addLayout(seed_row)

        return group

    def _build_ik_group(self) -> QGroupBox:
        self._ik_group = QGroupBox("IK Target")
        layout = QVBoxLayout(self._ik_group)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame A:"))
        self._frame_a_combo = QComboBox()
        self._frame_a_combo.setMinimumWidth(100)
        frame_row.addWidget(self._frame_a_combo, stretch=1)
        frame_row.addWidget(QLabel("Frame B:"))
        self._frame_b_combo = QComboBox()
        self._frame_b_combo.setMinimumWidth(100)
        frame_row.addWidget(self._frame_b_combo, stretch=1)
        layout.addLayout(frame_row)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItem("Statistical (RSS)", "rss")
        self._method_combo.addItem("Worst-Case (Linear Sum)", "wc")
        method_row.addWidget(self._method_combo, stretch=1)
        layout.addLayout(method_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max iterations:"))
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 500)
        self._max_iter_spin.setValue(30)
        self._max_iter_spin.setToolTip(
            "Maximum angular damping iterations. Increase if the solver reports "
            "non-convergence but the achieved envelope is close to the target."
        )
        iter_row.addWidget(self._max_iter_spin, stretch=1)
        layout.addLayout(iter_row)

        # Target bound table
        grid = QGridLayout()
        grid.addWidget(QLabel("<b>DoF</b>"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("<b>Target Bound</b>"), 0, 1, Qt.AlignmentFlag.AlignCenter)

        self._target_bound_spins: list[QDoubleSpinBox] = []
        for i, dof_name in enumerate(_DOF_NAMES):
            grid.addWidget(QLabel(f"<b>{dof_name}</b>"), i + 1, 0, Qt.AlignmentFlag.AlignCenter)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 9999.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.0001)
            spin.setValue(0.001)
            grid.addWidget(spin, i + 1, 1)
            self._target_bound_spins.append(spin)

        layout.addLayout(grid)
        self._ik_group.setVisible(False)
        return self._ik_group

    # ── Internal state ─────────────────────────────────────────────────────────

    def _load_sim_settings(self) -> None:
        if self._project is None:
            return
        s = self._project.sim_settings
        for w in (self._mode_combo, self._n_trials_spin, self._seed_spin):
            w.blockSignals(True)
        idx = self._mode_combo.findData(s.mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._n_trials_spin.setValue(s.n_trials)
        self._seed_spin.setValue(s.seed)
        for w in (self._mode_combo, self._n_trials_spin, self._seed_spin):
            w.blockSignals(False)
        self._update_ik_group_visibility()

    def _refresh_frame_combos(self) -> None:
        for combo in (self._frame_a_combo, self._frame_b_combo):
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            if self._project:
                for frame in self._project.frames:
                    combo.addItem(frame.name)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        # Default frame B to index 1 if available (avoids frame_a == frame_b on first load)
        if self._frame_b_combo.count() >= 2 and self._frame_b_combo.currentIndex() == 0:
            self._frame_b_combo.setCurrentIndex(1)

    def _update_ik_group_visibility(self) -> None:
        self._ik_group.setVisible(
            self._mode_combo.currentData() == "ik_allocation"
        )

    def _set_status(self, text: str, *, error: bool = False, warning: bool = False) -> None:
        if error:
            self._status_label.setStyleSheet("color: red;")
        elif warning:
            self._status_label.setStyleSheet("color: orange;")
        else:
            self._status_label.setStyleSheet("")
        self._status_label.setText(text)

    def _get_target_tol(self) -> ToleranceSpec6:
        specs = [
            ToleranceSpec(distribution="uniform", bound=spin.value())
            for spin in self._target_bound_spins
        ]
        return ToleranceSpec6(*specs)

    # ── Signal handlers ────────────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        self._update_ik_group_visibility()
        if self._project is not None:
            self._project.sim_settings.mode = self._mode_combo.currentData()
            self.project_changed.emit()

    def _on_n_trials_changed(self, value: int) -> None:
        if self._project is not None:
            self._project.sim_settings.n_trials = value
            self.project_changed.emit()

    def _on_seed_changed(self, value: int) -> None:
        if self._project is not None:
            self._project.sim_settings.seed = value
            self.project_changed.emit()

    def _on_randomize_seed(self) -> None:
        self._seed_spin.setValue(random.randint(0, 2**31 - 1))

    def _on_run_clicked(self) -> None:
        if self._project is None:
            return

        if not self._project.edges:
            self._set_status("Error: project has no edges to simulate", error=True)
            return

        mode = self._mode_combo.currentData()
        n_trials = self._n_trials_spin.value()
        seed = self._seed_spin.value()

        if mode == "ik_allocation":
            frame_a = self._frame_a_combo.currentText()
            frame_b = self._frame_b_combo.currentText()
            if not frame_a or not frame_b:
                self._set_status("Error: select Frame A and Frame B", error=True)
                return
            if frame_a == frame_b:
                self._set_status("Error: Frame A and Frame B must be different", error=True)
                return
            target_tol = self._get_target_tol()
            max_iter = self._max_iter_spin.value()
            objective = RSSAllocation() if self._method_combo.currentData() == "rss" else EqualAllocation()
        else:
            frame_a = frame_b = ""
            target_tol = None
            objective = None
            max_iter = 30

        try:
            frame_graph = project_model_to_frame_graph(self._project)
        except Exception as exc:
            self._set_status(f"Error building model: {exc}", error=True)
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._set_status("Running…")

        self._worker = _RunWorker(
            mode=mode,
            frame_graph=frame_graph,
            n_trials=n_trials,
            seed=seed,
            frame_a=frame_a,
            frame_b=frame_b,
            target_tol=target_tol,
            objective=objective,
            max_iter=max_iter,
            parent=self,
        )
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_run_finished(self, result: object) -> None:
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        if isinstance(result, TrialData):
            self._set_status(f"FK complete — {result.n_trials:,} trials (seed {result.seed})")
        elif isinstance(result, AllocationResult):
            if result.converged:
                self._set_status(
                    f"IK complete — converged in {result.iterations_used} iteration(s)"
                )
            else:
                self._set_status(
                    f"IK allocation did not converge ({result.iterations_used} iterations)",
                    warning=True,
                )
        self.run_completed.emit(result)

    def _on_run_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._set_status(f"Error: {error}", error=True)
        self.run_failed.emit(error)
