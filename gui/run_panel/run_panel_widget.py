"""
gui/run_panel/run_panel_widget.py — Simulation execution panel (Section 6.16).

This is the ONE place where persistence.schema objects are converted into live
core/sim objects (Section 5.3): project_model_to_frame_graph() is called on the
main thread immediately before starting the background worker. The worker owns
the FrameGraph and never touches the ProjectModel.

Modes:
  FK Verification — MonteCarloFKEngine.run() → TrialData
  IK Allocation   — AllocationEngine.allocate_multi() → AllocationResult

SimSettings (mode, n_trials, seed) are written back to project.sim_settings on
every change so they persist with the project file.
"""

from __future__ import annotations

import random

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
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
from persistence.schema import (
    IKConstraintModel,
    ProjectModel,
    ToleranceSpecModel,
    ToleranceSpec6Model,
    project_model_to_frame_graph,
)
from sim.allocation import AllocationEngine, AllocationResult
from sim.monte_carlo_fk import MonteCarloFKEngine, TrialData

_DOF_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz")


# ── Constraint row widget ─────────────────────────────────────────────────────

class _ConstraintRowWidget(QWidget):
    """One (frame_a → frame_b, target) row in the multi-pair IK constraint list."""

    removed = Signal(object)   # emits self; connected by RunPanelWidget
    changed = Signal()         # emits on any frame pair or target change

    def __init__(self, frame_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui(frame_names)

    def _build_ui(self, frame_names: list[str]) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        # ── Frame pair row ────────────────────────────────────────────────────
        pair_row = QHBoxLayout()
        pair_row.addWidget(QLabel("A:"))
        self._frame_a_combo = QComboBox()
        self._frame_a_combo.setMinimumWidth(90)
        for name in frame_names:
            self._frame_a_combo.addItem(name)
        pair_row.addWidget(self._frame_a_combo, stretch=1)

        pair_row.addWidget(QLabel("→  B:"))
        self._frame_b_combo = QComboBox()
        self._frame_b_combo.setMinimumWidth(90)
        for name in frame_names:
            self._frame_b_combo.addItem(name)
        if self._frame_b_combo.count() >= 2:
            self._frame_b_combo.setCurrentIndex(1)
        pair_row.addWidget(self._frame_b_combo, stretch=1)

        self._remove_btn = QPushButton("✕")
        self._remove_btn.setMaximumWidth(28)
        self._remove_btn.setToolTip("Remove this constraint")
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self))
        pair_row.addWidget(self._remove_btn)
        layout.addLayout(pair_row)

        # ── Target spinboxes ──────────────────────────────────────────────────
        target_grid = QGridLayout()
        target_grid.setHorizontalSpacing(4)
        target_grid.setVerticalSpacing(1)
        self._spins: list[QDoubleSpinBox] = []
        for i, dof in enumerate(_DOF_NAMES):
            col = (i % 3) * 2
            row = i // 3
            target_grid.addWidget(QLabel(f"{dof}:"), row, col, Qt.AlignmentFlag.AlignRight)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 9999.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.0001)
            spin.setValue(0.001)
            spin.setMinimumWidth(80)
            target_grid.addWidget(spin, row, col + 1)
            self._spins.append(spin)
        layout.addLayout(target_grid)

        # ── Separator ─────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Wire changes so the run panel can persist them to the project model.
        self._frame_a_combo.currentIndexChanged.connect(self.changed)
        self._frame_b_combo.currentIndexChanged.connect(self.changed)
        for spin in self._spins:
            spin.valueChanged.connect(self.changed)

    def refresh_frames(self, frame_names: list[str]) -> None:
        for combo in (self._frame_a_combo, self._frame_b_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for name in frame_names:
                combo.addItem(name)
            idx = combo.findText(current)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

    def set_frame_pair(self, frame_a: str, frame_b: str) -> None:
        """Set frame pair without emitting changed (used during load)."""
        for combo, name in ((self._frame_a_combo, frame_a), (self._frame_b_combo, frame_b)):
            combo.blockSignals(True)
            idx = combo.findText(name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def set_target_model(self, target: "ToleranceSpec6Model") -> None:
        """Populate spinboxes from a ToleranceSpec6Model without emitting changed."""
        for spin, dof in zip(self._spins, _DOF_NAMES):
            spin.blockSignals(True)
            spin.setValue(getattr(target, dof).bound)
            spin.blockSignals(False)

    def set_remove_enabled(self, enabled: bool) -> None:
        self._remove_btn.setEnabled(enabled)

    def get_frame_pair(self) -> tuple[str, str]:
        return self._frame_a_combo.currentText(), self._frame_b_combo.currentText()

    def get_target_model(self) -> ToleranceSpec6Model:
        return ToleranceSpec6Model(**{
            dof: ToleranceSpecModel(distribution="uniform", bound=spin.value())
            for dof, spin in zip(_DOF_NAMES, self._spins)
        })


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
        ik_targets: list[tuple[str, str, ToleranceSpec6]] | None = None,
        max_iter: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._frame_graph = frame_graph
        self._n_trials = n_trials
        self._seed = seed
        self._ik_targets = ik_targets or []
        self._max_iter = max_iter

    def run(self) -> None:
        try:
            if self._mode == "fk_verification":
                result = MonteCarloFKEngine.run(
                    self._frame_graph, self._n_trials, self._seed
                )
            else:
                result = AllocationEngine.allocate_multi(
                    self._frame_graph,
                    self._ik_targets,
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
        self._constraint_rows: list[_ConstraintRowWidget] = []
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_project(self, project: ProjectModel) -> None:
        self._project = project
        self._load_sim_settings()
        self._refresh_all_constraint_frames()

    def refresh_view(self) -> None:
        """Re-populate IK frame combos — call when frames are added/removed."""
        self._refresh_all_constraint_frames()

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
        self._max_iter_spin.valueChanged.connect(self._on_max_iter_changed)

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
        self._ik_group = QGroupBox("IK Allocation")
        layout = QVBoxLayout(self._ik_group)

        # ── Iterations ────────────────────────────────────────────────────────
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

        # ── Constraint list ───────────────────────────────────────────────────
        constraints_label = QLabel("<b>Point-Pair Constraints</b>")
        layout.addWidget(constraints_label)

        self._constraints_container = QWidget()
        self._constraints_layout = QVBoxLayout(self._constraints_container)
        self._constraints_layout.setContentsMargins(0, 0, 0, 0)
        self._constraints_layout.setSpacing(0)
        layout.addWidget(self._constraints_container)

        self._add_constraint_btn = QPushButton("+ Add Constraint")
        self._add_constraint_btn.clicked.connect(self._on_add_constraint)
        layout.addWidget(self._add_constraint_btn)

        # Seed the list with one row
        self._add_constraint_row()

        self._ik_group.setVisible(False)
        return self._ik_group

    # ── Constraint row management ─────────────────────────────────────────────

    def _add_constraint_row(self) -> _ConstraintRowWidget:
        frame_names = self._current_frame_names()
        row = _ConstraintRowWidget(frame_names, parent=self._constraints_container)
        row.removed.connect(self._on_remove_constraint)
        row.changed.connect(self._on_constraints_changed)
        self._constraints_layout.addWidget(row)
        self._constraint_rows.append(row)
        self._update_remove_buttons()
        return row

    def _on_add_constraint(self) -> None:
        self._add_constraint_row()
        self._save_ik_constraints_to_project()

    def _on_remove_constraint(self, row: _ConstraintRowWidget) -> None:
        if len(self._constraint_rows) <= 1:
            return
        self._constraint_rows.remove(row)
        self._constraints_layout.removeWidget(row)
        row.deleteLater()
        self._update_remove_buttons()
        self._save_ik_constraints_to_project()

    def _update_remove_buttons(self) -> None:
        only_one = len(self._constraint_rows) == 1
        for row in self._constraint_rows:
            row.set_remove_enabled(not only_one)

    def _current_frame_names(self) -> list[str]:
        if self._project is None:
            return []
        return [f.name for f in self._project.frames]

    def _refresh_all_constraint_frames(self) -> None:
        frame_names = self._current_frame_names()
        for row in self._constraint_rows:
            row.refresh_frames(frame_names)

    # ── Internal state ─────────────────────────────────────────────────────────

    def _load_sim_settings(self) -> None:
        if self._project is None:
            return
        s = self._project.sim_settings
        for w in (self._mode_combo, self._n_trials_spin, self._seed_spin, self._max_iter_spin):
            w.blockSignals(True)
        idx = self._mode_combo.findData(s.mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._n_trials_spin.setValue(s.n_trials)
        self._seed_spin.setValue(s.seed)
        self._max_iter_spin.setValue(s.ik_max_iter)
        for w in (self._mode_combo, self._n_trials_spin, self._seed_spin, self._max_iter_spin):
            w.blockSignals(False)
        self._update_ik_group_visibility()
        self._load_ik_constraints_from_project()

    def _load_ik_constraints_from_project(self) -> None:
        """Rebuild constraint rows from project.sim_settings.ik_constraints."""
        if self._project is None:
            return
        saved = self._project.sim_settings.ik_constraints
        if not saved:
            return  # keep the default single empty row

        # Clear existing rows silently.
        for row in list(self._constraint_rows):
            self._constraints_layout.removeWidget(row)
            row.deleteLater()
        self._constraint_rows.clear()

        frame_names = self._current_frame_names()
        for constraint in saved:
            row = _ConstraintRowWidget(frame_names, parent=self._constraints_container)
            row.removed.connect(self._on_remove_constraint)
            row.changed.connect(self._on_constraints_changed)
            self._constraints_layout.addWidget(row)
            self._constraint_rows.append(row)
            row.set_frame_pair(constraint.frame_a, constraint.frame_b)
            row.set_target_model(constraint.target)

        self._update_remove_buttons()

    def _save_ik_constraints_to_project(self) -> None:
        """Persist current constraint rows to project.sim_settings.ik_constraints."""
        if self._project is None:
            return
        constraints: list[IKConstraintModel] = []
        for row in self._constraint_rows:
            frame_a, frame_b = row.get_frame_pair()
            if not frame_a or not frame_b:
                continue
            constraints.append(
                IKConstraintModel(frame_a=frame_a, frame_b=frame_b, target=row.get_target_model())
            )
        self._project.sim_settings.ik_constraints = constraints

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

    def _collect_ik_targets(self) -> list[tuple[str, str, ToleranceSpec6Model]] | None:
        """Collect and validate all constraint rows. Returns None on error."""
        targets: list[tuple[str, str, ToleranceSpec6Model]] = []
        for i, row in enumerate(self._constraint_rows):
            frame_a, frame_b = row.get_frame_pair()
            if not frame_a or not frame_b:
                self._set_status(f"Error: constraint {i + 1} has no frames selected", error=True)
                return None
            if frame_a == frame_b:
                self._set_status(
                    f"Error: constraint {i + 1} — Frame A and Frame B must differ", error=True
                )
                return None
            targets.append((frame_a, frame_b, row.get_target_model()))
        return targets

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

    def _on_max_iter_changed(self, value: int) -> None:
        if self._project is not None:
            self._project.sim_settings.ik_max_iter = value
            self.project_changed.emit()

    def _on_constraints_changed(self) -> None:
        self._save_ik_constraints_to_project()
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

        ik_targets = None
        max_iter = 30

        if mode == "ik_allocation":
            ik_target_models = self._collect_ik_targets()
            if ik_target_models is None:
                return
            ik_targets = [
                (fa, fb, ToleranceSpec6(*(ToleranceSpec("uniform", bound=getattr(m, d).bound) for d in _DOF_NAMES)))
                for fa, fb, m in ik_target_models
            ]
            max_iter = self._max_iter_spin.value()

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
            ik_targets=ik_targets,
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
            n = len(self._constraint_rows)
            pair_label = f"{n} constraint{'s' if n > 1 else ''}"
            if result.converged:
                self._set_status(
                    f"IK complete — converged in {result.iterations_used} iteration(s)"
                    f"  [{pair_label}]"
                )
            else:
                self._set_status(
                    f"IK allocation did not converge ({result.iterations_used} iterations)"
                    f"  [{pair_label}]",
                    warning=True,
                )
        self.run_completed.emit(result)

    def _on_run_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._set_status(f"Error: {error}", error=True)
        self.run_failed.emit(error)
