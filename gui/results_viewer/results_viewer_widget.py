"""
gui/results_viewer/results_viewer_widget.py — Simulation results display panel (C-4).

Read-only. MainWindow calls set_result(result, project) after each run and
clear() on New / Open. No signals out, no project write-back.

FK mode (TrialData):
    - Frame selector → per-frame envelope table
    - "Open Frame Report" button → new window with generate_frame_report() figure
    - Pareto section → "Compute & Open" button → new window with sensitivity chart

IK mode (AllocationResult):
    - Convergence status label
    - Corrected allocation table (edge × DoF; locked DoFs shown as "—")
    - Achieved envelope table (DoF | Min | Max | Pass?) from ValidationReport

All Matplotlib figures open in standalone QWidget windows rather than embedded in
the dock — avoids the fixed-size canvas / dock sizing problem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from persistence.schema import ProjectModel, ToleranceSpecModel, project_model_to_frame_graph
from postprocess.reporting import generate_frame_report, generate_sensitivity_report
from postprocess.stats import DOF_LABELS, compute_tolerance_sensitivities, frame_envelope_box
from core.tolerance import ToleranceSpec6
from sim.allocation import AllocationResult, ValidationReport
from sim.monte_carlo_fk import TrialData


class _FigureWindow(QWidget):
    """Standalone window hosting a single Matplotlib figure."""

    def __init__(self, fig, title: str) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self._fig = fig
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        self.setWindowTitle(title)
        self.resize(1200, 900)

    def closeEvent(self, event) -> None:
        plt.close(self._fig)
        super().closeEvent(event)


def _scrollable(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QScrollArea.Shape.NoFrame)
    sa.setWidget(inner)
    return sa


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    return item


def _fill_envelope_table(table: QTableWidget, envelope: dict) -> None:
    for row, dof in enumerate(DOF_LABELS):
        d = envelope.get(dof, {})
        table.setItem(row, 0, _ro_item(dof))
        table.setItem(row, 1, _ro_item(f"{d.get('min', 0.0):.6f}"))
        table.setItem(row, 2, _ro_item(f"{d.get('max', 0.0):.6f}"))


class ResultsViewerWidget(QWidget):
    """Displays FK or IK simulation results.

    Emits project_changed when the user applies an IK allocation back to the
    project (the only write operation this widget performs).
    """

    project_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel | None = None
        self._trial_data: TrialData | None = None
        self._last_ik_result: AllocationResult | None = None
        self._open_windows: list[_FigureWindow] = []
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_result(self, result: object, project: ProjectModel) -> None:
        self._project = project
        if isinstance(result, TrialData):
            self._trial_data = result
            self._last_ik_result = None
            self._show_fk(result)
        elif isinstance(result, AllocationResult):
            self._trial_data = None
            self._last_ik_result = result
            self._show_ik(result)

    def clear(self) -> None:
        self._project = None
        self._trial_data = None
        self._last_ik_result = None
        self._apply_btn.setEnabled(False)
        self._close_windows()
        self._stack.setCurrentIndex(0)

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        placeholder = QLabel("Run a simulation to see results here.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic; padding: 20px;")
        self._stack.addWidget(placeholder)              # page 0

        self._stack.addWidget(_scrollable(self._build_fk_page()))   # page 1
        self._stack.addWidget(_scrollable(self._build_ik_page()))   # page 2

    def _build_fk_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self._frame_combo = QComboBox()
        self._frame_combo.setMinimumWidth(120)
        frame_row.addWidget(self._frame_combo, stretch=1)
        layout.addLayout(frame_row)

        self._view_report_btn = QPushButton("Open Frame Report in New Window")
        self._view_report_btn.setEnabled(False)
        layout.addWidget(self._view_report_btn)

        env_group = QGroupBox("Envelope (min/max per DoF)")
        env_layout = QVBoxLayout(env_group)
        self._envelope_table = QTableWidget(6, 3)
        self._envelope_table.setHorizontalHeaderLabels(["DoF", "Min", "Max"])
        self._envelope_table.verticalHeader().setVisible(False)
        self._envelope_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._envelope_table.setMaximumHeight(210)
        env_layout.addWidget(self._envelope_table)
        layout.addWidget(env_group)

        pareto_group = QGroupBox("Pareto Sensitivity (on demand)")
        pareto_layout = QVBoxLayout(pareto_group)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Frame A:"))
        self._pareto_a_combo = QComboBox()
        self._pareto_a_combo.setMinimumWidth(100)
        ctrl_row.addWidget(self._pareto_a_combo, stretch=1)
        ctrl_row.addWidget(QLabel("→"))
        ctrl_row.addWidget(QLabel("Frame B:"))
        self._pareto_b_combo = QComboBox()
        self._pareto_b_combo.setMinimumWidth(100)
        ctrl_row.addWidget(self._pareto_b_combo, stretch=1)
        self._pareto_btn = QPushButton("Compute && Open")
        self._pareto_btn.setMaximumWidth(120)
        ctrl_row.addWidget(self._pareto_btn)
        pareto_layout.addLayout(ctrl_row)
        self._pareto_status_label = QLabel()
        self._pareto_status_label.setWordWrap(True)
        pareto_layout.addWidget(self._pareto_status_label)
        layout.addWidget(pareto_group)

        layout.addStretch()

        self._frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        self._view_report_btn.clicked.connect(self._on_view_report_clicked)
        self._pareto_btn.clicked.connect(self._compute_pareto)

        return page

    def _build_ik_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._ik_status_label = QLabel()
        self._ik_status_label.setWordWrap(True)
        self._ik_status_label.setStyleSheet("font-size: 13px; padding: 4px;")
        layout.addWidget(self._ik_status_label)

        alloc_group = QGroupBox("Corrected Allocation")
        alloc_layout = QVBoxLayout(alloc_group)
        self._alloc_table = QTableWidget(0, 7)
        self._alloc_table.setHorizontalHeaderLabels(
            ["Edge", "dx", "dy", "dz", "rx", "ry", "rz"]
        )
        self._alloc_table.verticalHeader().setVisible(False)
        self._alloc_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        alloc_layout.addWidget(self._alloc_table)

        self._apply_btn = QPushButton("Apply Corrected Allocation to Project…")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip(
            "Write the corrected tolerance bounds back to the project's edge tolerance "
            "specs. Only available when the allocation converged. Locked DoFs are not "
            "changed. A confirmation prompt will appear before any values are overwritten."
        )
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        alloc_layout.addWidget(self._apply_btn)
        layout.addWidget(alloc_group)

        # Per-pair achieved envelope section — rebuilt dynamically in _show_ik().
        self._per_pair_container = QWidget()
        self._per_pair_layout = QVBoxLayout(self._per_pair_container)
        self._per_pair_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._per_pair_container)

        layout.addStretch()
        return page

    # ── FK display ─────────────────────────────────────────────────────────────

    def _show_fk(self, trial_data: TrialData) -> None:
        self._pareto_status_label.setText("")
        self._view_report_btn.setEnabled(False)

        frame_names = list(trial_data.frame_poses.keys())

        self._frame_combo.blockSignals(True)
        self._frame_combo.clear()
        for name in frame_names:
            self._frame_combo.addItem(name)
        self._frame_combo.blockSignals(False)

        for combo in (self._pareto_a_combo, self._pareto_b_combo):
            combo.blockSignals(True)
            combo.clear()
            for name in frame_names:
                combo.addItem(name)
            combo.blockSignals(False)
        if self._pareto_b_combo.count() >= 2:
            self._pareto_b_combo.setCurrentIndex(1)

        self._stack.setCurrentIndex(1)
        if frame_names:
            self._update_fk_display(frame_names[0])

    def _on_frame_changed(self, index: int) -> None:
        if self._trial_data is None:
            return
        name = self._frame_combo.itemText(index)
        if name:
            self._update_fk_display(name)

    def _update_fk_display(self, frame_name: str) -> None:
        envelope = frame_envelope_box(self._trial_data, frame_name)
        _fill_envelope_table(self._envelope_table, envelope)
        self._view_report_btn.setEnabled(True)

    def _on_view_report_clicked(self) -> None:
        if self._trial_data is None:
            return
        frame_name = self._frame_combo.currentText()
        if not frame_name:
            return
        fig = generate_frame_report(self._trial_data, frame_name)
        win = _FigureWindow(fig, f"Frame Report: {frame_name}")
        self._open_windows.append(win)
        win.destroyed.connect(lambda: self._open_windows.remove(win) if win in self._open_windows else None)
        win.show()

    # ── IK display ─────────────────────────────────────────────────────────────

    def _show_ik(self, result: AllocationResult) -> None:
        self._apply_btn.setEnabled(result.converged)
        method_label = "Loosest (LP)"

        if result.converged:
            if result.iterations_used == 0:
                status = f"✓ Converged — no angular correction needed  [{method_label}]"
            else:
                status = f"✓ Converged in {result.iterations_used} angular iteration(s)  [{method_label}]"
            self._ik_status_label.setStyleSheet(
                "font-size: 13px; padding: 4px; color: green;"
            )
        else:
            status = f"✗ {result.status_message}  [{method_label}]"
            self._ik_status_label.setStyleSheet(
                "font-size: 13px; padding: 4px; color: orange;"
            )
        self._ik_status_label.setText(status)

        # ── Allocation table ──────────────────────────────────────────────────
        edges = list(result.corrected_allocation.keys())
        self._alloc_table.setRowCount(len(edges))
        for row, edge_name in enumerate(edges):
            corrected = result.corrected_allocation[edge_name]
            baseline = result.baseline_linear_allocation.get(edge_name)
            self._alloc_table.setItem(row, 0, _ro_item(edge_name))
            for col, dof in enumerate(DOF_LABELS):
                c_spec = getattr(corrected, dof)
                if c_spec.locked:
                    item = _ro_item("—")
                    item.setForeground(QColor("gray"))
                else:
                    item = _ro_item(f"{c_spec.bound:.6f}")
                    if baseline is not None:
                        b_spec = getattr(baseline, dof)
                        if not b_spec.locked and abs(c_spec.bound - b_spec.bound) > 1e-9:
                            item.setToolTip(f"baseline: {b_spec.bound:.6f}")
                self._alloc_table.setItem(row, col + 1, item)

        # ── Per-pair achieved envelopes ───────────────────────────────────────
        self._clear_per_pair_widgets()

        if result.per_pair_validation:
            pair_targets = result.per_pair_targets or [None] * len(result.per_pair_validation)
            for (frame_a, frame_b, vr), pair_target in zip(result.per_pair_validation, pair_targets):
                target_for_pair = pair_target[2] if pair_target is not None else None
                title = f"Achieved Envelope: {frame_a} → {frame_b}"
                passed_label = "✓ PASS" if vr.passed else "✗ FAIL"
                color = "green" if vr.passed else "red"
                group = QGroupBox(f"{title}   [{passed_label}]")
                group.setStyleSheet(f"QGroupBox {{ color: {color}; }}")
                g_layout = QVBoxLayout(group)
                table = self._make_achieved_table(vr, target=target_for_pair)
                g_layout.addWidget(table)
                self._per_pair_layout.addWidget(group)
        else:
            # Single-pair legacy path (direct allocate() call).
            vr = result.final_validation_report
            target = result.target_tolerance
            group = QGroupBox("Achieved Envelope vs. Target")
            g_layout = QVBoxLayout(group)
            table = self._make_achieved_table(vr, target=target)
            g_layout.addWidget(table)
            self._per_pair_layout.addWidget(group)

        self._stack.setCurrentIndex(2)

    def _make_achieved_table(
        self, vr: "ValidationReport", target: "ToleranceSpec6 | None"
    ) -> QTableWidget:
        has_target = target is not None
        cols = ["DoF", "Target ±", "Min", "Max", "Pass?"] if has_target else ["DoF", "Min", "Max", "Pass?"]
        table = QTableWidget(6, len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setMaximumHeight(210)

        for i, dof in enumerate(DOF_LABELS):
            d = vr.achieved_envelope.get(dof, {})
            passed = vr.per_dof_pass.get(dof, False)
            col = 0
            table.setItem(i, col, _ro_item(dof)); col += 1
            if has_target:
                table.setItem(i, col, _ro_item(f"±{target[i].bound:.6f}")); col += 1
            table.setItem(i, col, _ro_item(f"{d.get('min', 0.0):.6f}")); col += 1
            table.setItem(i, col, _ro_item(f"{d.get('max', 0.0):.6f}")); col += 1
            pass_item = _ro_item("✓" if passed else "✗")
            pass_item.setForeground(QColor("green") if passed else QColor("red"))
            table.setItem(i, col, pass_item)

        return table

    def _on_apply_clicked(self) -> None:
        if self._last_ik_result is None or self._project is None:
            return

        allocation = self._last_ik_result.corrected_allocation
        edges_affected = [
            e.name for e in self._project.edges if e.name in allocation
        ]
        if not edges_affected:
            QMessageBox.information(
                self,
                "Apply Allocation",
                "No matching edges found in the project for this allocation result.",
            )
            return

        # Build a preview of what will change for the confirmation dialog.
        lines = ["The following edge tolerance bounds will be overwritten:\n"]
        for edge_name in edges_affected:
            tol6 = allocation[edge_name]
            parts = []
            for dof in DOF_LABELS:
                spec = getattr(tol6, dof)
                if not spec.locked:
                    parts.append(f"{dof}: {spec.bound:.6f}")
            if parts:
                lines.append(f"  {edge_name}: {', '.join(parts)}")
        lines.append(
            "\nLocked DoFs are preserved. Distribution and sigma level are "
            "set to uniform for overwritten DoFs."
        )

        reply = QMessageBox.question(
            self,
            "Apply Corrected Allocation",
            "\n".join(lines),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return

        # Write corrected bounds into the project's edge tolerance models.
        for edge_model in self._project.edges:
            if edge_model.name not in allocation:
                continue
            tol6 = allocation[edge_model.name]
            for dof in DOF_LABELS:
                spec = getattr(tol6, dof)
                if spec.locked:
                    continue  # never touch locked DoFs
                setattr(
                    edge_model.tolerance,
                    dof,
                    ToleranceSpecModel(
                        distribution="uniform",
                        bound=spec.bound,
                        sigma_level=spec.sigma_level,
                        locked=False,
                        lower=None,
                        upper=None,
                    ),
                )

        self.project_changed.emit()

    def _clear_per_pair_widgets(self) -> None:
        while self._per_pair_layout.count():
            item = self._per_pair_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ── Pareto ─────────────────────────────────────────────────────────────────

    def _compute_pareto(self) -> None:
        if self._project is None:
            return
        frame_a = self._pareto_a_combo.currentText()
        frame_b = self._pareto_b_combo.currentText()
        if not frame_a or not frame_b:
            self._pareto_status_label.setText("Select Frame A and Frame B.")
            return
        if frame_a == frame_b:
            self._pareto_status_label.setText("Frame A and Frame B must differ.")
            return
        try:
            fg = project_model_to_frame_graph(self._project)
            report = compute_tolerance_sensitivities(fg, frame_a, frame_b)
            fig = generate_sensitivity_report(report)
            win = _FigureWindow(fig, f"Pareto Sensitivity: {frame_a} → {frame_b}")
            self._open_windows.append(win)
            win.destroyed.connect(lambda: self._open_windows.remove(win) if win in self._open_windows else None)
            win.show()
            self._pareto_status_label.setText("")
        except Exception as exc:
            self._pareto_status_label.setText(f"Error: {exc}")

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def _close_windows(self) -> None:
        for win in self._open_windows:
            try:
                win.close()
            except RuntimeError:
                pass
        self._open_windows.clear()
