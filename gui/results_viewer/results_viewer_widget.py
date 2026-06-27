"""
gui/results_viewer/results_viewer_widget.py — Simulation results display panel (C-4).

Read-only. MainWindow calls set_result(result, project) after each run and
clear() on New / Open. No signals out, no project write-back.

FK mode (TrialData):
    - Frame selector → per-frame envelope table + generate_frame_report() figure
    - Pareto sensitivity section (on-demand, frame A/B selection required)

IK mode (AllocationResult):
    - Convergence status label
    - Corrected allocation table (edge × DoF; locked DoFs shown as "—")
    - Achieved envelope table (DoF | Min | Max | Pass?) from ValidationReport
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from persistence.schema import ProjectModel, project_model_to_frame_graph
from postprocess.reporting import generate_frame_report, generate_sensitivity_report
from postprocess.stats import DOF_LABELS, compute_tolerance_sensitivities, frame_envelope_box
from sim.allocation import AllocationResult
from sim.monte_carlo_fk import TrialData


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
    """Displays FK or IK simulation results. Read-only — no signals out."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel | None = None
        self._trial_data: TrialData | None = None
        self._fk_figure = None
        self._pareto_figure = None
        self._fk_canvas: FigureCanvas | None = None
        self._pareto_canvas: FigureCanvas | None = None
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_result(self, result: object, project: ProjectModel) -> None:
        self._project = project
        if isinstance(result, TrialData):
            self._trial_data = result
            self._show_fk(result)
        elif isinstance(result, AllocationResult):
            self._trial_data = None
            self._show_ik(result)

    def clear(self) -> None:
        self._project = None
        self._trial_data = None
        self._close_figures()
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

        # Envelope table
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

        # Frame report figure (canvas created on demand in _update_fk_display)
        fig_group = QGroupBox("Frame Error Report")
        self._fk_canvas_layout = QVBoxLayout(fig_group)
        layout.addWidget(fig_group)

        # Pareto section
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
        self._pareto_btn = QPushButton("Compute")
        self._pareto_btn.setMaximumWidth(80)
        ctrl_row.addWidget(self._pareto_btn)
        pareto_layout.addLayout(ctrl_row)

        self._pareto_status_label = QLabel()
        self._pareto_status_label.setWordWrap(True)
        pareto_layout.addWidget(self._pareto_status_label)

        self._pareto_canvas_layout = QVBoxLayout()
        pareto_layout.addLayout(self._pareto_canvas_layout)

        layout.addWidget(pareto_group)
        layout.addStretch()

        self._frame_combo.currentIndexChanged.connect(self._on_frame_changed)
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
        layout.addWidget(alloc_group)

        achieved_group = QGroupBox("Achieved Envelope vs. Target")
        achieved_layout = QVBoxLayout(achieved_group)
        self._achieved_table = QTableWidget(6, 4)
        self._achieved_table.setHorizontalHeaderLabels(["DoF", "Min", "Max", "Pass?"])
        self._achieved_table.verticalHeader().setVisible(False)
        self._achieved_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._achieved_table.setMaximumHeight(210)
        achieved_layout.addWidget(self._achieved_table)
        layout.addWidget(achieved_group)

        layout.addStretch()
        return page

    # ── FK display ─────────────────────────────────────────────────────────────

    def _show_fk(self, trial_data: TrialData) -> None:
        self._close_figures()
        self._pareto_status_label.setText("")

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

        if self._fk_figure is not None:
            plt.close(self._fk_figure)
        self._fk_figure = generate_frame_report(self._trial_data, frame_name)

        # Replace canvas widget in the group box layout
        if self._fk_canvas is not None:
            self._fk_canvas_layout.removeWidget(self._fk_canvas)
            self._fk_canvas.setParent(None)
            self._fk_canvas.deleteLater()

        self._fk_canvas = FigureCanvas(self._fk_figure)
        self._fk_canvas_layout.addWidget(self._fk_canvas)

    # ── IK display ─────────────────────────────────────────────────────────────

    def _show_ik(self, result: AllocationResult) -> None:
        self._close_figures()

        if result.converged:
            if result.iterations_used == 0:
                status = "✓ Baseline linear allocation passed validation"
            else:
                status = f"✓ Converged in {result.iterations_used} iteration(s)"
            self._ik_status_label.setStyleSheet(
                "font-size: 13px; padding: 4px; color: green;"
            )
        else:
            status = f"✗ {result.status_message}"
            self._ik_status_label.setStyleSheet(
                "font-size: 13px; padding: 4px; color: orange;"
            )
        self._ik_status_label.setText(status)

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

        vr = result.final_validation_report
        for row, dof in enumerate(DOF_LABELS):
            d = vr.achieved_envelope.get(dof, {})
            passed = vr.per_dof_pass.get(dof, False)
            self._achieved_table.setItem(row, 0, _ro_item(dof))
            self._achieved_table.setItem(row, 1, _ro_item(f"{d.get('min', 0.0):.6f}"))
            self._achieved_table.setItem(row, 2, _ro_item(f"{d.get('max', 0.0):.6f}"))
            pass_item = _ro_item("✓" if passed else "✗")
            pass_item.setForeground(QColor("green") if passed else QColor("red"))
            self._achieved_table.setItem(row, 3, pass_item)

        self._stack.setCurrentIndex(2)

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
            if self._pareto_figure is not None:
                plt.close(self._pareto_figure)
            self._pareto_figure = generate_sensitivity_report(report)
            if self._pareto_canvas is not None:
                self._pareto_canvas_layout.removeWidget(self._pareto_canvas)
                self._pareto_canvas.setParent(None)
                self._pareto_canvas.deleteLater()
            self._pareto_canvas = FigureCanvas(self._pareto_figure)
            self._pareto_canvas_layout.addWidget(self._pareto_canvas)
            self._pareto_status_label.setText("")
        except Exception as exc:
            self._pareto_status_label.setText(f"Error: {exc}")

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def _close_figures(self) -> None:
        if self._fk_figure is not None:
            plt.close(self._fk_figure)
            self._fk_figure = None
        if self._pareto_figure is not None:
            plt.close(self._pareto_figure)
            self._pareto_figure = None
