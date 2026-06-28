"""
gui/point_pair_panel/point_pair_panel_widget.py — Point-Pair Analysis panel (C-5).

Read-write panel: saves / deletes named (frame_a, frame_b) analyses from
project.saved_analyses, emits project_changed on each mutation.

When a FK TrialData result is available and the selected frame pair is
connected, computes and shows the relative-pose envelope via
point_pair_envelope_box(). IK AllocationResult is silently ignored.
"""

from __future__ import annotations

import networkx as nx

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QComboBox,
    QWidget,
)

from persistence.schema import ProjectModel, SavedAnalysisModel, project_model_to_frame_graph
from postprocess.stats import DOF_LABELS, point_pair_envelope_box
from sim.monte_carlo_fk import TrialData


def _are_connected(project: ProjectModel, frame_a: str, frame_b: str) -> bool:
    """Return True iff frame_a and frame_b are in the same connected component."""
    if not frame_a or not frame_b or frame_a == frame_b:
        return False
    g = nx.Graph()
    for f in project.frames:
        g.add_node(f.name)
    for e in project.edges:
        g.add_edge(e.parent, e.child)
    return nx.has_path(g, frame_a, frame_b)


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


class PointPairPanelWidget(QWidget):
    """Point-Pair Analysis panel — define named frame pairs, view relative-pose envelope."""

    project_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel | None = None
        self._trial_data: TrialData | None = None
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_project(self, project: ProjectModel) -> None:
        self._project = project
        self._trial_data = None
        self._repopulate_frame_combos()
        self._repopulate_saved_list()
        self._update_envelope()

    def set_result(self, result: object) -> None:
        if isinstance(result, TrialData):
            self._trial_data = result
            self._update_envelope()
        # AllocationResult: silently ignore

    def refresh_view(self) -> None:
        """Re-populate frame combos after frames are added / removed."""
        self._repopulate_frame_combos()
        self._update_envelope()

    def clear(self) -> None:
        """Hide the envelope table; preserve project and combos."""
        self._trial_data = None
        self._update_envelope()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        container = QWidget()
        outer = QVBoxLayout(container)

        outer.addWidget(self._build_frame_pair_group())
        outer.addWidget(self._build_saved_analyses_group())
        outer.addWidget(self._build_envelope_group())
        outer.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(container)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)

    def _build_frame_pair_group(self) -> QGroupBox:
        group = QGroupBox("Frame Pair")
        layout = QVBoxLayout(group)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Frame A:"))
        self._frame_a_combo = QComboBox()
        self._frame_a_combo.setMinimumWidth(100)
        combo_row.addWidget(self._frame_a_combo, stretch=1)
        combo_row.addWidget(QLabel("Frame B:"))
        self._frame_b_combo = QComboBox()
        self._frame_b_combo.setMinimumWidth(100)
        combo_row.addWidget(self._frame_b_combo, stretch=1)
        layout.addLayout(combo_row)

        self._connectivity_label = QLabel()
        self._connectivity_label.setWordWrap(True)
        self._connectivity_label.setStyleSheet("color: red;")
        layout.addWidget(self._connectivity_label)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        name_row.addWidget(self._name_edit, stretch=1)
        self._save_btn = QPushButton("Save Analysis")
        self._save_btn.setMaximumWidth(110)
        name_row.addWidget(self._save_btn)
        layout.addLayout(name_row)

        self._save_error_label = QLabel()
        self._save_error_label.setStyleSheet("color: red;")
        layout.addWidget(self._save_error_label)

        self._frame_a_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._frame_b_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._save_btn.clicked.connect(self._on_save_clicked)

        return group

    def _build_saved_analyses_group(self) -> QGroupBox:
        group = QGroupBox("Saved Analyses")
        layout = QVBoxLayout(group)

        self._saved_list = QListWidget()
        self._saved_list.setMaximumHeight(130)
        layout.addWidget(self._saved_list)

        self._delete_btn = QPushButton("Delete Selected")
        layout.addWidget(self._delete_btn)

        self._saved_list.currentRowChanged.connect(self._on_saved_row_changed)
        self._delete_btn.clicked.connect(self._on_delete_clicked)

        return group

    def _build_envelope_group(self) -> QGroupBox:
        group = QGroupBox("Relative-Pose Envelope")
        layout = QVBoxLayout(group)

        self._envelope_placeholder = QLabel("Run FK simulation to see relative-pose envelope.")
        self._envelope_placeholder.setStyleSheet("color: gray; font-style: italic;")
        self._envelope_placeholder.setWordWrap(True)
        layout.addWidget(self._envelope_placeholder)

        self._envelope_table = QTableWidget(6, 3)
        self._envelope_table.setHorizontalHeaderLabels(["DoF", "Min", "Max"])
        self._envelope_table.verticalHeader().setVisible(False)
        self._envelope_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._envelope_table.setMaximumHeight(210)
        self._envelope_table.setVisible(False)
        layout.addWidget(self._envelope_table)

        return group

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _repopulate_frame_combos(self) -> None:
        for combo in (self._frame_a_combo, self._frame_b_combo):
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            if self._project:
                for f in self._project.frames:
                    combo.addItem(f.name)
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        if self._frame_b_combo.count() >= 2 and self._frame_b_combo.currentIndex() == 0:
            self._frame_b_combo.blockSignals(True)
            self._frame_b_combo.setCurrentIndex(1)
            self._frame_b_combo.blockSignals(False)
        self._on_selection_changed()

    def _repopulate_saved_list(self) -> None:
        self._saved_list.blockSignals(True)
        self._saved_list.clear()
        if self._project:
            for analysis in self._project.saved_analyses:
                self._saved_list.addItem(f"{analysis.name}  ({analysis.frame_a} → {analysis.frame_b})")
        self._saved_list.blockSignals(False)

    def _on_selection_changed(self) -> None:
        frame_a = self._frame_a_combo.currentText()
        frame_b = self._frame_b_combo.currentText()

        if not frame_a or not frame_b:
            self._connectivity_label.setText("")
            self._save_btn.setEnabled(False)
            self._update_envelope()
            return

        if frame_a == frame_b:
            self._connectivity_label.setText("Frame A and Frame B must be different.")
            self._save_btn.setEnabled(False)
            self._update_envelope()
            return

        if self._project and not _are_connected(self._project, frame_a, frame_b):
            self._connectivity_label.setText(
                f"Frames '{frame_a}' and '{frame_b}' are not connected."
            )
            self._save_btn.setEnabled(False)
            self._update_envelope()
            return

        self._connectivity_label.setText("")
        self._save_btn.setEnabled(True)
        self._name_edit.setText(f"{frame_a} → {frame_b}")
        self._save_error_label.setText("")
        self._update_envelope()

    def _on_save_clicked(self) -> None:
        if self._project is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            self._save_error_label.setText("Name is required.")
            return
        if any(a.name == name for a in self._project.saved_analyses):
            self._save_error_label.setText(f"Analysis '{name}' already exists.")
            return
        frame_a = self._frame_a_combo.currentText()
        frame_b = self._frame_b_combo.currentText()
        self._project.saved_analyses.append(
            SavedAnalysisModel(name=name, frame_a=frame_a, frame_b=frame_b)
        )
        self._repopulate_saved_list()
        self._save_error_label.setText("")
        self.project_changed.emit()

    def _on_saved_row_changed(self, row: int) -> None:
        if self._project is None or row < 0 or row >= len(self._project.saved_analyses):
            return
        analysis = self._project.saved_analyses[row]
        for combo, name in (
            (self._frame_a_combo, analysis.frame_a),
            (self._frame_b_combo, analysis.frame_b),
        ):
            combo.blockSignals(True)
            idx = combo.findText(name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._on_selection_changed()

    def _on_delete_clicked(self) -> None:
        if self._project is None:
            return
        row = self._saved_list.currentRow()
        if row < 0:
            return
        self._project.saved_analyses.pop(row)
        self._saved_list.takeItem(row)
        self.project_changed.emit()

    def _update_envelope(self) -> None:
        frame_a = self._frame_a_combo.currentText()
        frame_b = self._frame_b_combo.currentText()

        if (
            self._trial_data is None
            or self._project is None
            or not _are_connected(self._project, frame_a, frame_b)
        ):
            self._envelope_placeholder.setVisible(True)
            self._envelope_table.setVisible(False)
            return

        try:
            fg = project_model_to_frame_graph(self._project)
            envelope = point_pair_envelope_box(self._trial_data, fg, frame_a, frame_b)
            _fill_envelope_table(self._envelope_table, envelope)
            self._envelope_placeholder.setVisible(False)
            self._envelope_table.setVisible(True)
        except Exception as exc:
            self._connectivity_label.setText(f"Error computing envelope: {exc}")
            self._envelope_placeholder.setVisible(True)
            self._envelope_table.setVisible(False)
