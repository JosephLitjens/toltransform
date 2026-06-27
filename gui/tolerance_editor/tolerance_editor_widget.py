"""
gui/tolerance_editor/tolerance_editor_widget.py — Per-edge tolerance editor panel (Section 6.15).

Exposes all 6 DoF (dx, dy, dz, rx, ry, rz) for the selected edge, each with:
  - Distribution selector (uniform / normal)
  - Bound entry
  - σ-level entry (enabled only when distribution = normal)
  - Locked checkbox

Live validation uses core/tolerance.ToleranceSpec's constructor as the validator.
All edits write directly to the in-memory ToleranceSpec6Model (Section 5.3 — GUI never
touches core/sim objects during editing).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.tolerance import ToleranceSpec
from persistence.schema import ProjectModel, ToleranceSpecModel

_DOF_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz")
_GRID_HEADERS = ("DoF", "Distribution", "Bound", "σ-level", "Locked", "")


class _DofRow:
    """Controls for one DoF within the tolerance editor grid.

    Not a QWidget — widgets are added directly to the parent QGridLayout.
    """

    def __init__(self, grid: QGridLayout, grid_row: int, dof_name: str) -> None:
        self._loading = False

        lbl = QLabel(f"<b>{dof_name}</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(lbl, grid_row, 0)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["uniform", "normal"])
        self.dist_combo.setMinimumWidth(90)
        grid.addWidget(self.dist_combo, grid_row, 1)

        self.bound_spin = QDoubleSpinBox()
        self.bound_spin.setRange(0.0, 9999.0)
        self.bound_spin.setDecimals(6)
        self.bound_spin.setSingleStep(0.0001)
        grid.addWidget(self.bound_spin, grid_row, 2)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.001, 99.0)
        self.sigma_spin.setDecimals(3)
        self.sigma_spin.setValue(3.0)
        self.sigma_spin.setEnabled(False)
        self.sigma_spin.setStyleSheet("color: gray;")
        grid.addWidget(self.sigma_spin, grid_row, 3)

        self.locked_check = QCheckBox()
        grid.addWidget(self.locked_check, grid_row, 4, Qt.AlignmentFlag.AlignCenter)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red; font-size: 11px;")
        self.error_label.setWordWrap(True)
        grid.addWidget(self.error_label, grid_row, 5)

        self.dist_combo.currentTextChanged.connect(self._on_dist_changed)

    def _on_dist_changed(self, dist: str) -> None:
        is_normal = dist == "normal"
        self.sigma_spin.setEnabled(is_normal)
        self.sigma_spin.setStyleSheet("" if is_normal else "color: gray;")

    def load(self, spec: ToleranceSpecModel) -> None:
        for w in (self.dist_combo, self.bound_spin, self.sigma_spin, self.locked_check):
            w.blockSignals(True)
        self.dist_combo.setCurrentText(spec.distribution)
        self.bound_spin.setValue(spec.bound)
        self.sigma_spin.setValue(spec.sigma_level)
        self.locked_check.setChecked(spec.locked)
        self._on_dist_changed(spec.distribution)
        for w in (self.dist_combo, self.bound_spin, self.sigma_spin, self.locked_check):
            w.blockSignals(False)
        self.error_label.setText("")

    def get_model(self) -> ToleranceSpecModel:
        return ToleranceSpecModel(
            distribution=self.dist_combo.currentText(),
            bound=self.bound_spin.value(),
            sigma_level=self.sigma_spin.value(),
            locked=self.locked_check.isChecked(),
        )

    def is_valid(self) -> bool:
        try:
            ToleranceSpec(
                distribution=self.dist_combo.currentText(),
                bound=self.bound_spin.value(),
                sigma_level=self.sigma_spin.value(),
            )
            self.error_label.setText("")
            return True
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return False

    def connect_changed(self, slot) -> None:
        self.dist_combo.currentTextChanged.connect(slot)
        self.bound_spin.valueChanged.connect(slot)
        self.sigma_spin.valueChanged.connect(slot)
        self.locked_check.stateChanged.connect(slot)


class ToleranceEditorWidget(QWidget):
    """Per-edge, per-DoF tolerance editing panel."""

    project_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel | None = None
        self._selected_edge_name: str | None = None
        self._loading = False
        self._rows: list[_DofRow] = []
        self._setup_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_project(self, project: ProjectModel) -> None:
        self._project = project
        self._selected_edge_name = None
        self._refresh_edge_combo()
        self._load_selected_edge()

    def set_selected_edge(self, edge_name: str) -> None:
        """Select edge by name (called by MainWindow when graph-editor selection changes)."""
        self._selected_edge_name = edge_name
        idx = self._edge_combo.findText(edge_name)
        if idx >= 0:
            self._edge_combo.blockSignals(True)
            self._edge_combo.setCurrentIndex(idx)
            self._edge_combo.blockSignals(False)
        self._load_selected_edge()

    def refresh_view(self) -> None:
        self._refresh_edge_combo()
        self._load_selected_edge()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        container = QWidget()
        outer = QVBoxLayout(container)

        # Edge selector row
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Edge:"))
        self._edge_combo = QComboBox()
        self._edge_combo.setMinimumWidth(180)
        sel_row.addWidget(self._edge_combo, stretch=1)
        outer.addLayout(sel_row)

        # Stacked widget: placeholder (page 0) vs. DoF panel (page 1)
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        placeholder = QLabel(
            "Select an edge to edit its tolerances.\n"
            "Add edges in the Graph Editor panel."
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic; padding: 20px;")
        self._stack.addWidget(placeholder)

        self._stack.addWidget(self._build_dof_page())

        # activated fires on every user pick, even when the index doesn't change
        self._edge_combo.activated.connect(self._on_edge_combo_changed)
        self._stack.setCurrentIndex(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(container)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)

    def _build_dof_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # Edge name header — updated each time an edge is loaded
        self._edge_header_label = QLabel()
        self._edge_header_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 4px 2px; color: #333;"
        )
        page_layout.addWidget(self._edge_header_label)

        # Column header row
        grid = QGridLayout()
        grid.setColumnStretch(5, 1)
        for col, header in enumerate(_GRID_HEADERS):
            lbl = QLabel(f"<b>{header}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)

        # DoF rows
        for i, dof_name in enumerate(_DOF_NAMES):
            row = _DofRow(grid, i + 1, dof_name)
            row.connect_changed(self._on_field_changed)
            self._rows.append(row)

        page_layout.addLayout(grid)
        page_layout.addSpacerItem(
            QSpacerItem(0, 8, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        # Bulk apply group
        bulk_group = QGroupBox("Bulk Apply")
        bulk_vbox = QVBoxLayout(bulk_group)

        bulk_fields = QHBoxLayout()
        bulk_fields.addWidget(QLabel("Distribution:"))
        self._bulk_dist_combo = QComboBox()
        self._bulk_dist_combo.addItems(["uniform", "normal"])
        bulk_fields.addWidget(self._bulk_dist_combo)

        bulk_fields.addWidget(QLabel("Bound:"))
        self._bulk_bound_spin = QDoubleSpinBox()
        self._bulk_bound_spin.setRange(0.0, 9999.0)
        self._bulk_bound_spin.setDecimals(6)
        self._bulk_bound_spin.setSingleStep(0.0001)
        bulk_fields.addWidget(self._bulk_bound_spin)

        bulk_fields.addWidget(QLabel("σ-level:"))
        self._bulk_sigma_spin = QDoubleSpinBox()
        self._bulk_sigma_spin.setRange(0.001, 99.0)
        self._bulk_sigma_spin.setDecimals(3)
        self._bulk_sigma_spin.setValue(3.0)
        bulk_fields.addWidget(self._bulk_sigma_spin)
        bulk_vbox.addLayout(bulk_fields)

        bulk_btns = QHBoxLayout()
        self._apply_edge_btn = QPushButton("Apply to All DoF (This Edge)")
        self._apply_all_btn = QPushButton("Apply to All Edges")
        bulk_btns.addWidget(self._apply_edge_btn)
        bulk_btns.addWidget(self._apply_all_btn)
        bulk_vbox.addLayout(bulk_btns)

        page_layout.addWidget(bulk_group)

        self._apply_edge_btn.clicked.connect(self._on_bulk_apply_to_edge)
        self._apply_all_btn.clicked.connect(self._on_bulk_apply_to_all_edges)

        return page

    # ── Internal state ────────────────────────────────────────────────────────

    def _refresh_edge_combo(self) -> None:
        self._edge_combo.blockSignals(True)
        self._edge_combo.clear()
        if self._project:
            for edge in self._project.edges:
                self._edge_combo.addItem(edge.name)
        if self._selected_edge_name:
            idx = self._edge_combo.findText(self._selected_edge_name)
            if idx >= 0:
                self._edge_combo.setCurrentIndex(idx)
            else:
                self._selected_edge_name = None
        # Auto-select first edge so the DoF panel shows without requiring a click
        if not self._selected_edge_name and self._edge_combo.count() > 0:
            self._selected_edge_name = self._edge_combo.itemText(0)
        self._edge_combo.blockSignals(False)

    def _on_edge_combo_changed(self) -> None:
        name = self._edge_combo.currentText()
        self._selected_edge_name = name if name else None
        self._load_selected_edge()

    def _load_selected_edge(self) -> None:
        edge = self._find_selected_edge()
        if edge is None:
            self._stack.setCurrentIndex(0)
            return
        self._edge_header_label.setText(
            f"Edge: <b>{edge.name}</b>  &nbsp;({edge.parent} → {edge.child})"
        )
        self._stack.setCurrentIndex(1)
        self._loading = True
        for i, dof_name in enumerate(_DOF_NAMES):
            self._rows[i].load(getattr(edge.tolerance, dof_name))
        self._loading = False

    def _find_selected_edge(self):
        if self._project is None or not self._selected_edge_name:
            return None
        return next(
            (e for e in self._project.edges if e.name == self._selected_edge_name),
            None,
        )

    # ── Field-change handler ──────────────────────────────────────────────────

    def _on_field_changed(self) -> None:
        if self._loading:
            return
        edge = self._find_selected_edge()
        if edge is None:
            return
        for i, dof_name in enumerate(_DOF_NAMES):
            row = self._rows[i]
            if row.is_valid():
                setattr(edge.tolerance, dof_name, row.get_model())
        self.project_changed.emit()

    # ── Bulk apply ────────────────────────────────────────────────────────────

    def _on_bulk_apply_to_edge(self) -> None:
        edge = self._find_selected_edge()
        if edge is None:
            return
        dist = self._bulk_dist_combo.currentText()
        bound = self._bulk_bound_spin.value()
        sigma = self._bulk_sigma_spin.value()
        for i, dof_name in enumerate(_DOF_NAMES):
            existing = getattr(edge.tolerance, dof_name)
            new_spec = ToleranceSpecModel(
                distribution=dist, bound=bound,
                sigma_level=sigma, locked=existing.locked,
            )
            setattr(edge.tolerance, dof_name, new_spec)
            self._rows[i].load(new_spec)
        self.project_changed.emit()

    def _on_bulk_apply_to_all_edges(self) -> None:
        if self._project is None:
            return
        dist = self._bulk_dist_combo.currentText()
        bound = self._bulk_bound_spin.value()
        sigma = self._bulk_sigma_spin.value()
        for edge in self._project.edges:
            for dof_name in _DOF_NAMES:
                existing = getattr(edge.tolerance, dof_name)
                new_spec = ToleranceSpecModel(
                    distribution=dist, bound=bound,
                    sigma_level=sigma, locked=existing.locked,
                )
                setattr(edge.tolerance, dof_name, new_spec)
        self._load_selected_edge()
        self.project_changed.emit()
