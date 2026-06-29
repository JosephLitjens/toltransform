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
# 7 columns: DoF | Mode | Distribution | Bound/Range | σ-level | Locked | Error
_GRID_HEADERS = ("DoF", "", "Distribution", "Bound / Range", "σ-level", "Locked", "")

_SPIN_STEP = 0.0001
_SPIN_DECIMALS = 6


def _make_bound_spin(minimum: float = 0.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimum, 9999.0)
    s.setDecimals(_SPIN_DECIMALS)
    s.setSingleStep(_SPIN_STEP)
    return s


class _DofRow:
    """Controls for one DoF within the tolerance editor grid.

    Not a QWidget — widgets are added directly to the parent QGridLayout.

    Two input modes (toggled per-row):
      Symmetric (±):   one bound spinbox, samples from [-bound, +bound].
      Asymmetric (↔):  two spinboxes (lower, upper), samples from [lower, upper].

    When switching ± → ↔ the lower/upper spinboxes are pre-populated from ±bound.
    When switching ↔ → ± the bound spinbox is set to max(|lower|, |upper|).
    """

    def __init__(self, grid: QGridLayout, grid_row: int, dof_name: str) -> None:
        self._loading = False

        # Col 0 — DoF label
        lbl = QLabel(f"<b>{dof_name}</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(lbl, grid_row, 0)

        # Col 1 — Mode toggle button
        self.mode_btn = QPushButton("±")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setChecked(False)   # False = symmetric (±)
        self.mode_btn.setFixedWidth(38)
        self.mode_btn.setToolTip(
            "Toggle between symmetric ±bound and asymmetric min/max mode.\n"
            "Asymmetric mode is for FK only; IK targets stay ±."
        )
        grid.addWidget(self.mode_btn, grid_row, 1)

        # Col 2 — Distribution combo
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["uniform", "normal"])
        self.dist_combo.setMinimumWidth(90)
        grid.addWidget(self.dist_combo, grid_row, 2)

        # Col 3 — Stacked: page 0 = bound spinbox, page 1 = lower+upper
        self._bound_stack = QStackedWidget()

        self.bound_spin = _make_bound_spin(minimum=0.0)
        self._bound_stack.addWidget(self.bound_spin)

        asym_widget = QWidget()
        asym_layout = QHBoxLayout(asym_widget)
        asym_layout.setContentsMargins(0, 0, 0, 0)
        asym_layout.setSpacing(4)
        self.lower_spin = _make_bound_spin(minimum=-9999.0)
        self.lower_spin.setToolTip("Lower bound (min)")
        self.upper_spin = _make_bound_spin(minimum=-9999.0)
        self.upper_spin.setToolTip("Upper bound (max)")
        lower_lbl = QLabel("min")
        lower_lbl.setStyleSheet("color: gray; font-size: 10px;")
        upper_lbl = QLabel("max")
        upper_lbl.setStyleSheet("color: gray; font-size: 10px;")
        asym_layout.addWidget(lower_lbl)
        asym_layout.addWidget(self.lower_spin)
        asym_layout.addWidget(upper_lbl)
        asym_layout.addWidget(self.upper_spin)
        self._bound_stack.addWidget(asym_widget)

        grid.addWidget(self._bound_stack, grid_row, 3)

        # Col 4 — σ-level
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.001, 99.0)
        self.sigma_spin.setDecimals(3)
        self.sigma_spin.setValue(3.0)
        self.sigma_spin.setEnabled(False)
        self.sigma_spin.setStyleSheet("color: gray;")
        grid.addWidget(self.sigma_spin, grid_row, 4)

        # Col 5 — Locked checkbox
        self.locked_check = QCheckBox()
        grid.addWidget(self.locked_check, grid_row, 5, Qt.AlignmentFlag.AlignCenter)

        # Col 6 — Inline error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red; font-size: 11px;")
        self.error_label.setWordWrap(True)
        grid.addWidget(self.error_label, grid_row, 6)

        self.dist_combo.currentTextChanged.connect(self._on_dist_changed)
        self.mode_btn.clicked.connect(self._on_mode_toggled)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_asymmetric(self) -> bool:
        return self.mode_btn.isChecked()

    # ── Internal handlers ─────────────────────────────────────────────────────

    def _on_dist_changed(self, dist: str) -> None:
        is_normal = dist == "normal"
        self.sigma_spin.setEnabled(is_normal)
        self.sigma_spin.setStyleSheet("" if is_normal else "color: gray;")

    def _on_mode_toggled(self, checked: bool) -> None:
        if checked:
            # Switching to asymmetric: pre-populate lower/upper from ±bound
            b = self.bound_spin.value()
            self.lower_spin.blockSignals(True)
            self.upper_spin.blockSignals(True)
            self.lower_spin.setValue(-b)
            self.upper_spin.setValue(b)
            self.lower_spin.blockSignals(False)
            self.upper_spin.blockSignals(False)
            self._bound_stack.setCurrentIndex(1)
            self.mode_btn.setText("↔")
        else:
            # Switching to symmetric: set bound = max(|lower|, |upper|)
            b = max(abs(self.lower_spin.value()), abs(self.upper_spin.value()))
            self.bound_spin.blockSignals(True)
            self.bound_spin.setValue(b)
            self.bound_spin.blockSignals(False)
            self._bound_stack.setCurrentIndex(0)
            self.mode_btn.setText("±")

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, spec: ToleranceSpecModel) -> None:
        """Populate row from a ToleranceSpecModel (either symmetric or asymmetric)."""
        all_widgets = (
            self.dist_combo, self.bound_spin, self.lower_spin,
            self.upper_spin, self.sigma_spin, self.locked_check, self.mode_btn,
        )
        for w in all_widgets:
            w.blockSignals(True)

        self.dist_combo.setCurrentText(spec.distribution)
        self.sigma_spin.setValue(spec.sigma_level)
        self.locked_check.setChecked(spec.locked)
        self._on_dist_changed(spec.distribution)

        if spec.lower is not None and spec.upper is not None:
            self.lower_spin.setValue(spec.lower)
            self.upper_spin.setValue(spec.upper)
            self.bound_spin.setValue(spec.bound)
            self._bound_stack.setCurrentIndex(1)
            self.mode_btn.setChecked(True)
            self.mode_btn.setText("↔")
        else:
            self.bound_spin.setValue(spec.bound)
            self._bound_stack.setCurrentIndex(0)
            self.mode_btn.setChecked(False)
            self.mode_btn.setText("±")

        for w in all_widgets:
            w.blockSignals(False)
        self.error_label.setText("")

    def get_model(self) -> ToleranceSpecModel:
        """Return the current ToleranceSpecModel from widget state."""
        dist = self.dist_combo.currentText()
        sigma = self.sigma_spin.value()
        locked = self.locked_check.isChecked()
        if self.is_asymmetric:
            lo, hi = self.lower_spin.value(), self.upper_spin.value()
            bound = max(abs(lo), abs(hi))
            return ToleranceSpecModel(
                distribution=dist, bound=bound,
                sigma_level=sigma, locked=locked,
                lower=lo, upper=hi,
            )
        return ToleranceSpecModel(
            distribution=dist,
            bound=self.bound_spin.value(),
            sigma_level=sigma,
            locked=locked,
        )

    def is_valid(self) -> bool:
        """Validate current widget state using ToleranceSpec as the validator."""
        try:
            if self.is_asymmetric:
                lo, hi = self.lower_spin.value(), self.upper_spin.value()
                ToleranceSpec(
                    distribution=self.dist_combo.currentText(),
                    sigma_level=self.sigma_spin.value(),
                    lower=lo, upper=hi,
                )
            else:
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
        self.lower_spin.valueChanged.connect(slot)
        self.upper_spin.valueChanged.connect(slot)
        self.sigma_spin.valueChanged.connect(slot)
        self.locked_check.stateChanged.connect(slot)
        self.mode_btn.clicked.connect(slot)


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
        grid.setColumnStretch(6, 1)
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
