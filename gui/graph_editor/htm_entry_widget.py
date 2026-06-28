"""
gui/graph_editor/htm_entry_widget.py — Multi-format HTM entry with live validation.

Supports all four HTM input representations (Section 2.1):
  XYZ + Euler (ZYX) | 4×4 Matrix | Quaternion + XYZ | Screw Parameters

Live validation calls core/transforms.py's HTM constructors on every field change.
The OK button in any host dialog should be gated on is_valid().
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.transforms import HTM
from persistence.schema import (
    HTMInputMatrix,
    HTMInputModel,
    HTMInputQuaternion,
    HTMInputScrew,
    HTMInputXyzEuler,
)

_FORMAT_LABELS = [
    "XYZ + Euler (ZYX)",
    "4×4 Matrix",
    "Quaternion + XYZ",
    "Screw Parameters",
]

_IDX_XYZ_EULER = 0
_IDX_MATRIX = 1
_IDX_QUATERNION = 2
_IDX_SCREW = 3


def _spinbox(lo: float = -9999.0, hi: float = 9999.0, decimals: int = 6,
             val: float = 0.0) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setValue(val)
    sb.setSingleStep(0.001)
    return sb


class HTMEntryWidget(QWidget):
    """Multi-format nominal-transform entry widget with real-time validation feedback."""

    validation_changed = Signal(bool)  # emitted when validity flips; True = valid

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._valid = False
        self._setup_ui()
        self._validate()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        return self._valid

    def get_htm_input_model(self) -> HTMInputModel:
        """Return the current entry as the appropriate Pydantic discriminated-union variant.

        Raises ValueError if the current entry is not valid.
        """
        if not self._valid:
            raise ValueError("HTM entry is not currently valid")
        idx = self._format_selector.currentIndex()
        if idx == _IDX_XYZ_EULER:
            return self._build_xyz_euler_model()
        elif idx == _IDX_MATRIX:
            return self._build_matrix_model()
        elif idx == _IDX_QUATERNION:
            return self._build_quaternion_model()
        else:
            return self._build_screw_model()

    def set_htm_input_model(self, model: HTMInputModel) -> None:
        """Populate fields from an existing HTMInputModel (for editing)."""
        if isinstance(model, HTMInputXyzEuler):
            self._format_selector.setCurrentIndex(_IDX_XYZ_EULER)
            for i, sb in enumerate(self._xyz_euler_xyz):
                sb.setValue(model.xyz[i])
            for i, sb in enumerate(self._xyz_euler_angles):
                sb.setValue(model.euler_angles[i])
        elif isinstance(model, HTMInputMatrix):
            self._format_selector.setCurrentIndex(_IDX_MATRIX)
            for r in range(4):
                for c in range(4):
                    self._matrix_cells[r][c].setValue(model.matrix[r][c])
        elif isinstance(model, HTMInputQuaternion):
            self._format_selector.setCurrentIndex(_IDX_QUATERNION)
            for i, sb in enumerate(self._quat_wxyz):
                sb.setValue(model.quat_wxyz[i])
            for i, sb in enumerate(self._quat_xyz):
                sb.setValue(model.xyz[i])
        else:
            self._format_selector.setCurrentIndex(_IDX_SCREW)
            for i, sb in enumerate(self._screw_axis):
                sb.setValue(model.axis[i])
            self._screw_angle.setValue(model.angle)
            self._screw_translation.setValue(model.translation_along_axis)
            poa = model.point_on_axis or [0.0, 0.0, 0.0]
            for i, sb in enumerate(self._screw_point):
                sb.setValue(poa[i])
        self._validate()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._format_selector = QComboBox()
        self._format_selector.addItems(_FORMAT_LABELS)
        layout.addWidget(self._format_selector)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_xyz_euler_page())
        self._stack.addWidget(self._build_matrix_page())
        self._stack.addWidget(self._build_quaternion_page())
        self._stack.addWidget(self._build_screw_page())
        layout.addWidget(self._stack)

        self._validation_label = QLabel()
        layout.addWidget(self._validation_label)

        self._format_selector.currentIndexChanged.connect(self._on_format_changed)

    def _on_format_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        self._validate()

    def _build_xyz_euler_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._xyz_euler_xyz = [_spinbox() for _ in range(3)]
        import math as _math
        self._xyz_euler_angles = [_spinbox(-9999.0, 9999.0) for _ in range(3)]
        for sb in self._xyz_euler_angles:
            sb.setSingleStep(_math.pi / 12)  # 15° in radians
        for label, sb in zip(["x", "y", "z"], self._xyz_euler_xyz):
            form.addRow(label, sb)
            sb.valueChanged.connect(self._validate)
        for label, sb in zip(["ez (yaw, rad)", "ey (pitch, rad)", "ex (roll, rad)"],
                              self._xyz_euler_angles):
            form.addRow(label, sb)
            sb.valueChanged.connect(self._validate)
        return page

    def _build_matrix_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        self._matrix_cells: list[list[QDoubleSpinBox]] = []
        identity_diag = {(0, 0), (1, 1), (2, 2), (3, 3)}
        for r in range(4):
            row_cells: list[QDoubleSpinBox] = []
            for c in range(4):
                default = 1.0 if (r, c) in identity_diag else 0.0
                sb = _spinbox(-9999.0, 9999.0, 8, default)
                if r == 3:
                    sb.setEnabled(False)  # bottom row always [0,0,0,1]
                else:
                    sb.valueChanged.connect(self._validate)
                grid.addWidget(sb, r, c)
                row_cells.append(sb)
            self._matrix_cells.append(row_cells)
        return page

    def _build_quaternion_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._quat_wxyz = [_spinbox(-9999.0, 9999.0, 8,
                                    1.0 if i == 0 else 0.0)
                           for i in range(4)]
        self._quat_xyz = [_spinbox() for _ in range(3)]
        for label, sb in zip(["w", "x", "y", "z"], self._quat_wxyz):
            form.addRow(label, sb)
            sb.valueChanged.connect(self._validate)
        for label, sb in zip(["tx", "ty", "tz"], self._quat_xyz):
            form.addRow(label, sb)
            sb.valueChanged.connect(self._validate)
        return page

    def _build_screw_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._screw_axis = [_spinbox(-9999.0, 9999.0, 8,
                                     1.0 if i == 0 else 0.0)
                            for i in range(3)]
        self._screw_angle = _spinbox(-9999.0, 9999.0)
        self._screw_translation = _spinbox()
        self._screw_point = [_spinbox() for _ in range(3)]
        for label, sb in zip(["axis x", "axis y", "axis z"], self._screw_axis):
            form.addRow(label, sb)
            sb.valueChanged.connect(self._validate)
        form.addRow("angle (rad)", self._screw_angle)
        self._screw_angle.valueChanged.connect(self._validate)
        form.addRow("translation along axis", self._screw_translation)
        self._screw_translation.valueChanged.connect(self._validate)
        for label, sb in zip(["point x", "point y", "point z"], self._screw_point):
            form.addRow(label, sb)
            sb.valueChanged.connect(self._validate)
        return page

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        try:
            idx = self._format_selector.currentIndex()
            if idx == _IDX_XYZ_EULER:
                xyz = [sb.value() for sb in self._xyz_euler_xyz]
                ea = [sb.value() for sb in self._xyz_euler_angles]
                HTM.from_xyz_euler(xyz, ea)
            elif idx == _IDX_MATRIX:
                mat = [[self._matrix_cells[r][c].value() for c in range(4)]
                       for r in range(4)]
                HTM.from_matrix(np.array(mat))
            elif idx == _IDX_QUATERNION:
                q = [sb.value() for sb in self._quat_wxyz]
                xyz = [sb.value() for sb in self._quat_xyz]
                HTM.from_quaternion(q, xyz)
            else:
                axis = [sb.value() for sb in self._screw_axis]
                angle = self._screw_angle.value()
                t = self._screw_translation.value()
                poa = [sb.value() for sb in self._screw_point]
                HTM.from_screw(axis, angle, t, poa)
            self._set_valid(True, "")
        except Exception as exc:
            self._set_valid(False, str(exc))

    def _set_valid(self, valid: bool, message: str) -> None:
        changed = valid != self._valid
        self._valid = valid
        if valid:
            self._validation_label.setText("✓ Valid")
            self._validation_label.setStyleSheet("color: green;")
        else:
            short = message[:120] + "..." if len(message) > 120 else message
            self._validation_label.setText(f"✗ {short}")
            self._validation_label.setStyleSheet("color: red;")
        if changed:
            self.validation_changed.emit(valid)

    # ── Model builders ────────────────────────────────────────────────────────

    def _build_xyz_euler_model(self) -> HTMInputXyzEuler:
        return HTMInputXyzEuler(
            kind="xyz_euler",
            xyz=[sb.value() for sb in self._xyz_euler_xyz],
            euler_angles=[sb.value() for sb in self._xyz_euler_angles],
        )

    def _build_matrix_model(self) -> HTMInputMatrix:
        return HTMInputMatrix(
            kind="matrix",
            matrix=[[self._matrix_cells[r][c].value() for c in range(4)]
                    for r in range(4)],
        )

    def _build_quaternion_model(self) -> HTMInputQuaternion:
        return HTMInputQuaternion(
            kind="quaternion",
            quat_wxyz=[sb.value() for sb in self._quat_wxyz],
            xyz=[sb.value() for sb in self._quat_xyz],
        )

    def _build_screw_model(self) -> HTMInputScrew:
        return HTMInputScrew(
            kind="screw",
            axis=[sb.value() for sb in self._screw_axis],
            angle=self._screw_angle.value(),
            translation_along_axis=self._screw_translation.value(),
            point_on_axis=[sb.value() for sb in self._screw_point],
        )
