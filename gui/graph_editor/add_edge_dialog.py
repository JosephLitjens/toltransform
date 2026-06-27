"""
gui/graph_editor/add_edge_dialog.py — Dialog for adding an HTMEdgeModel to the project.

Tolerance is NOT set here — the tolerance editor (C-2) handles that.
New edges are created with a placeholder all-locked-zero tolerance so they are safe
to forward-simulate without contributing error until the user assigns real bounds.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from gui.graph_editor.htm_entry_widget import HTMEntryWidget
from persistence.schema import (
    HTMEdgeModel,
    ProjectModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
)


def _default_tolerance() -> ToleranceSpec6Model:
    locked_zero = ToleranceSpecModel(distribution="uniform", bound=0.0, locked=True)
    return ToleranceSpec6Model(
        dx=locked_zero, dy=locked_zero, dz=locked_zero,
        rx=locked_zero, ry=locked_zero, rz=locked_zero,
    )


class AddEdgeDialog(QDialog):
    """Dialog for defining a new kinematic edge (parent → child, nominal HTM)."""

    def __init__(self, project: ProjectModel,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Edge")
        self._project = project
        self._result: HTMEdgeModel | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._edge_name_edit = QLineEdit()
        self._edge_name_edit.setPlaceholderText("e.g. base_to_sensor")
        form.addRow("Edge name:", self._edge_name_edit)

        frame_names = [f.name for f in self._project.frames]
        self._parent_combo = QComboBox()
        self._parent_combo.addItems(frame_names)
        form.addRow("Parent frame:", self._parent_combo)

        self._child_combo = QComboBox()
        self._child_combo.addItems(frame_names)
        if len(frame_names) >= 2:
            self._child_combo.setCurrentIndex(1)
        form.addRow("Child frame:", self._child_combo)

        layout.addLayout(form)

        htm_group = QGroupBox("Nominal Transform")
        htm_vbox = QVBoxLayout(htm_group)
        self._htm_entry = HTMEntryWidget()
        htm_vbox.addWidget(self._htm_entry)
        layout.addWidget(htm_group)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        layout.addWidget(self._error_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._edge_name_edit.textChanged.connect(self._update_ok)
        self._parent_combo.currentIndexChanged.connect(self._update_ok)
        self._child_combo.currentIndexChanged.connect(self._update_ok)
        self._htm_entry.validation_changed.connect(self._update_ok)

        self._update_ok()

    def _validate_fields(self) -> str:
        name = self._edge_name_edit.text().strip()
        if not name:
            return "Edge name is required."
        if name in {e.name for e in self._project.edges}:
            return f"Edge '{name}' already exists."
        if self._parent_combo.currentText() == self._child_combo.currentText():
            return "Parent and child frames must be different."
        if not self._htm_entry.is_valid():
            return "Nominal transform is not valid — check the entry above."
        return ""

    def _update_ok(self, *_) -> None:
        error = self._validate_fields()
        self._error_label.setText(error)
        ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setEnabled(not error)

    def _on_accept(self) -> None:
        if self._validate_fields():
            return
        self._result = HTMEdgeModel(
            name=self._edge_name_edit.text().strip(),
            parent=self._parent_combo.currentText(),
            child=self._child_combo.currentText(),
            nominal=self._htm_entry.get_htm_input_model(),
            tolerance=_default_tolerance(),
        )
        self.accept()

    def result_edge(self) -> HTMEdgeModel:
        """Return the created HTMEdgeModel. Only valid after Accepted."""
        if self._result is None:
            raise RuntimeError("result_edge() called before dialog was accepted")
        return self._result
