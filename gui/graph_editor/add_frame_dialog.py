"""
gui/graph_editor/add_frame_dialog.py — Dialog for adding a FrameModel to the project.

Validates: non-empty name, no duplicate names.
OK button is disabled until validation passes.
Caller (GraphEditorWidget) is responsible for appending result_frame() to ProjectModel.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from persistence.schema import FrameModel, ProjectModel


class AddFrameDialog(QDialog):
    """Single-field dialog for adding a named Frame to the project."""

    def __init__(self, project: ProjectModel,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Frame")
        self._project = project
        self._result: FrameModel | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. base, sensor, tip")
        form.addRow("Frame name:", self._name_edit)
        layout.addLayout(form)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        layout.addWidget(self._error_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(False)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._name_edit.textChanged.connect(self._on_name_changed)

    def _on_name_changed(self, text: str) -> None:
        name = text.strip()
        existing = {f.name for f in self._project.frames}
        if not name:
            self._error_label.setText("")
            self._ok_button().setEnabled(False)
        elif name in existing:
            self._error_label.setText(f"Frame '{name}' already exists.")
            self._ok_button().setEnabled(False)
        else:
            self._error_label.setText("")
            self._ok_button().setEnabled(True)

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name or name in {f.name for f in self._project.frames}:
            return
        self._result = FrameModel(name=name)
        self.accept()

    def result_frame(self) -> FrameModel:
        """Return the created FrameModel. Only valid after Accepted."""
        if self._result is None:
            raise RuntimeError("result_frame() called before dialog was accepted")
        return self._result

    def _ok_button(self):
        return self._buttons.button(QDialogButtonBox.StandardButton.Ok)
