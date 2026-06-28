"""
gui/graph_editor/edit_edge_dialog.py — Dialog for editing an existing HTMEdgeModel (D-2).

Allows changing edge name and nominal transform.  Parent/child frames are shown as a
read-only label — structural changes (rerouting edges) require delete + recreate to
maintain DAG integrity.  Tolerance is unchanged and stays in the tolerance editor.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
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
from persistence.schema import HTMEdgeModel, ProjectModel


class EditEdgeDialog(QDialog):
    """Pre-populated dialog for editing the name and nominal transform of an edge."""

    def __init__(
        self,
        edge: HTMEdgeModel,
        project: ProjectModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit Edge — {edge.name}")
        self._edge = edge
        self._project = project
        self._original_name = edge.name
        self._result: HTMEdgeModel | None = None
        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Parent → child (read-only)
        route_label = QLabel(f"{self._edge.parent}  →  {self._edge.child}")
        route_label.setStyleSheet("color: gray;")
        form.addRow("Route (read-only):", route_label)

        # Edge name (editable)
        self._name_edit = QLineEdit(self._edge.name)
        form.addRow("Edge name:", self._name_edit)
        layout.addLayout(form)

        # Nominal transform
        htm_group = QGroupBox("Nominal Transform")
        from PySide6.QtWidgets import QVBoxLayout as _VBox
        htm_vbox = _VBox(htm_group)
        self._htm_entry = HTMEntryWidget()
        self._htm_entry.set_htm_input_model(self._edge.nominal)
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

        self._name_edit.textChanged.connect(self._update_ok)
        self._htm_entry.validation_changed.connect(self._update_ok)
        self._update_ok()

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_fields(self) -> str:
        name = self._name_edit.text().strip()
        if not name:
            return "Edge name is required."
        # Collision check: ignore the original name (we're editing it)
        existing = {e.name for e in self._project.edges if e.name != self._original_name}
        if name in existing:
            return f"An edge named '{name}' already exists."
        if not self._htm_entry.is_valid():
            return "Nominal transform is not valid."
        return ""

    def _update_ok(self, *_) -> None:
        error = self._validate_fields()
        self._error_label.setText(error)
        ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setEnabled(not error)

    # ── Accept ─────────────────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        if self._validate_fields():
            return
        self._result = HTMEdgeModel(
            name=self._name_edit.text().strip(),
            parent=self._edge.parent,
            child=self._edge.child,
            nominal=self._htm_entry.get_htm_input_model(),
            tolerance=self._edge.tolerance,   # tolerance unchanged
        )
        self.accept()

    def result_edge(self) -> HTMEdgeModel:
        """Return the edited HTMEdgeModel. Only valid after Accepted."""
        if self._result is None:
            raise RuntimeError("result_edge() called before dialog was accepted")
        return self._result
