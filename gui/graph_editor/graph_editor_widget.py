"""
gui/graph_editor/graph_editor_widget.py — Top-level widget for the graph/chain editor.

Hosts the FrameEdgeTree and action buttons. Writes directly to the in-memory
ProjectModel (Section 5.3 — GUI only touches persistence.schema objects during editing).

Emits project_changed after any mutation so MainWindow can set the dirty flag.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.graph_editor.add_edge_dialog import AddEdgeDialog
from gui.graph_editor.add_frame_dialog import AddFrameDialog
from gui.graph_editor.frame_edge_tree import FrameEdgeTree
from persistence.schema import ProjectModel


class GraphEditorWidget(QWidget):
    """Frame/Edge graph editor panel."""

    project_changed = Signal()
    edge_selected = Signal(str)  # edge name; emitted when user clicks an edge in the tree

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel | None = None
        self._setup_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_project(self, project: ProjectModel) -> None:
        self._project = project
        self.refresh_view()

    def refresh_view(self) -> None:
        if self._project is not None:
            self._tree.refresh(self._project)

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._tree = FrameEdgeTree()
        layout.addWidget(self._tree, stretch=1)

        btn_row = QHBoxLayout()
        self._add_frame_btn = QPushButton("+ Frame")
        self._add_edge_btn = QPushButton("+ Edge")
        self._delete_btn = QPushButton("Delete Selected")
        btn_row.addWidget(self._add_frame_btn)
        btn_row.addWidget(self._add_edge_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._delete_btn)
        layout.addLayout(btn_row)

        self._add_frame_btn.clicked.connect(self._on_add_frame)
        self._add_edge_btn.clicked.connect(self._on_add_edge)
        self._delete_btn.clicked.connect(self._on_delete_selected)
        self._tree.currentItemChanged.connect(self._on_tree_selection_changed)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_tree_selection_changed(self, current, previous) -> None:
        info = self._tree.selected_item_info()
        if info is not None and info[1] == "edge":
            self.edge_selected.emit(info[0])

    def _on_add_frame(self) -> None:
        if self._project is None:
            return
        dlg = AddFrameDialog(self._project, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._project.frames.append(dlg.result_frame())
        self.refresh_view()
        self.project_changed.emit()

    def _on_add_edge(self) -> None:
        if self._project is None:
            return
        if len(self._project.frames) < 2:
            QMessageBox.information(
                self, "Add Edge",
                "You need at least two frames before you can add an edge."
            )
            return
        dlg = AddEdgeDialog(self._project, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._project.edges.append(dlg.result_edge())
        self.refresh_view()
        self.project_changed.emit()

    def _on_delete_selected(self) -> None:
        if self._project is None:
            return
        info = self._tree.selected_item_info()
        if info is None:
            return
        name, kind = info

        if kind == "edge":
            to_remove = next((e for e in self._project.edges if e.name == name), None)
            if to_remove:
                self._project.edges.remove(to_remove)
                self.refresh_view()
                self.project_changed.emit()
        elif kind == "frame":
            referencing = [e.name for e in self._project.edges
                           if e.parent == name or e.child == name]
            if referencing:
                QMessageBox.warning(
                    self, "Cannot Delete Frame",
                    f"Frame '{name}' is used by edge(s): {', '.join(referencing)}.\n"
                    "Delete those edges first."
                )
                return
            to_remove = next((f for f in self._project.frames if f.name == name), None)
            if to_remove:
                self._project.frames.remove(to_remove)
                self.refresh_view()
                self.project_changed.emit()


