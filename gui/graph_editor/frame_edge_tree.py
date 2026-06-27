"""
gui/graph_editor/frame_edge_tree.py — QTreeWidget listing Frames and Edges.

Two top-level sections: "Frames" (with root/junction annotations) and "Edges".
Root frames (no incoming edge) are shown in bold blue with a [ROOT] prefix.
Junction frames (2+ incoming edges) are shown in orange with a [JUNCTION] prefix.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from persistence.schema import ProjectModel

_ROOT_COLOR = QColor("#1E6BBF")
_JUNCTION_COLOR = QColor("#C06000")


class FrameEdgeTree(QTreeWidget):
    """Two-section project-structure tree: Frames and Edges."""

    def __init__(self, parent: QTreeWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabel("Project Structure")
        self.setColumnCount(1)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)

        self._frames_root = QTreeWidgetItem(self, ["Frames"])
        self._frames_root.setExpanded(True)
        self._edges_root = QTreeWidgetItem(self, ["Edges"])
        self._edges_root.setExpanded(True)

    def refresh(self, project: ProjectModel) -> None:
        """Rebuild the tree from the current ProjectModel state."""
        self._frames_root.takeChildren()
        self._edges_root.takeChildren()

        incoming: dict[str, int] = {f.name: 0 for f in project.frames}
        for edge in project.edges:
            incoming[edge.child] = incoming.get(edge.child, 0) + 1

        root_brush = QBrush(_ROOT_COLOR)
        junction_brush = QBrush(_JUNCTION_COLOR)
        bold_font = QFont()
        bold_font.setBold(True)

        for frame in project.frames:
            count = incoming.get(frame.name, 0)
            if count == 0:
                item = QTreeWidgetItem(self._frames_root, [f"[ROOT] {frame.name}"])
                item.setForeground(0, root_brush)
                item.setFont(0, bold_font)
            elif count > 1:
                item = QTreeWidgetItem(self._frames_root, [f"[JUNCTION] {frame.name}"])
                item.setForeground(0, junction_brush)
            else:
                item = QTreeWidgetItem(self._frames_root, [frame.name])
            item.setData(0, Qt.ItemDataRole.UserRole, frame.name)

        for edge in project.edges:
            label = f"{edge.name}  ({edge.parent} → {edge.child})"
            item = QTreeWidgetItem(self._edges_root, [label])
            item.setData(0, Qt.ItemDataRole.UserRole, edge.name)

    def selected_item_info(self) -> tuple[str, str] | None:
        """Return (name, kind) for the selected item, or None.

        kind is 'frame' or 'edge'.
        """
        item = self.currentItem()
        if item is None:
            return None
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if name is None:
            return None
        parent = item.parent()
        if parent is self._frames_root:
            return (name, "frame")
        if parent is self._edges_root:
            return (name, "edge")
        return None
