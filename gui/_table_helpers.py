"""Shared read-only table helpers for GUI panels that display DoF envelope data."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from postprocess.stats import DOF_LABELS


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
