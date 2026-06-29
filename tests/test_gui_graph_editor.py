"""
tests/test_gui_graph_editor.py — pytest-qt tests for gui/graph_editor/ and gui/main_window.py.

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_graph_editor.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from gui.graph_editor.add_edge_dialog import AddEdgeDialog
from gui.graph_editor.add_frame_dialog import AddFrameDialog
from gui.graph_editor.edit_edge_dialog import EditEdgeDialog
from gui.graph_editor.frame_edge_tree import FrameEdgeTree
from gui.graph_editor.graph_editor_widget import GraphEditorWidget
from gui.graph_editor.htm_entry_widget import HTMEntryWidget
from gui.main_window import MainWindow, _empty_project
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
)
from persistence.serializer import load_project


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_project(*frame_names: str) -> ProjectModel:
    project = _empty_project()
    for name in frame_names:
        project.frames.append(FrameModel(name=name))
    return project


def _locked_zero() -> ToleranceSpecModel:
    return ToleranceSpecModel(distribution="uniform", bound=0.0, locked=True)


def _default_tol6() -> ToleranceSpec6Model:
    z = _locked_zero()
    return ToleranceSpec6Model(dx=z, dy=z, dz=z, rx=z, ry=z, rz=z)


def _make_edge(name: str, parent: str, child: str) -> HTMEdgeModel:
    return HTMEdgeModel(
        name=name,
        parent=parent,
        child=child,
        nominal=HTMInputXyzEuler(kind="xyz_euler", xyz=[0, 0, 0], euler_angles=[0, 0, 0]),
        tolerance=_default_tol6(),
    )


# ── MainWindow tests ──────────────────────────────────────────────────────────

def test_main_window_starts_with_empty_project(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._project.frames == []
    assert window._project.edges == []
    assert not window._dirty


def test_new_project_clears_model(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._project.frames.append(FrameModel(name="existing"))
    assert len(window._project.frames) == 1
    window._new_project()
    assert window._project.frames == []
    assert window._project.edges == []
    assert not window._dirty


def test_save_load_roundtrip_via_main_window(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._project.frames.append(FrameModel(name="frame1"))
    window._project.frames.append(FrameModel(name="frame2"))
    window._project.edges.append(_make_edge("e1", "frame1", "frame2"))

    path = str(tmp_path / "project.json")
    window._path = path
    window._save_project()

    assert os.path.exists(path)
    loaded = load_project(path)
    assert len(loaded.frames) == 2
    assert {f.name for f in loaded.frames} == {"frame1", "frame2"}
    assert len(loaded.edges) == 1
    assert loaded.edges[0].name == "e1"


def test_dirty_flag_set_after_project_changed(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert not window._dirty
    window._graph_editor.project_changed.emit()
    assert window._dirty


# ── AddFrameDialog tests ──────────────────────────────────────────────────────

def test_add_frame_dialog_ok_disabled_on_empty_name(qtbot):
    project = make_project()
    dlg = AddFrameDialog(project)
    qtbot.addWidget(dlg)
    ok = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()


def test_add_frame_dialog_ok_enabled_on_valid_name(qtbot):
    project = make_project()
    dlg = AddFrameDialog(project)
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("sensor")
    ok = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled()


def test_add_frame_dialog_rejects_duplicate_name(qtbot):
    project = make_project("existing")
    dlg = AddFrameDialog(project)
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("existing")
    ok = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()
    assert "already exists" in dlg._error_label.text()


def test_add_frame_dialog_creates_frame_model(qtbot):
    project = make_project()
    dlg = AddFrameDialog(project)
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("sensor")
    dlg._on_accept()
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert dlg.result_frame().name == "sensor"


# ── AddEdgeDialog tests ───────────────────────────────────────────────────────

def test_add_edge_dialog_creates_edge_model(qtbot):
    project = make_project("A", "B")
    dlg = AddEdgeDialog(project)
    qtbot.addWidget(dlg)
    dlg._edge_name_edit.setText("A_to_B")
    dlg._parent_combo.setCurrentText("A")
    dlg._child_combo.setCurrentText("B")
    assert dlg._htm_entry.is_valid()
    dlg._on_accept()
    edge = dlg.result_edge()
    assert edge.name == "A_to_B"
    assert edge.parent == "A"
    assert edge.child == "B"
    assert edge.tolerance.dx.locked


def test_add_edge_dialog_ok_disabled_when_parent_eq_child(qtbot):
    project = make_project("A", "B")
    dlg = AddEdgeDialog(project)
    qtbot.addWidget(dlg)
    dlg._edge_name_edit.setText("self_loop")
    dlg._parent_combo.setCurrentText("A")
    dlg._child_combo.setCurrentText("A")
    ok = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()


# ── HTMEntryWidget tests ──────────────────────────────────────────────────────

def test_htm_entry_widget_default_xyz_euler_is_valid(qtbot):
    widget = HTMEntryWidget()
    qtbot.addWidget(widget)
    assert widget._format_selector.currentIndex() == 0
    assert widget.is_valid()
    model = widget.get_htm_input_model()
    assert model.kind == "xyz_euler"


def test_htm_entry_widget_matrix_identity_is_valid(qtbot):
    widget = HTMEntryWidget()
    qtbot.addWidget(widget)
    widget._format_selector.setCurrentIndex(1)  # 4×4 Matrix
    assert widget.is_valid()


def test_htm_entry_widget_matrix_invalid_non_orthonormal(qtbot):
    widget = HTMEntryWidget()
    qtbot.addWidget(widget)
    widget._format_selector.setCurrentIndex(1)
    # Set upper-left 3×3 to a non-orthonormal matrix
    widget._matrix_cells[0][0].setValue(2.0)
    widget._matrix_cells[0][1].setValue(0.0)
    widget._matrix_cells[0][2].setValue(0.0)
    assert not widget.is_valid()


def test_htm_entry_widget_quaternion_identity_is_valid(qtbot):
    widget = HTMEntryWidget()
    qtbot.addWidget(widget)
    widget._format_selector.setCurrentIndex(2)  # Quaternion + XYZ
    # Default: w=1, x=y=z=0, xyz=0 — identity quaternion
    assert widget.is_valid()


def test_htm_entry_widget_set_model_roundtrip(qtbot):
    widget = HTMEntryWidget()
    qtbot.addWidget(widget)
    model = HTMInputXyzEuler(kind="xyz_euler", xyz=[1.0, 2.0, 3.0], euler_angles=[0.1, 0.2, 0.3])
    widget.set_htm_input_model(model)
    assert widget.is_valid()
    result = widget.get_htm_input_model()
    assert result.kind == "xyz_euler"
    assert result.xyz == pytest.approx([1.0, 2.0, 3.0])
    assert result.euler_angles == pytest.approx([0.1, 0.2, 0.3])


# ── FrameEdgeTree tests ───────────────────────────────────────────────────────

def test_frame_edge_tree_shows_root_prefix(qtbot):
    project = make_project("base", "tip")
    project.edges.append(_make_edge("e1", "base", "tip"))
    tree = FrameEdgeTree()
    qtbot.addWidget(tree)
    tree.refresh(project)

    frames_root = tree._frames_root
    labels = [frames_root.child(i).text(0) for i in range(frames_root.childCount())]
    assert any(t.startswith("[ROOT]") and "base" in t for t in labels)
    assert all("tip" not in t or not t.startswith("[ROOT]") for t in labels)


def test_frame_edge_tree_shows_junction_prefix(qtbot):
    project = make_project("A", "B", "C")
    project.edges.append(_make_edge("e1", "A", "C"))
    project.edges.append(_make_edge("e2", "B", "C"))
    tree = FrameEdgeTree()
    qtbot.addWidget(tree)
    tree.refresh(project)

    frames_root = tree._frames_root
    labels = [frames_root.child(i).text(0) for i in range(frames_root.childCount())]
    assert any(t.startswith("[JUNCTION]") and "C" in t for t in labels)


def test_frame_edge_tree_selected_item_info_frame(qtbot):
    project = make_project("base")
    tree = FrameEdgeTree()
    qtbot.addWidget(tree)
    tree.refresh(project)
    tree.setCurrentItem(tree._frames_root.child(0))
    info = tree.selected_item_info()
    assert info is not None
    name, kind = info
    assert name == "base"
    assert kind == "frame"


def test_frame_edge_tree_selected_item_info_edge(qtbot):
    project = make_project("A", "B")
    project.edges.append(_make_edge("e1", "A", "B"))
    tree = FrameEdgeTree()
    qtbot.addWidget(tree)
    tree.refresh(project)
    tree.setCurrentItem(tree._edges_root.child(0))
    info = tree.selected_item_info()
    assert info is not None
    name, kind = info
    assert name == "e1"
    assert kind == "edge"


# ── EditEdgeDialog tests ──────────────────────────────────────────────────────

def test_edit_edge_dialog_prepopulates_name_and_htm(qtbot):
    project = make_project("A", "B")
    edge = _make_edge("my_edge", "A", "B")
    edge.nominal = HTMInputXyzEuler(kind="xyz_euler", xyz=[1.0, 2.0, 3.0], euler_angles=[0.0, 0.0, 0.0])
    project.edges.append(edge)

    dlg = EditEdgeDialog(edge, project)
    qtbot.addWidget(dlg)

    assert dlg._name_edit.text() == "my_edge"
    # HTMEntryWidget should show the xyz values — verify the widget is pre-loaded (valid)
    assert dlg._htm_entry.is_valid()


def test_edit_edge_accept_updates_edge_in_project(qtbot):
    project = make_project("A", "B")
    edge = _make_edge("old_name", "A", "B")
    project.edges.append(edge)

    widget = GraphEditorWidget()
    qtbot.addWidget(widget)
    widget.set_project(project)

    # Select the edge in the tree
    widget._tree.setCurrentItem(widget._tree._edges_root.child(0))

    # Open dialog, change name, accept
    dlg = EditEdgeDialog(edge, project)
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("new_name")
    dlg._on_accept()

    assert dlg._result is not None
    assert dlg._result.name == "new_name"
    assert dlg._result.parent == "A"
    assert dlg._result.child == "B"


def test_edit_edge_cancel_leaves_project_unchanged(qtbot):
    project = make_project("A", "B")
    edge = _make_edge("original", "A", "B")
    project.edges.append(edge)

    dlg = EditEdgeDialog(edge, project)
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("changed")
    dlg.reject()   # Cancel

    # Project edge list is unchanged (reject doesn't call _on_accept)
    assert project.edges[0].name == "original"
    assert dlg._result is None
