"""
gui/main_window.py — Top-level QMainWindow for TolTransform (Section 6.13).

Owns the in-memory ProjectModel, hosts all GUI panels as dock widgets,
and provides File > New / Open / Save / Save As actions via persistence/serializer.py.

Panels hosted here:
  - GraphEditorWidget (C-1) — left dock
  - ToleranceEditorWidget (C-2) — right dock
  - RunPanelWidget (C-3) — right dock
  - ResultsViewerWidget (C-4) — right dock
  - PointPairPanelWidget (C-5) — right dock

Window geometry, dock layout, and the Recent Files list (capped at 5) are
persisted between sessions via QSettings("TolTransform", "TolTransform").
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from gui.graph_editor.graph_editor_widget import GraphEditorWidget
from gui.point_pair_panel.point_pair_panel_widget import PointPairPanelWidget
from gui.results_viewer.results_viewer_widget import ResultsViewerWidget
from gui.run_panel.run_panel_widget import RunPanelWidget
from gui.tolerance_editor.tolerance_editor_widget import ToleranceEditorWidget
from persistence.schema import ProjectModel, SimSettingsModel
from persistence.serializer import ProjectLoadError, load_project, save_project

_MAX_RECENT = 5


def _empty_project() -> ProjectModel:
    return ProjectModel(
        sim_settings=SimSettingsModel(
            mode="fk_verification",
            n_trials=10000,
            seed=42,
        )
    )


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: ProjectModel = _empty_project()
        self._path: str | None = None
        self._dirty: bool = False
        self._last_run_result: object = None
        self._recent_files: list[str] = []
        self._frame_viewer = None   # lazily created in _toggle_frame_viewer
        self._setup_ui()
        self._restore_settings()
        self._update_title()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.resize(1200, 800)
        self._setup_menu_bar()
        self.statusBar().showMessage("Ready")
        self._setup_docks()

    def _setup_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("&New", self._new_project, "Ctrl+N")
        file_menu.addAction("&Open...", self._open_project, "Ctrl+O")
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction("&Save", self._save_project, "Ctrl+S")
        file_menu.addAction("Save &As...", self._save_project_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("3D Frame Viewer", self._toggle_frame_viewer, "Ctrl+3")

    def _setup_docks(self) -> None:
        self._graph_editor = GraphEditorWidget()
        self._graph_editor.project_changed.connect(self._on_graph_editor_changed)
        graph_dock = QDockWidget("Graph Editor", self)
        graph_dock.setObjectName("GraphEditorDock")
        graph_dock.setWidget(self._graph_editor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, graph_dock)

        # Tolerance Editor (C-2)
        self._tolerance_editor = ToleranceEditorWidget()
        self._tolerance_editor.project_changed.connect(self._on_project_changed)
        tol_dock = QDockWidget("Tolerance Editor", self)
        tol_dock.setObjectName("ToleranceEditorDock")
        tol_dock.setWidget(self._tolerance_editor)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, tol_dock)

        # Wire graph editor → tolerance editor: clicking an edge auto-selects it
        self._graph_editor.edge_selected.connect(self._tolerance_editor.set_selected_edge)

        # Run Panel (C-3)
        self._run_panel = RunPanelWidget()
        self._run_panel.project_changed.connect(self._on_project_changed)
        self._run_panel.run_completed.connect(self._on_run_completed)
        self._run_panel.run_failed.connect(self._on_run_failed)
        run_dock = QDockWidget("Run Panel", self)
        run_dock.setObjectName("RunPanelDock")
        run_dock.setWidget(self._run_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, run_dock)

        # Results Viewer (C-4)
        self._results_viewer = ResultsViewerWidget()
        self._results_viewer.project_changed.connect(self._on_allocation_applied)
        rv_dock = QDockWidget("Results Viewer", self)
        rv_dock.setObjectName("ResultsViewerDock")
        rv_dock.setWidget(self._results_viewer)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, rv_dock)

        # Point-Pair Analysis (C-5)
        self._point_pair_panel = PointPairPanelWidget()
        self._point_pair_panel.project_changed.connect(self._on_project_changed)
        pp_dock = QDockWidget("Point-Pair Analysis", self)
        pp_dock.setObjectName("PointPairAnalysisDock")
        pp_dock.setWidget(self._point_pair_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, pp_dock)

        self._graph_editor.set_project(self._project)
        self._tolerance_editor.set_project(self._project)
        self._run_panel.set_project(self._project)
        self._point_pair_panel.set_project(self._project)

    # ── Settings persistence ──────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        s = QSettings("TolTransform", "TolTransform")
        geom = s.value("window/geometry")
        if geom:
            self.restoreGeometry(geom)
        state = s.value("window/state")
        if state:
            self.restoreState(state)
        self._recent_files = s.value("recentFiles", []) or []
        self._rebuild_recent_menu()

    def _save_settings(self) -> None:
        s = QSettings("TolTransform", "TolTransform")
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/state", self.saveState())
        s.setValue("recentFiles", self._recent_files)

    # ── Recent Files ──────────────────────────────────────────────────────────

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        if not self._recent_files:
            action = self._recent_menu.addAction("(No recent files)")
            action.setEnabled(False)
            return
        for path in self._recent_files:
            action = self._recent_menu.addAction(os.path.basename(path))
            action.setData(path)
            action.setToolTip(path)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
        self._recent_menu.addSeparator()
        self._recent_menu.addAction("Clear Recent", self._clear_recent_files)

    def _add_recent(self, path: str) -> None:
        self._recent_files = [path] + [p for p in self._recent_files if p != path]
        self._recent_files = self._recent_files[:_MAX_RECENT]
        self._rebuild_recent_menu()

    def _remove_recent(self, path: str) -> None:
        self._recent_files = [p for p in self._recent_files if p != path]
        self._rebuild_recent_menu()

    def _clear_recent_files(self) -> None:
        self._recent_files = []
        self._rebuild_recent_menu()

    def _apply_project_to_ui(self) -> None:
        """Push self._project into all panels and reset transient viewer state."""
        self._graph_editor.set_project(self._project)
        self._tolerance_editor.set_project(self._project)
        self._run_panel.set_project(self._project)
        self._point_pair_panel.set_project(self._project)
        self._results_viewer.clear()
        if self._frame_viewer is not None:
            self._frame_viewer.update_graph(self._project)

    def _open_recent(self, path: str) -> None:
        if not self._confirm_discard_changes():
            return
        if not os.path.exists(path):
            self._show_error("File Not Found", f"'{path}' no longer exists.")
            self._remove_recent(path)
            return
        try:
            project = load_project(path)
        except ProjectLoadError as exc:
            self._show_error("Cannot Open Project", str(exc))
            return
        self._project = project
        self._path = path
        self._apply_project_to_ui()
        self._set_dirty(False)
        self._add_recent(path)
        self.statusBar().showMessage(f"Opened: {os.path.basename(path)}")

    # ── File actions ──────────────────────────────────────────────────────────

    def _new_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        self._project = _empty_project()
        self._path = None
        self._apply_project_to_ui()
        if self._frame_viewer is not None:
            self._frame_viewer.clear()
        self._set_dirty(False)
        self.statusBar().showMessage("New project created")

    def _open_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "TolTransform Project (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            project = load_project(path)
        except ProjectLoadError as exc:
            self._show_error("Cannot Open Project", str(exc))
            return
        self._project = project
        self._path = path
        self._apply_project_to_ui()
        self._set_dirty(False)
        self._add_recent(path)
        self.statusBar().showMessage(f"Opened: {os.path.basename(path)}")

    def _save_project(self) -> None:
        if self._path is None:
            self._save_project_as()
            return
        try:
            save_project(self._project, self._path)
        except Exception as exc:
            self._show_error("Cannot Save Project", str(exc))
            return
        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved: {os.path.basename(self._path)}")

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "",
            "TolTransform Project (*.json);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        self._path = path
        self._save_project()
        self._add_recent(path)

    # ── State management ──────────────────────────────────────────────────────

    def _on_graph_editor_changed(self) -> None:
        """Called when graph editor mutates the project (frames/edges added or removed)."""
        self._set_dirty(True)
        self._tolerance_editor.refresh_view()
        self._run_panel.refresh_view()
        self._point_pair_panel.refresh_view()
        if self._frame_viewer is not None:
            self._frame_viewer.update_graph(self._project)

    def _on_run_completed(self, result: object) -> None:
        self._last_run_result = result
        self._results_viewer.set_result(result, self._project)
        self._point_pair_panel.set_result(result)
        if self._frame_viewer is not None:
            self._frame_viewer.set_result(result)

    def _on_run_failed(self, error: str) -> None:
        pass  # RunPanelWidget already shows the error in its own status label

    def _on_project_changed(self) -> None:
        self._set_dirty(True)

    def _on_allocation_applied(self) -> None:
        """Called when the results viewer writes IK allocation bounds back to the project."""
        self._set_dirty(True)
        self._tolerance_editor.refresh_view()

    def _set_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        self._update_title()

    def _update_title(self) -> None:
        base = "TolTransform"
        if self._path:
            base = f"TolTransform — {os.path.basename(self._path)}"
        self.setWindowTitle(f"{base} *" if self._dirty else base)

    def _confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Discard them and continue?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._save_project()
            return not self._dirty
        return reply == QMessageBox.StandardButton.Discard

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _toggle_frame_viewer(self) -> None:
        if self._frame_viewer is None:
            from gui.frame_viewer.frame_viewer_window import FrameViewerWindow
            self._frame_viewer = FrameViewerWindow()
            self._frame_viewer.destroyed.connect(
                lambda: setattr(self, "_frame_viewer", None)
            )
        self._frame_viewer.update_graph(self._project)
        if self._last_run_result is not None:
            self._frame_viewer.set_result(self._last_run_result)
        self._frame_viewer.show()
        self._frame_viewer.raise_()

    def closeEvent(self, event) -> None:
        if self._confirm_discard_changes():
            self._save_settings()
            if self._frame_viewer is not None:
                self._frame_viewer.close()
            event.accept()
        else:
            event.ignore()
