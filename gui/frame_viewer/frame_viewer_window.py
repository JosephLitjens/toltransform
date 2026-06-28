"""
gui/frame_viewer/frame_viewer_window.py — Standalone 3D frame / point-cloud viewer (D-1).

Opens via View → 3D Frame Viewer (Ctrl+3).  Shows the nominal coordinate frames as
RGB triads in real-time as the graph is edited.  After simulation, a mode toggle
switches to an MC scatter of any chosen frame's origin positions.

Rendering: pyqtgraph.opengl.GLViewWidget (OpenGL via Qt).
Section 5.3 compliance: _compute_world_transforms() uses only persistence.schema types
and numpy — no FrameGraph or core.* objects are constructed during editing.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLViewWidget

from persistence.schema import (
    HTMInputModel,
    HTMInputMatrix,
    HTMInputScrew,
    HTMInputXyzEuler,
    HTMInputQuaternion,
    ProjectModel,
)
from sim.monte_carlo_fk import TrialData

_AXIS_COLORS = [
    (1.0, 0.0, 0.0, 1.0),   # X — red
    (0.0, 0.8, 0.0, 1.0),   # Y — green
    (0.0, 0.4, 1.0, 1.0),   # Z — blue
]
_EDGE_COLOR   = (0.55, 0.55, 0.55, 0.7)
_DOT_COLOR    = (1.0, 1.0, 0.0, 1.0)
_AXIS_LEN_DEFAULT = 0.05   # metres; auto-scaled by _pick_axis_len()


# ── Schema → 4×4 matrix (no core.* imports) ───────────────────────────────────

def _nominal_to_matrix(model: HTMInputModel) -> np.ndarray:
    """Convert any HTMInputModel to a 4×4 float64 numpy array.

    Uses scipy.spatial.transform.Rotation — no core.transforms.HTM is created,
    so this is safe to call during editing (Section 5.3).
    """
    T = np.eye(4, dtype=float)
    if isinstance(model, HTMInputXyzEuler):
        ez, ey, ex = model.euler_angles
        T[:3, :3] = SciRotation.from_euler("ZYX", [ez, ey, ex]).as_matrix()
        T[:3, 3] = model.xyz
    elif isinstance(model, HTMInputMatrix):
        T = np.array(model.matrix, dtype=float)
    elif isinstance(model, HTMInputQuaternion):
        w, x, y, z = model.quat_wxyz  # schema stores [w, x, y, z]
        T[:3, :3] = SciRotation.from_quat([x, y, z, w]).as_matrix()  # scipy: [x,y,z,w]
        T[:3, 3] = model.xyz
    elif isinstance(model, HTMInputScrew):
        axis = np.array(model.screw_axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm > 1e-10:
            axis = axis / norm
        T[:3, :3] = SciRotation.from_rotvec(axis * model.angle).as_matrix()
        poa = np.array(model.point_on_axis) if model.point_on_axis is not None else np.zeros(3)
        # Translation: d*axis + (poa - R@poa)  (standard screw-axis formula)
        R = T[:3, :3]
        T[:3, 3] = model.translation_along_axis * axis + poa - R @ poa
    return T


def _compute_world_transforms(project: ProjectModel) -> dict[str, np.ndarray]:
    """Compute nominal world transforms for every frame using schema types + numpy only.

    Traverses the project DAG in topological order (Kahn's algorithm), chaining
    4×4 matrices.  Disconnected components each start from their own root (identity).
    Returns {frame_name: np.ndarray (4,4)}.
    """
    if not project.frames:
        return {}

    # Build parent map and in-degree count
    parent_of: dict[str, str] = {}    # child → parent
    edge_mat: dict[str, np.ndarray] = {}  # child → 4×4 edge matrix
    in_degree: dict[str, int] = {f.name: 0 for f in project.frames}

    for edge in project.edges:
        parent_of[edge.child] = edge.parent
        edge_mat[edge.child] = _nominal_to_matrix(edge.nominal)
        in_degree[edge.child] = in_degree.get(edge.child, 0) + 1

    # Kahn's topological sort
    queue = [name for name, deg in in_degree.items() if deg == 0]
    topo: list[str] = []
    remaining = dict(in_degree)

    children_of: dict[str, list[str]] = {f.name: [] for f in project.frames}
    for edge in project.edges:
        children_of[edge.parent].append(edge.child)

    while queue:
        node = queue.pop(0)
        topo.append(node)
        for child in children_of.get(node, []):
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)

    # Any frames not in topo (cycle / isolated) get identity
    visited = set(topo)
    for f in project.frames:
        if f.name not in visited:
            topo.append(f.name)

    # Chain transforms in topo order
    world: dict[str, np.ndarray] = {}
    for name in topo:
        if name in parent_of:
            world[name] = world[parent_of[name]] @ edge_mat[name]
        else:
            world[name] = np.eye(4, dtype=float)

    return world


def _pick_axis_len(world_transforms: dict[str, np.ndarray]) -> float:
    """Auto-scale axis length to ~15% of the median inter-frame distance."""
    origins = np.array([T[:3, 3] for T in world_transforms.values()])
    if len(origins) < 2:
        return _AXIS_LEN_DEFAULT
    dists = [np.linalg.norm(origins[i] - origins[j])
             for i in range(len(origins)) for j in range(i + 1, len(origins))]
    median_d = np.median(dists)
    return max(median_d * 0.15, 0.005)


# ── Main window ────────────────────────────────────────────────────────────────

class FrameViewerWindow(QWidget):
    """Standalone 3D viewer window.  Two display modes:
      - Frames mode:  nominal coordinate triads + edge connection lines
      - Cloud mode:   MC point cloud scatter for a chosen output frame
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("TolTransform — 3D Frame Viewer")
        self.resize(900, 700)
        self._project: ProjectModel | None = None
        self._trial_data: TrialData | None = None
        self._scene_items: list = []
        self._mode = "frames"
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_graph(self, project: ProjectModel) -> None:
        """Recompute and redraw nominal frame triads. Called on every graph change."""
        self._project = project
        self._repopulate_cloud_combo()
        if self._mode == "frames":
            self._draw_frames()

    def set_result(self, trial_data: TrialData) -> None:
        if not isinstance(trial_data, TrialData):
            return
        self._trial_data = trial_data
        self._repopulate_cloud_combo()
        self._toggle_btn.setEnabled(True)
        if self._mode == "cloud":
            self._draw_cloud()

    def clear(self) -> None:
        """Remove all scene items and forget trial data."""
        self._trial_data = None
        self._clear_scene()
        self._mode = "frames"
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setText("Show Point Cloud")
        self._cloud_combo.setVisible(False)
        self._cloud_label.setVisible(False)
        self._status_label.setText("")

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        # Control bar
        ctrl = QHBoxLayout()
        self._toggle_btn = QPushButton("Show Point Cloud")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setFixedWidth(160)
        self._toggle_btn.clicked.connect(self._on_toggle_mode)
        ctrl.addWidget(self._toggle_btn)

        self._cloud_label = QLabel("Frame:")
        self._cloud_label.setVisible(False)
        ctrl.addWidget(self._cloud_label)

        self._cloud_combo = QComboBox()
        self._cloud_combo.setMinimumWidth(120)
        self._cloud_combo.setVisible(False)
        self._cloud_combo.currentTextChanged.connect(self._on_cloud_frame_changed)
        ctrl.addWidget(self._cloud_combo)

        ctrl.addStretch()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        ctrl.addWidget(self._status_label)
        main.addLayout(ctrl)

        # 3D viewport
        self._view = GLViewWidget()
        self._view.setCameraPosition(distance=1.5)
        self._view.setBackgroundColor((30, 30, 30))
        main.addWidget(self._view, stretch=1)

        # World-axes reference (fixed)
        axis_item = gl.GLAxisItem()
        axis_item.setSize(x=0.1, y=0.1, z=0.1)
        self._view.addItem(axis_item)

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _clear_scene(self) -> None:
        for item in self._scene_items:
            self._view.removeItem(item)
        self._scene_items.clear()

    def _draw_frames(self) -> None:
        self._clear_scene()
        if self._project is None:
            self._status_label.setText("No project loaded.")
            return

        world = _compute_world_transforms(self._project)
        if not world:
            self._status_label.setText("No frames to display.")
            return

        axis_len = _pick_axis_len(world)

        # Coordinate triads
        for T in world.values():
            origin = T[:3, 3]
            for ax_idx, color in enumerate(_AXIS_COLORS):
                tip = origin + T[:3, ax_idx] * axis_len
                pts = np.array([origin, tip])
                line = gl.GLLinePlotItem(pos=pts, color=color, width=2.0, antialias=True)
                self._view.addItem(line)
                self._scene_items.append(line)

        # Edge connection lines (grey)
        for edge in self._project.edges:
            if edge.parent in world and edge.child in world:
                pts = np.array([world[edge.parent][:3, 3], world[edge.child][:3, 3]])
                line = gl.GLLinePlotItem(pos=pts, color=_EDGE_COLOR, width=1.2,
                                         antialias=True)
                self._view.addItem(line)
                self._scene_items.append(line)

        # Frame origin dots
        origins = np.array([T[:3, 3] for T in world.values()])
        if len(origins):
            dots = gl.GLScatterPlotItem(pos=origins, color=_DOT_COLOR, size=6)
            self._view.addItem(dots)
            self._scene_items.append(dots)

        n = len(world)
        self._status_label.setText(
            f"{n} frame{'s' if n != 1 else ''} — axis length {axis_len*1000:.1f} mm"
        )

    def _draw_cloud(self) -> None:
        self._clear_scene()
        if self._trial_data is None:
            self._status_label.setText("Run simulation first.")
            return

        frame = self._cloud_combo.currentText()
        if not frame or frame not in self._trial_data.frame_poses:
            self._status_label.setText("Select a frame above.")
            return

        poses = self._trial_data.frame_poses[frame]   # (N, 4, 4)
        xyz = poses[:, :3, 3].astype(float)            # (N, 3)
        N = len(xyz)

        # Color by z depth (viridis-like: blue→green→yellow)
        z = xyz[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            t = (z - z_min) / (z_max - z_min)
        else:
            t = np.zeros(N)
        r = np.clip(t * 2 - 1, 0, 1)
        g = np.clip(1 - np.abs(t * 2 - 1), 0, 1)
        b = np.clip(1 - t * 2, 0, 1)
        a = np.ones(N)
        colors = np.column_stack([r, g, b, a]).astype(float)

        scatter = gl.GLScatterPlotItem(pos=xyz, color=colors, size=2.0)
        self._view.addItem(scatter)
        self._scene_items.append(scatter)

        # Also draw the nominal frame origin as a cross
        if frame in self._trial_data.nominal_poses:
            nom = self._trial_data.nominal_poses[frame]
            nom_origin = np.array([nom[:3, 3]])
            marker = gl.GLScatterPlotItem(pos=nom_origin,
                                           color=(1.0, 1.0, 1.0, 1.0), size=10)
            self._view.addItem(marker)
            self._scene_items.append(marker)

        self._status_label.setText(f"{N:,} trials — frame: {frame}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _repopulate_cloud_combo(self) -> None:
        frame_names = (
            list(self._trial_data.frame_poses.keys())
            if self._trial_data is not None
            else (
                [f.name for f in self._project.frames]
                if self._project is not None else []
            )
        )
        current = self._cloud_combo.currentText()
        self._cloud_combo.blockSignals(True)
        self._cloud_combo.clear()
        self._cloud_combo.addItems(frame_names)
        idx = self._cloud_combo.findText(current)
        self._cloud_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._cloud_combo.blockSignals(False)

    def _on_toggle_mode(self) -> None:
        if self._mode == "frames":
            self._mode = "cloud"
            self._toggle_btn.setText("Show Frames")
            self._cloud_combo.setVisible(True)
            self._cloud_label.setVisible(True)
            self._draw_cloud()
        else:
            self._mode = "frames"
            self._toggle_btn.setText("Show Point Cloud")
            self._cloud_combo.setVisible(False)
            self._cloud_label.setVisible(False)
            self._draw_frames()

    def _on_cloud_frame_changed(self, frame: str) -> None:
        if self._mode == "cloud":
            self._draw_cloud()
