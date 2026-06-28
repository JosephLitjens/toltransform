"""
tests/test_gui_frame_viewer.py — Tests for gui/frame_viewer/ (D-1).

Tests focus on the _compute_world_transforms() helper (pure numpy, no OpenGL).
GLViewWidget rendering is not tested headlessly (requires an active OpenGL context).

Run headlessly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_frame_viewer.py -v
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from gui.frame_viewer.frame_viewer_window import _compute_world_transforms
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputXyzEuler,
    ProjectModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xyz_euler(x=0.0, y=0.0, z=0.0, ez=0.0, ey=0.0, ex=0.0) -> HTMInputXyzEuler:
    return HTMInputXyzEuler(kind="xyz_euler", xyz=[x, y, z], euler_angles=[ez, ey, ex])


def _locked_tol6() -> ToleranceSpec6Model:
    z = ToleranceSpecModel(distribution="uniform", bound=0.0, locked=True)
    return ToleranceSpec6Model(dx=z, dy=z, dz=z, rx=z, ry=z, rz=z)


def _project(*frame_names, edges=()) -> ProjectModel:
    p = ProjectModel(sim_settings=SimSettingsModel(mode="fk_verification", n_trials=50, seed=0))
    for name in frame_names:
        p.frames.append(FrameModel(name=name))
    for parent, child, nom in edges:
        p.edges.append(HTMEdgeModel(
            name=f"{parent}_{child}", parent=parent, child=child, nominal=nom,
            tolerance=_locked_tol6(),
        ))
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_world_transforms_empty_project():
    p = ProjectModel(sim_settings=SimSettingsModel(mode="fk_verification", n_trials=50, seed=0))
    result = _compute_world_transforms(p)
    assert result == {}


def test_world_transforms_single_frame_no_edges():
    p = _project("root")
    result = _compute_world_transforms(p)
    assert "root" in result
    np.testing.assert_allclose(result["root"], np.eye(4), atol=1e-9)


def test_world_transforms_single_edge_translation():
    p = _project("A", "B", edges=[("A", "B", _xyz_euler(x=1.0, y=2.0, z=3.0))])
    result = _compute_world_transforms(p)
    np.testing.assert_allclose(result["A"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(result["B"][:3, 3], [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(result["B"][:3, :3], np.eye(3), atol=1e-9)


def test_world_transforms_serial_chain_accumulates():
    """3-frame serial chain: B offset from A, C offset from B."""
    p = _project(
        "A", "B", "C",
        edges=[
            ("A", "B", _xyz_euler(x=1.0)),
            ("B", "C", _xyz_euler(x=0.5)),
        ],
    )
    result = _compute_world_transforms(p)
    np.testing.assert_allclose(result["A"][:3, 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result["B"][:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result["C"][:3, 3], [1.5, 0.0, 0.0], atol=1e-9)


def test_world_transforms_rotation_propagates():
    """90° rz rotation on first edge rotates child's coordinate axes."""
    import math
    p = _project("A", "B", edges=[("A", "B", _xyz_euler(ez=math.pi / 2))])
    result = _compute_world_transforms(p)
    # After 90° Z rotation, X-axis of B points in world-Y direction
    np.testing.assert_allclose(result["B"][:3, 0], [0.0, 1.0, 0.0], atol=1e-6)


def test_world_transforms_disconnected_components():
    """Two disconnected components both get sensible transforms."""
    p = _project(
        "A", "B", "C", "D",
        edges=[
            ("A", "B", _xyz_euler(x=1.0)),
            ("C", "D", _xyz_euler(y=2.0)),
        ],
    )
    result = _compute_world_transforms(p)
    # A and C are both roots → identity
    np.testing.assert_allclose(result["A"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(result["C"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(result["B"][:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result["D"][:3, 3], [0.0, 2.0, 0.0], atol=1e-9)
