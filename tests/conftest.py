"""
Shared pytest fixtures and numerical-tolerance constants for the TolTransform test suite.

Numerical tolerance convention (lock these; individual tests must not invent their own):
    DEFAULT_ATOL     = 1e-9  -- near-exact / pure floating-point composition checks
    SMALL_ANGLE_ATOL = 1e-6  -- checks expected to carry small-angle residual error
                               (e.g. sin(δθ) vs δθ at ~1 mrad leaves ~1.7e-10 residual,
                               but pytest.approx and np.testing calls still need room)

Fixture catalogue:
    two_edge_chain      -- root → B (x=5 mm nominal) → C (y=10 mm nominal)
    three_edge_chain    -- root → B (Rz=π/4) → C (x=50 mm) → D (zero tol)
    shared_frame_graph  -- root → shared → leaf1 and shared → leaf2 (multi-branch)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM

try:
    from PySide6.QtWidgets import QApplication

    @pytest.fixture(autouse=True)
    def _qt_main_window_test_guard(monkeypatch):
        """Prevent MainWindow QSettings and dirty-dialog hangs in offscreen tests.

        Two root causes interact when consecutive tests each create a MainWindow:
          1. _set_dirty(True) → qtbot closeEvent → _confirm_discard_changes()
             → QMessageBox.question() → hangs in offscreen mode.
          2. _save_settings() writes dock state; _restore_settings() in the next
             test's MainWindow() call reads and tries to apply it, which can
             block when the offscreen display has no real window geometry.

        Fix: patch both methods to no-ops for every test. Monkeypatch auto-reverts.
        """
        try:
            from gui.main_window import MainWindow
            monkeypatch.setattr(MainWindow, '_save_settings', lambda self: None)
            monkeypatch.setattr(MainWindow, '_restore_settings', lambda self: None)
        except ImportError:
            pass
        yield
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
except ImportError:
    pass  # PySide6 not available; skip fixture

# ── Global numerical tolerance constants ──────────────────────────────────────

DEFAULT_ATOL = 1e-9
SMALL_ANGLE_ATOL = 1e-6


# ── Module-level helpers (importable by test files) ───────────────────────────

def make_tol(bound: float) -> ToleranceSpec6:
    """Uniform ±bound on all 6 DoF."""
    s = ToleranceSpec("uniform", bound=bound)
    return ToleranceSpec6(s, s, s, s, s, s)


def make_zero_tol() -> ToleranceSpec6:
    return make_tol(0.0)


def make_htm(x: float = 0.0, y: float = 0.0, z: float = 0.0,
             ez: float = 0.0, ey: float = 0.0, ex: float = 0.0) -> HTM:
    return HTM.from_xyz_euler([x, y, z], [ez, ey, ex])


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def two_edge_chain() -> FrameGraph:
    """root → B (x=5 mm nominal, ±1 mm tol) → C (y=10 mm nominal, ±2 mm tol).

    Representative pure-translation serial chain for envelope and stack-up tests.
    """
    fg = FrameGraph()
    for name in ["root", "B", "C"]:
        fg.add_frame(name)
    fg.add_edge("root", "B", make_htm(x=0.005), make_tol(0.001), name="root->B")
    fg.add_edge("B", "C", make_htm(y=0.010), make_tol(0.002), name="B->C")
    return fg


@pytest.fixture
def three_edge_chain() -> FrameGraph:
    """root → B (Rz=π/4, ±1 mm) → C (x=50 mm, ±1 mm) → D (zero tol).

    Mixes rotation and translation nominals — for lever-arm-in-a-chain tests.
    """
    fg = FrameGraph()
    for name in ["root", "B", "C", "D"]:
        fg.add_frame(name)
    fg.add_edge("root", "B", make_htm(ez=np.pi / 4), make_tol(0.001), name="root->B")
    fg.add_edge("B", "C", make_htm(x=0.050), make_tol(0.001), name="B->C")
    fg.add_edge("C", "D", make_htm(), make_zero_tol(), name="C->D")
    return fg


@pytest.fixture
def shared_frame_graph() -> FrameGraph:
    """root → shared → leaf1 and shared → leaf2 (shared-base multi-branch).

    root→shared: ±5 mm (large shared tolerance).
    shared→leaf1: x=+20 mm offset, ±1 mm.
    shared→leaf2: y=+30 mm offset, ±1 mm.
    """
    fg = FrameGraph()
    for name in ["root", "shared", "leaf1", "leaf2"]:
        fg.add_frame(name)
    fg.add_edge("root", "shared", make_htm(), make_tol(0.005), name="root->shared")
    fg.add_edge("shared", "leaf1", make_htm(x=0.020), make_tol(0.001), name="shared->leaf1")
    fg.add_edge("shared", "leaf2", make_htm(y=0.030), make_tol(0.001), name="shared->leaf2")
    return fg
