"""
tests/test_serializer.py — Tests for io/serializer.py.

Tests file I/O (save_project, load_project) and error handling (ProjectLoadError).
All tests that write to disk use the pytest tmp_path fixture for automatic cleanup.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from io.schema import (
    ProjectModel,
    SavedAnalysisModel,
    SimSettingsModel,
    frame_graph_to_project_model,
    project_model_to_frame_graph,
)
from io.serializer import ProjectLoadError, load_project, save_project


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_sim_settings() -> SimSettingsModel:
    return SimSettingsModel(mode="fk_verification", n_trials=500, seed=7)


def _make_tol6(bound: float = 0.005) -> ToleranceSpec6:
    spec = ToleranceSpec(distribution="uniform", bound=bound)
    return ToleranceSpec6(dx=spec, dy=spec, dz=spec, rx=spec, ry=spec, rz=spec)


def _make_project() -> ProjectModel:
    """Minimal valid ProjectModel for serializer round-trip tests."""
    fg = FrameGraph()
    for name in ["world", "tool"]:
        fg.add_frame(name)
    fg.add_edge(
        "world", "tool",
        HTM.from_xyz_euler([0, 0, 150], [0, 0, 0]),
        _make_tol6(),
        name="world_to_tool",
    )
    return frame_graph_to_project_model(fg, _make_sim_settings())


# ── TestSaveLoadProject ───────────────────────────────────────────────────────

class TestSaveLoadProject:
    def test_round_trip_model_dump_equal(self, tmp_path):
        """save + load produces an equivalent model (compared via model_dump)."""
        project = _make_project()
        path = str(tmp_path / "project.json")
        save_project(project, path)
        loaded = load_project(path)
        assert loaded.model_dump() == project.model_dump()

    def test_round_trip_frame_names(self, tmp_path):
        """Frame names are identical after save/load."""
        project = _make_project()
        path = str(tmp_path / "project.json")
        save_project(project, path)
        loaded = load_project(path)
        assert {f.name for f in loaded.frames} == {f.name for f in project.frames}

    def test_round_trip_edge_connectivity(self, tmp_path):
        """Edge parent/child/name are identical after save/load."""
        project = _make_project()
        path = str(tmp_path / "project.json")
        save_project(project, path)
        loaded = load_project(path)
        orig = {e.name: (e.parent, e.child) for e in project.edges}
        rt = {e.name: (e.parent, e.child) for e in loaded.edges}
        assert orig == rt

    def test_saved_file_is_valid_json(self, tmp_path):
        """The saved file parses with json.loads without error."""
        project = _make_project()
        path = str(tmp_path / "project.json")
        save_project(project, path)
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.loads(f.read())
        assert parsed["schema_version"] == 1

    def test_saved_file_has_indent(self, tmp_path):
        """The file uses indented formatting (human-readable, diff-friendly)."""
        project = _make_project()
        path = str(tmp_path / "project.json")
        save_project(project, path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Indented JSON always has lines starting with spaces
        assert any(line.startswith("  ") for line in content.splitlines())

    def test_round_trip_with_saved_analysis(self, tmp_path):
        """SavedAnalysisModel entries survive save/load."""
        fg = FrameGraph()
        for name in ["world", "flange"]:
            fg.add_frame(name)
        fg.add_edge("world", "flange", HTM.from_xyz_euler([0, 0, 100], [0, 0, 0]),
                    _make_tol6(), name="e0")
        analyses = [SavedAnalysisModel(name="wrist_error", frame_a="world", frame_b="flange")]
        project = frame_graph_to_project_model(fg, _make_sim_settings(),
                                               saved_analyses=analyses)
        path = str(tmp_path / "project.json")
        save_project(project, path)
        loaded = load_project(path)
        assert len(loaded.saved_analyses) == 1
        assert loaded.saved_analyses[0].name == "wrist_error"

    def test_round_trip_htm_matrix_close(self, tmp_path):
        """Nominal HTM 4×4 matrix survives save/load within 1e-12."""
        fg = FrameGraph()
        for name in ["world", "tool"]:
            fg.add_frame(name)
        M = np.eye(4)
        M[:3, 3] = [10.5, -3.2, 87.9]
        fg.add_edge("world", "tool", HTM.from_matrix(M), _make_tol6(), name="e0")
        project = frame_graph_to_project_model(fg, _make_sim_settings())
        path = str(tmp_path / "project.json")
        save_project(project, path)
        loaded_project = load_project(path)
        fg2 = project_model_to_frame_graph(loaded_project)
        for e in fg2.all_edges():
            np.testing.assert_allclose(e.nominal.matrix, M, atol=1e-12)

    def test_round_trip_tolerance_values(self, tmp_path):
        """Tolerance bound and sigma_level survive save/load without loss."""
        spec = ToleranceSpec(distribution="normal", bound=0.00314, sigma_level=2.5)
        tol6 = ToleranceSpec6(dx=spec, dy=spec, dz=spec, rx=spec, ry=spec, rz=spec)
        fg = FrameGraph()
        for name in ["a", "b"]:
            fg.add_frame(name)
        fg.add_edge("a", "b", HTM.from_xyz_euler([0, 0, 0], [0, 0, 0]), tol6, name="e0")
        project = frame_graph_to_project_model(fg, _make_sim_settings())
        path = str(tmp_path / "project.json")
        save_project(project, path)
        loaded_project = load_project(path)
        fg2 = project_model_to_frame_graph(loaded_project)
        e = list(fg2.all_edges())[0]
        assert e.tolerance.dx.bound == pytest.approx(0.00314)
        assert e.tolerance.dx.sigma_level == pytest.approx(2.5)
        assert e.tolerance.dx.distribution == "normal"


# ── Error handling tests ──────────────────────────────────────────────────────

class TestLoadProjectErrorHandling:
    def test_file_not_found_raises_project_load_error(self, tmp_path):
        """Missing file raises ProjectLoadError (not FileNotFoundError directly)."""
        path = str(tmp_path / "does_not_exist.json")
        with pytest.raises(ProjectLoadError, match="not found"):
            load_project(path)

    def test_file_not_found_error_wraps_original(self, tmp_path):
        """The ProjectLoadError wraps the original FileNotFoundError as __cause__."""
        path = str(tmp_path / "ghost.json")
        with pytest.raises(ProjectLoadError) as exc_info:
            load_project(path)
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    def test_malformed_json_raises_project_load_error(self, tmp_path):
        """A file with invalid JSON raises ProjectLoadError."""
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json {{{{")
        with pytest.raises(ProjectLoadError):
            load_project(path)

    def test_empty_file_raises_project_load_error(self, tmp_path):
        """An empty file raises ProjectLoadError."""
        path = str(tmp_path / "empty.json")
        with open(path, "w") as f:
            f.write("")
        with pytest.raises(ProjectLoadError):
            load_project(path)

    def test_schema_version_mismatch_raises_project_load_error(self, tmp_path):
        """A file with schema_version != 1 raises ProjectLoadError."""
        project = _make_project()
        path = str(tmp_path / "v999.json")
        # Write valid JSON with wrong schema version
        data = project.model_dump()
        data["schema_version"] = 999
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
        with pytest.raises(ProjectLoadError, match="schema_version"):
            load_project(path)

    def test_schema_version_mismatch_message_contains_version_numbers(self, tmp_path):
        """Error message includes both the found and expected version numbers."""
        project = _make_project()
        path = str(tmp_path / "v5.json")
        data = project.model_dump()
        data["schema_version"] = 5
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
        with pytest.raises(ProjectLoadError, match="5"):
            load_project(path)

    def test_dangling_edge_reference_raises_project_load_error(self, tmp_path):
        """A JSON file with a dangling edge.parent raises ProjectLoadError."""
        project = _make_project()
        data = project.model_dump()
        data["edges"][0]["parent"] = "nonexistent_frame"
        path = str(tmp_path / "dangling.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
        with pytest.raises(ProjectLoadError, match="nonexistent_frame"):
            load_project(path)

    def test_missing_required_field_raises_project_load_error(self, tmp_path):
        """A JSON file missing required fields (sim_settings) raises ProjectLoadError."""
        path = str(tmp_path / "incomplete.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"schema_version": 1, "frames": [], "edges": []}, indent=2))
        with pytest.raises(ProjectLoadError):
            load_project(path)
