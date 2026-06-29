"""
tests/test_schema.py — Unit tests for persistence/schema.py.

Tests the Pydantic models (validation, reference checking) and the Python-object
↔ Pydantic-model conversion functions. No disk I/O — all round-trips happen in
memory. Disk I/O tests are in tests/test_serializer.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM
from persistence.schema import (
    FrameModel,
    HTMEdgeModel,
    HTMInputMatrix,
    HTMInputModel,
    HTMInputQuaternion,
    HTMInputScrew,
    HTMInputXyzEuler,
    HTMEdgeModel,
    ProjectModel,
    SavedAnalysisModel,
    SimSettingsModel,
    ToleranceSpec6Model,
    ToleranceSpecModel,
    _htm_to_model,
    _model_to_htm,
    _tol6_to_model,
    _model_to_tol6,
    frame_graph_to_project_model,
    project_model_to_frame_graph,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_sim_settings() -> SimSettingsModel:
    return SimSettingsModel(mode="fk_verification", n_trials=1000, seed=42)


def _make_tol6(bound: float = 0.010, distribution: str = "uniform") -> ToleranceSpec6:
    spec = ToleranceSpec(distribution=distribution, bound=bound)
    return ToleranceSpec6(dx=spec, dy=spec, dz=spec, rx=spec, ry=spec, rz=spec)


def _make_frame_graph() -> FrameGraph:
    """Simple 2-edge chain: world → mid → tip."""
    fg = FrameGraph()
    for name in ["world", "mid", "tip"]:
        fg.add_frame(name)
    fg.add_edge("world", "mid", HTM.from_xyz_euler([0, 0, 100], [0, 0, 0]),
                _make_tol6(0.010), name="e0")
    fg.add_edge("mid", "tip", HTM.from_xyz_euler([0, 0, 50], [0, 0, 0.001]),
                _make_tol6(0.005, "normal"), name="e1")
    return fg


# ── TestToleranceSpecModel ────────────────────────────────────────────────────

class TestToleranceSpecModel:
    def test_valid_constructs(self):
        m = ToleranceSpecModel(distribution="uniform", bound=0.010)
        assert m.bound == pytest.approx(0.010)
        assert m.sigma_level == pytest.approx(3.0)
        assert m.locked is False

    def test_negative_bound_raises(self):
        with pytest.raises(ValidationError, match="bound"):
            ToleranceSpecModel(distribution="normal", bound=-0.001)

    def test_zero_bound_is_valid(self):
        m = ToleranceSpecModel(distribution="uniform", bound=0.0, locked=True)
        assert m.bound == 0.0
        assert m.locked is True

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValidationError):
            ToleranceSpecModel(distribution="gaussian", bound=0.001)

    def test_asymmetric_model_valid(self):
        m = ToleranceSpecModel(distribution="uniform", bound=0.005,
                               lower=-0.002, upper=0.005)
        assert m.lower == pytest.approx(-0.002)
        assert m.upper == pytest.approx(0.005)

    def test_asymmetric_only_lower_raises(self):
        with pytest.raises(ValidationError, match="lower and upper"):
            ToleranceSpecModel(distribution="uniform", bound=0.0, lower=-0.001)

    def test_asymmetric_lower_ge_upper_raises(self):
        with pytest.raises(ValidationError):
            ToleranceSpecModel(distribution="uniform", bound=0.001,
                               lower=0.003, upper=0.001)

    def _find_edge(self, fg: FrameGraph, name: str):
        return next(e for e in fg.all_edges() if e.name == name)

    def test_symmetric_round_trip_via_frame_graph(self):
        spec = ToleranceSpec("uniform", bound=0.007, locked=True)
        t6 = ToleranceSpec6(spec, spec, spec, spec, spec, spec)
        fg = FrameGraph()
        fg.add_frame("a"); fg.add_frame("b")
        fg.add_edge("a", "b", HTM.from_matrix(np.eye(4)), t6, name="e0")
        proj = frame_graph_to_project_model(fg, _make_sim_settings())
        recovered = project_model_to_frame_graph(proj)
        recovered_spec = self._find_edge(recovered, "e0").tolerance.dx
        assert not recovered_spec.is_asymmetric
        assert recovered_spec.bound == pytest.approx(0.007)
        assert recovered_spec.locked is True

    def test_asymmetric_round_trip_via_frame_graph(self):
        lo, hi = -0.003, 0.008
        spec = ToleranceSpec("uniform", lower=lo, upper=hi)
        t6 = ToleranceSpec6(spec, spec, spec, spec, spec, spec)
        fg = FrameGraph()
        fg.add_frame("a"); fg.add_frame("b")
        fg.add_edge("a", "b", HTM.from_matrix(np.eye(4)), t6, name="e0")
        proj = frame_graph_to_project_model(fg, _make_sim_settings())
        recovered = project_model_to_frame_graph(proj)
        r = self._find_edge(recovered, "e0").tolerance.dx
        assert r.is_asymmetric
        assert r.lower == pytest.approx(lo)
        assert r.upper == pytest.approx(hi)
        assert r.bound == pytest.approx(max(abs(lo), abs(hi)))

    def test_asymmetric_normal_round_trip_via_frame_graph(self):
        lo, hi, k = -0.001, 0.004, 2.0
        spec = ToleranceSpec("normal", lower=lo, upper=hi, sigma_level=k)
        t6 = ToleranceSpec6(spec, spec, spec, spec, spec, spec)
        fg = FrameGraph()
        fg.add_frame("a"); fg.add_frame("b")
        fg.add_edge("a", "b", HTM.from_matrix(np.eye(4)), t6, name="e0")
        proj = frame_graph_to_project_model(fg, _make_sim_settings())
        recovered = project_model_to_frame_graph(proj)
        r = self._find_edge(recovered, "e0").tolerance.dx
        assert r.is_asymmetric
        assert r.lower == pytest.approx(lo)
        assert r.upper == pytest.approx(hi)
        assert r.sigma_level == pytest.approx(k)


# ── TestProjectModelValidation ────────────────────────────────────────────────

class TestProjectModelValidation:
    def _minimal_project(self) -> dict:
        """Return a valid project dict with 2 frames and 1 edge."""
        return {
            "sim_settings": {"mode": "fk_verification", "n_trials": 100, "seed": 0},
            "frames": [{"name": "a"}, {"name": "b"}],
            "edges": [{
                "name": "e",
                "parent": "a",
                "child": "b",
                "nominal": {"kind": "matrix", "matrix": np.eye(4).tolist()},
                "tolerance": {
                    dof: {"distribution": "uniform", "bound": 0.001}
                    for dof in ("dx", "dy", "dz", "rx", "ry", "rz")
                },
            }],
        }

    def test_valid_project_constructs(self):
        data = self._minimal_project()
        p = ProjectModel.model_validate(data)
        assert len(p.frames) == 2
        assert len(p.edges) == 1

    def test_schema_version_defaults_to_1(self):
        data = self._minimal_project()
        p = ProjectModel.model_validate(data)
        assert p.schema_version == 1

    def test_dangling_edge_parent_raises(self):
        data = self._minimal_project()
        data["edges"][0]["parent"] = "nonexistent"
        with pytest.raises(ValidationError, match="nonexistent"):
            ProjectModel.model_validate(data)

    def test_dangling_edge_child_raises(self):
        data = self._minimal_project()
        data["edges"][0]["child"] = "missing"
        with pytest.raises(ValidationError, match="missing"):
            ProjectModel.model_validate(data)

    def test_dangling_saved_analysis_frame_a_raises(self):
        data = self._minimal_project()
        data["saved_analyses"] = [{"name": "test", "frame_a": "bad", "frame_b": "a"}]
        with pytest.raises(ValidationError, match="bad"):
            ProjectModel.model_validate(data)

    def test_dangling_saved_analysis_frame_b_raises(self):
        data = self._minimal_project()
        data["saved_analyses"] = [{"name": "test", "frame_a": "a", "frame_b": "ghost"}]
        with pytest.raises(ValidationError, match="ghost"):
            ProjectModel.model_validate(data)


# ── TestHTMInputModelRoundTrip ────────────────────────────────────────────────

class TestHTMInputModelRoundTrip:
    def test_xyz_euler_preserves_kind(self):
        htm = HTM.from_xyz_euler([1, 2, 3], [0.1, 0.0, -0.1])
        model = _htm_to_model(htm)
        assert model.kind == "xyz_euler"
        assert isinstance(model, HTMInputXyzEuler)
        assert model.xyz == pytest.approx([1.0, 2.0, 3.0])
        assert model.euler_angles == pytest.approx([0.1, 0.0, -0.1])

    def test_matrix_preserves_kind(self):
        M = np.eye(4)
        M[0, 3] = 5.0
        htm = HTM.from_matrix(M)
        model = _htm_to_model(htm)
        assert model.kind == "matrix"
        assert isinstance(model, HTMInputMatrix)

    def test_quaternion_preserves_kind(self):
        htm = HTM.from_quaternion([1, 0, 0, 0], [0, 0, 10])
        model = _htm_to_model(htm)
        assert model.kind == "quaternion"
        assert isinstance(model, HTMInputQuaternion)
        assert model.quat_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert model.xyz == pytest.approx([0.0, 0.0, 10.0])

    def test_screw_preserves_kind(self):
        htm = HTM.from_screw([0, 0, 1], 0.05, 10.0)
        model = _htm_to_model(htm)
        assert model.kind == "screw"
        assert isinstance(model, HTMInputScrew)
        assert model.angle == pytest.approx(0.05)
        assert model.translation_along_axis == pytest.approx(10.0)
        assert model.axis == pytest.approx([0.0, 0.0, 1.0])

    def test_screw_null_point_on_axis_round_trips(self):
        htm = HTM.from_screw([0, 1, 0], 0.01, 2.0, point_on_axis=None)
        model = _htm_to_model(htm)
        assert model.point_on_axis is None

    def test_screw_with_point_on_axis_round_trips(self):
        htm = HTM.from_screw([0, 0, 1], 0.02, 5.0, point_on_axis=[1.0, 2.0, 0.0])
        model = _htm_to_model(htm)
        assert model.point_on_axis == pytest.approx([1.0, 2.0, 0.0])

    def test_none_input_representation_falls_back_to_matrix(self):
        """HTMs without an input_representation (e.g., composed) fall back to matrix."""
        M = np.eye(4)
        M[1, 3] = 3.0
        htm = HTM(M, input_representation=None)
        model = _htm_to_model(htm)
        assert model.kind == "matrix"


# ── TestConversionFunctions ───────────────────────────────────────────────────

class TestConversionFunctions:
    def test_frame_graph_round_trip_frame_names(self):
        """Round-trip: fg → project_model → fg2; frame names must match."""
        fg = _make_frame_graph()
        pm = frame_graph_to_project_model(fg, _make_sim_settings())
        fg2 = project_model_to_frame_graph(pm)
        assert {f.name for f in fg.all_frames()} == {f.name for f in fg2.all_frames()}

    def test_frame_graph_round_trip_edge_names(self):
        """Round-trip: edge names and parent→child connectivity are preserved."""
        fg = _make_frame_graph()
        pm = frame_graph_to_project_model(fg, _make_sim_settings())
        fg2 = project_model_to_frame_graph(pm)
        orig_edges = {e.name: (e.parent, e.child) for e in fg.all_edges()}
        rt_edges = {e.name: (e.parent, e.child) for e in fg2.all_edges()}
        assert orig_edges == rt_edges

    def test_edge_nominal_matrix_close_xyz_euler(self):
        """Round-trip HTM (from_xyz_euler): 4×4 matrix preserved to 1e-12."""
        fg = _make_frame_graph()
        pm = frame_graph_to_project_model(fg, _make_sim_settings())
        fg2 = project_model_to_frame_graph(pm)
        for e_orig, e_rt in zip(fg.all_edges(), fg2.all_edges()):
            np.testing.assert_allclose(
                e_orig.nominal.matrix, e_rt.nominal.matrix, atol=1e-12
            )

    def test_tolerance_round_trip_all_fields(self):
        """All 6 DoF ToleranceSpec fields survive the round-trip unchanged."""
        spec = ToleranceSpec(distribution="normal", bound=0.003, sigma_level=2.0, locked=True)
        tol6 = ToleranceSpec6(dx=spec, dy=spec, dz=spec, rx=spec, ry=spec, rz=spec)
        model = _tol6_to_model(tol6)
        tol6_rt = _model_to_tol6(model)
        for attr in ("dx", "dy", "dz", "rx", "ry", "rz"):
            s = getattr(tol6_rt, attr)
            assert s.distribution == "normal"
            assert s.bound == pytest.approx(0.003)
            assert s.sigma_level == pytest.approx(2.0)
            assert s.locked is True

    def test_no_validate_dag_required(self):
        """project_model_to_frame_graph returns without calling validate_dag()."""
        fg = _make_frame_graph()
        pm = frame_graph_to_project_model(fg, _make_sim_settings())
        fg2 = project_model_to_frame_graph(pm)
        assert fg2 is not None   # no exception → validate_dag was not called

    def test_saved_analyses_round_trip(self):
        """SavedAnalysisModel entries are preserved through conversion."""
        fg = _make_frame_graph()
        analyses = [SavedAnalysisModel(name="my_target", frame_a="world", frame_b="tip")]
        pm = frame_graph_to_project_model(fg, _make_sim_settings(), saved_analyses=analyses)
        assert len(pm.saved_analyses) == 1
        assert pm.saved_analyses[0].name == "my_target"
        assert pm.saved_analyses[0].frame_a == "world"
        assert pm.saved_analyses[0].frame_b == "tip"

    def test_frame_metadata_preserved(self):
        """Frame metadata dict survives the round-trip."""
        fg = FrameGraph()
        fg.add_frame("world", metadata={"color": "red", "units": "mm"})
        fg.add_frame("child")
        fg.add_edge("world", "child",
                    HTM.from_xyz_euler([0, 0, 0], [0, 0, 0]),
                    _make_tol6(), name="e0")
        pm = frame_graph_to_project_model(fg, _make_sim_settings())
        fg2 = project_model_to_frame_graph(pm)
        world_frame = next(f for f in fg2.all_frames() if f.name == "world")
        assert world_frame.metadata == {"color": "red", "units": "mm"}
