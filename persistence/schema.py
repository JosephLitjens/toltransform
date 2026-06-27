"""
io/schema.py — Pydantic models and Python-object ↔ model conversion functions.

Handles the data-model layer of project persistence: mapping between live Python
objects (FrameGraph, HTM, ToleranceSpec6) and JSON-serializable Pydantic models.
File I/O lives in io/serializer.py; this module has no disk operations.

Public models:
    ToleranceSpecModel, ToleranceSpec6Model
    HTMInputXyzEuler, HTMInputMatrix, HTMInputQuaternion, HTMInputScrew
    HTMInputModel          (discriminated union of the four input variants)
    FrameModel, HTMEdgeModel, SimSettingsModel, SavedAnalysisModel
    ProjectModel           (top-level container with cross-reference validation)

Public conversion functions:
    frame_graph_to_project_model(frame_graph, sim_settings, saved_analyses=None)
    project_model_to_frame_graph(project) -> FrameGraph

Scope (B1-5 only):
    TrialData, ParetoSensitivityReport, and AllocationResult are run outputs and
    are NOT persisted. Only project topology (frames, edges, nominal HTMs, tolerances,
    sim settings, saved point-pair analysis targets) is saved.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
import pydantic
from pydantic import BaseModel, Field, field_validator, model_validator

from core.frame_graph import FrameGraph
from core.tolerance import ToleranceSpec, ToleranceSpec6
from core.transforms import HTM


# ── Tolerance models ──────────────────────────────────────────────────────────

class ToleranceSpecModel(BaseModel):
    """JSON-serializable form of core.tolerance.ToleranceSpec."""
    distribution: Literal["uniform", "normal"]
    bound: float
    sigma_level: float = 3.0
    locked: bool = False

    @field_validator("bound")
    @classmethod
    def bound_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"bound must be >= 0; got {v}")
        return v


class ToleranceSpec6Model(BaseModel):
    """JSON-serializable form of core.tolerance.ToleranceSpec6 (all 6 DoF)."""
    dx: ToleranceSpecModel
    dy: ToleranceSpecModel
    dz: ToleranceSpecModel
    rx: ToleranceSpecModel
    ry: ToleranceSpecModel
    rz: ToleranceSpecModel


# ── HTM input-representation models ──────────────────────────────────────────
# Four variants, one per HTM named constructor. Using a discriminated union
# preserves the original input form through save/load — no silent canonicalization
# to matrix form unless the HTM was created without an input_representation.

class HTMInputXyzEuler(BaseModel):
    """HTM constructed via HTM.from_xyz_euler(xyz, euler_angles)."""
    kind: Literal["xyz_euler"]
    xyz: list[float]            # [x, y, z] in mm
    euler_angles: list[float]   # [ez, ey, ex] intrinsic ZYX in radians
    convention: str = "intrinsic_zyx"


class HTMInputMatrix(BaseModel):
    """HTM constructed via HTM.from_matrix(4x4_array) — also the fallback form."""
    kind: Literal["matrix"]
    matrix: list[list[float]]   # 4×4 as nested lists


class HTMInputQuaternion(BaseModel):
    """HTM constructed via HTM.from_quaternion(quat_wxyz, xyz)."""
    kind: Literal["quaternion"]
    quat_wxyz: list[float]  # [w, x, y, z] unit quaternion, scalar first
    xyz: list[float]        # [x, y, z] in mm


class HTMInputScrew(BaseModel):
    """HTM constructed via HTM.from_screw(axis, angle, translation_along_axis, point_on_axis)."""
    kind: Literal["screw"]
    axis: list[float]                        # [x, y, z] unit direction vector
    angle: float                             # rotation in radians
    translation_along_axis: float            # translation along the axis
    point_on_axis: list[float] | None = None # None → axis passes through origin


# Discriminated union — Pydantic uses the "kind" field to select the right model.
HTMInputModel = Annotated[
    Union[HTMInputXyzEuler, HTMInputMatrix, HTMInputQuaternion, HTMInputScrew],
    Field(discriminator="kind"),
]


# ── Frame and edge models ─────────────────────────────────────────────────────

class FrameModel(BaseModel):
    """JSON-serializable form of core.frame_graph.Frame."""
    name: str
    metadata: dict = Field(default_factory=dict)


class HTMEdgeModel(BaseModel):
    """JSON-serializable form of core.frame_graph.HTMEdge."""
    name: str
    parent: str
    child: str
    nominal: HTMInputModel
    tolerance: ToleranceSpec6Model


# ── Simulation settings model ─────────────────────────────────────────────────

class SimSettingsModel(BaseModel):
    """Simulation parameters stored alongside the graph topology."""
    mode: Literal["fk_verification", "ik_allocation"]
    n_trials: int
    seed: int
    default_distribution: Literal["uniform", "normal"] = "uniform"
    default_sigma_level: float = 3.0


# ── Saved analysis model ──────────────────────────────────────────────────────

class SavedAnalysisModel(BaseModel):
    """A named point-pair analysis target stored in the project."""
    name: str
    frame_a: str
    frame_b: str


# ── Top-level project model ───────────────────────────────────────────────────

class ProjectModel(BaseModel):
    """Top-level project container. schema_version=1 for B1-5 and beyond.

    Validates cross-references after field-level parsing: every edge.parent/child
    and every saved_analysis.frame_a/frame_b must name a declared frame.
    """
    schema_version: int = 1
    frames: list[FrameModel] = Field(default_factory=list)
    edges: list[HTMEdgeModel] = Field(default_factory=list)
    sim_settings: SimSettingsModel
    saved_analyses: list[SavedAnalysisModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "ProjectModel":
        """Confirm all edge and analysis frame references name declared frames."""
        frame_names = {f.name for f in self.frames}
        for edge in self.edges:
            if edge.parent not in frame_names:
                raise ValueError(
                    f"Edge '{edge.name}' references parent '{edge.parent}' "
                    f"which is not a declared frame. "
                    f"Declared frames: {sorted(frame_names)}"
                )
            if edge.child not in frame_names:
                raise ValueError(
                    f"Edge '{edge.name}' references child '{edge.child}' "
                    f"which is not a declared frame. "
                    f"Declared frames: {sorted(frame_names)}"
                )
        for analysis in self.saved_analyses:
            for attr in ("frame_a", "frame_b"):
                ref = getattr(analysis, attr)
                if ref not in frame_names:
                    raise ValueError(
                        f"SavedAnalysis '{analysis.name}' references {attr}='{ref}' "
                        "which is not a declared frame. "
                        f"Declared frames: {sorted(frame_names)}"
                    )
        return self


# ── Private conversion helpers ────────────────────────────────────────────────

def _tol_to_model(spec: ToleranceSpec) -> ToleranceSpecModel:
    return ToleranceSpecModel(
        distribution=spec.distribution,
        bound=spec.bound,
        sigma_level=spec.sigma_level,
        locked=spec.locked,
    )


def _tol6_to_model(tol6: ToleranceSpec6) -> ToleranceSpec6Model:
    return ToleranceSpec6Model(
        dx=_tol_to_model(tol6.dx),
        dy=_tol_to_model(tol6.dy),
        dz=_tol_to_model(tol6.dz),
        rx=_tol_to_model(tol6.rx),
        ry=_tol_to_model(tol6.ry),
        rz=_tol_to_model(tol6.rz),
    )


def _model_to_tol(model: ToleranceSpecModel) -> ToleranceSpec:
    return ToleranceSpec(
        distribution=model.distribution,
        bound=model.bound,
        sigma_level=model.sigma_level,
        locked=model.locked,
    )


def _model_to_tol6(model: ToleranceSpec6Model) -> ToleranceSpec6:
    return ToleranceSpec6(
        dx=_model_to_tol(model.dx),
        dy=_model_to_tol(model.dy),
        dz=_model_to_tol(model.dz),
        rx=_model_to_tol(model.rx),
        ry=_model_to_tol(model.ry),
        rz=_model_to_tol(model.rz),
    )


def _htm_to_model(htm: HTM) -> HTMInputModel:
    """Convert a live HTM to its Pydantic input-representation model.

    Preserves the original construction form (xyz_euler, matrix, quaternion, screw)
    by reading htm.input_representation. Falls back to matrix form if the
    input_representation is None (e.g., HTMs built by composition).
    """
    rep = htm.input_representation
    if rep is None:
        return HTMInputMatrix(kind="matrix", matrix=htm.matrix.tolist())

    kind = rep["kind"]
    raw = rep["raw_params"]

    if kind == "xyz_euler":
        return HTMInputXyzEuler(
            kind="xyz_euler",
            xyz=raw["xyz"].tolist(),
            euler_angles=raw["euler_angles"].tolist(),
            convention=raw.get("convention", "intrinsic_zyx"),
        )
    elif kind == "matrix":
        return HTMInputMatrix(kind="matrix", matrix=raw["matrix"].tolist())
    elif kind == "quaternion":
        return HTMInputQuaternion(
            kind="quaternion",
            quat_wxyz=raw["quat_wxyz"].tolist(),
            xyz=raw["xyz"].tolist(),
        )
    elif kind == "screw":
        poa = raw["point_on_axis"]
        return HTMInputScrew(
            kind="screw",
            axis=raw["axis"].tolist(),
            angle=float(raw["angle"]),
            translation_along_axis=float(raw["translation_along_axis"]),
            point_on_axis=None if poa is None else poa.tolist(),
        )
    else:
        return HTMInputMatrix(kind="matrix", matrix=htm.matrix.tolist())


def _model_to_htm(model: HTMInputModel) -> HTM:
    """Reconstruct a live HTM from its Pydantic input-representation model."""
    if model.kind == "xyz_euler":
        return HTM.from_xyz_euler(model.xyz, model.euler_angles, model.convention)
    elif model.kind == "matrix":
        return HTM.from_matrix(np.array(model.matrix))
    elif model.kind == "quaternion":
        return HTM.from_quaternion(model.quat_wxyz, model.xyz)
    else:  # screw
        poa = None if model.point_on_axis is None else np.array(model.point_on_axis)
        return HTM.from_screw(
            model.axis, model.angle, model.translation_along_axis, poa
        )


# ── Public conversion functions ───────────────────────────────────────────────

def frame_graph_to_project_model(
    frame_graph: FrameGraph,
    sim_settings: SimSettingsModel,
    saved_analyses: list[SavedAnalysisModel] | None = None,
) -> ProjectModel:
    """Convert a live FrameGraph to a JSON-serializable ProjectModel.

    Parameters
    ----------
    frame_graph    : FrameGraph — the graph to serialize
    sim_settings   : SimSettingsModel — simulation parameters to embed
    saved_analyses : optional list of SavedAnalysisModel (default: empty)

    Returns
    -------
    ProjectModel — ready for model_dump_json()
    """
    if saved_analyses is None:
        saved_analyses = []

    frame_models = [
        FrameModel(name=f.name, metadata=f.metadata or {})
        for f in frame_graph.all_frames()
    ]

    edge_models = [
        HTMEdgeModel(
            name=edge.name,
            parent=edge.parent,
            child=edge.child,
            nominal=_htm_to_model(edge.nominal),
            tolerance=_tol6_to_model(edge.tolerance),
        )
        for edge in frame_graph.all_edges()
    ]

    return ProjectModel(
        schema_version=1,
        frames=frame_models,
        edges=edge_models,
        sim_settings=sim_settings,
        saved_analyses=saved_analyses,
    )


def project_model_to_frame_graph(project: ProjectModel) -> FrameGraph:
    """Reconstruct a live FrameGraph from a validated ProjectModel.

    Does NOT call validate_dag() — that is the responsibility of the simulation
    engine that consumes the graph. This keeps the loader permissive and lets
    callers validate on their own schedule.

    Parameters
    ----------
    project : ProjectModel — must already be validated (e.g., via load_project)

    Returns
    -------
    FrameGraph
    """
    fg = FrameGraph()
    for f in project.frames:
        fg.add_frame(f.name, metadata=f.metadata or None)
    for e in project.edges:
        htm = _model_to_htm(e.nominal)
        tol6 = _model_to_tol6(e.tolerance)
        fg.add_edge(e.parent, e.child, htm, tol6, name=e.name)
    return fg
