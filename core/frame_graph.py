"""
Frame graph: topology, validation, path queries, and sensitivity primitives.

Design decisions (locked, Section 2.3):
- Strictly serial/open chains: max one incoming edge per Frame (DAG constraint).
- Disjoint components are never auto-connected; they raise DisjointFramesError.
- Root frames (in-degree 0) have identity pose every trial — no edge to sample.
- adjoint() and compute_sensitivity() live here (not in sim/) so both the Pareto
  sensitivity breakdown and the inverse allocator share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from core.tolerance import ToleranceSpec6, skew
from core.transforms import HTM

if TYPE_CHECKING:
    pass  # avoid circular imports if needed later


# ── Exception ────────────────────────────────────────────────────────────────

class DisjointFramesError(Exception):
    """Raised when a path query spans two disconnected components."""


def _disjoint_message(frame_a: str, frame_b: str) -> str:
    # Locked message text — Section 2.3.1. Do not alter wording.
    return (
        f"Frames '{frame_a}' and '{frame_b}' have no connected path between them.\n"
        "If these components share a common physical reference (e.g., a machine\n"
        "base or optical table), add an explicit joint Frame and define the\n"
        "structural edges connecting both sub-assembly roots to it."
    )


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Frame:
    name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class HTMEdge:
    parent: str
    child: str
    nominal: HTM
    tolerance: ToleranceSpec6
    name: str  # unique within graph; enforced at add_edge() time


# ── FrameGraph ────────────────────────────────────────────────────────────────

class FrameGraph:
    """NetworkX-backed directed graph of Frames connected by HTMEdges.

    Build the graph with add_frame / add_edge, then call validate_dag() before
    passing to any simulation engine. Simulation engines call validate_dag()
    internally as a precondition.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._frames: dict[str, Frame] = {}
        self._edges: dict[str, HTMEdge] = {}  # keyed by edge name

    # ── Build ────────────────────────────────────────────────────────────────

    def add_frame(self, name: str, metadata: dict | None = None) -> Frame:
        """Add a Frame node. Raises ValueError if name already exists."""
        if name in self._frames:
            raise ValueError(f"Frame '{name}' already exists in this graph.")
        frame = Frame(name=name, metadata=metadata or {})
        self._frames[name] = frame
        self._g.add_node(name)
        return frame

    def add_edge(
        self,
        parent: str,
        child: str,
        nominal: HTM,
        tolerance: ToleranceSpec6,
        name: str | None = None,
    ) -> HTMEdge:
        """Add an HTMEdge. Raises ValueError on missing frames or name collision."""
        if parent not in self._frames:
            raise ValueError(f"Parent frame '{parent}' does not exist. Add it first.")
        if child not in self._frames:
            raise ValueError(f"Child frame '{child}' does not exist. Add it first.")
        edge_name = name if name is not None else f"{parent}->{child}"
        if edge_name in self._edges:
            raise ValueError(f"Edge name '{edge_name}' already exists in this graph.")
        edge = HTMEdge(parent=parent, child=child, nominal=nominal,
                       tolerance=tolerance, name=edge_name)
        self._edges[edge_name] = edge
        self._g.add_edge(parent, child, edge=edge)
        return edge

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_dag(self) -> None:
        """Raise ValueError if the graph contains a cycle or multi-parent frame.

        Simulation engines call this internally — callers need not invoke it manually.
        """
        try:
            cycle = nx.find_cycle(self._g)
            nodes = [u for u, _ in cycle] + [cycle[-1][1]]
            path_str = " -> ".join(nodes)
            raise ValueError(f"Cycle detected: {path_str}")
        except nx.NetworkXNoCycle:
            pass

        for node in self._g.nodes:
            in_edges = list(self._g.in_edges(node, data=True))
            if len(in_edges) > 1:
                incoming_names = [d['edge'].name for _, _, d in in_edges]
                raise ValueError(
                    f"Frame '{node}' has {len(in_edges)} incoming edges "
                    f"({', '.join(incoming_names)}). "
                    "Only one incoming edge per Frame is allowed (strictly serial chains)."
                )

    # ── Query — topology ─────────────────────────────────────────────────────

    def root_frames(self) -> list[str]:
        """Frames with no incoming edges. Each root's pose is identity every trial."""
        return [n for n in self._g.nodes if self._g.in_degree(n) == 0]

    def weakly_connected_components(self) -> list[set[str]]:
        """All weakly-connected components as sets of Frame names."""
        return [set(c) for c in nx.weakly_connected_components(self._g)]

    def topological_edge_order(self) -> list[str]:
        """Edge names in an order safe for forward-kinematics composition.

        Guaranteed: a child's edge comes after its parent's edge.
        """
        result: list[str] = []
        for node in nx.topological_sort(self._g):
            for _, child, data in self._g.out_edges(node, data=True):
                result.append(data['edge'].name)
        return result

    def all_frames(self) -> list[Frame]:
        return list(self._frames.values())

    def all_edges(self) -> list[HTMEdge]:
        return list(self._edges.values())

    # ── Query — transforms ───────────────────────────────────────────────────

    def nominal_transform_between(self, frame_a: str, frame_b: str) -> HTM:
        """Compose nominal HTMs along the unique path from frame_a to frame_b.

        Raises DisjointFramesError if the frames are in different components.
        Handles both forward (parent→child) and reverse (child→parent) steps.
        """
        if frame_a == frame_b:
            return HTM.from_matrix(np.eye(4))
        self._assert_same_component(frame_a, frame_b)
        T = HTM.from_matrix(np.eye(4))
        for edge, is_forward in self.path_edges_between(frame_a, frame_b):
            T = T.compose(edge.nominal if is_forward else edge.nominal.inverse())
        return T

    def path_edges_between(
        self, frame_a: str, frame_b: str
    ) -> list[tuple[HTMEdge, bool]]:
        """Return (edge, is_forward) pairs along the undirected path frame_a → frame_b.

        is_forward=True  → edge traversed parent→child (with the DAG direction)
        is_forward=False → edge traversed child→parent (against the DAG direction)

        Raises DisjointFramesError if frames are in different components.
        """
        if frame_a == frame_b:
            return []
        self._assert_same_component(frame_a, frame_b)
        undirected = self._g.to_undirected()
        path_nodes: list[str] = nx.shortest_path(undirected, frame_a, frame_b)
        result: list[tuple[HTMEdge, bool]] = []
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            if self._g.has_edge(u, v):
                result.append((self._g.edges[u, v]['edge'], True))
            else:
                result.append((self._g.edges[v, u]['edge'], False))
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _assert_same_component(self, frame_a: str, frame_b: str) -> None:
        components = self.weakly_connected_components()
        for component in components:
            if frame_a in component and frame_b in component:
                return
        raise DisjointFramesError(_disjoint_message(frame_a, frame_b))


# ── Sensitivity primitives (relocated from sim/allocation.py per Mod 2) ──────

def adjoint(T: HTM) -> np.ndarray:
    """6×6 spatial adjoint of a rigid transform T.

    Uses the [v, ω] ordering that matches our delta convention [dx,dy,dz,rx,ry,rz]:
        Ad_T = [[R,    skew(t)@R],
                [0,    R        ]]
    where R = T.matrix[:3,:3], t = T.matrix[:3,3].

    Maps a 6-vector perturbation [δv, δω] from T's local frame to the world frame,
    consistent with the right-multiply perturbation convention (Section 2.2.2).
    """
    R = T.matrix[:3, :3]
    t = T.matrix[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew(t) @ R  # couples ω input to v output
    return Ad


def compute_sensitivity(
    frame_graph: FrameGraph,
    frame_a: str,
    frame_b: str,
    edge_names: list[str],
) -> np.ndarray:
    """Sensitivity of frame_b's pose to perturbations at each listed edge.

    Parameters
    ----------
    frame_graph : FrameGraph
    frame_a, frame_b : str
        The analysis endpoints. Typically frame_a is the chain root.
    edge_names : list[str]
        Edge names in the order you want them as columns. Usually the result of
        [e.name for e, _ in frame_graph.path_edges_between(frame_a, frame_b)].

    Returns
    -------
    np.ndarray, shape (6, 6 * len(edge_names))
        Columns are grouped in blocks of 6, one block per edge in edge_names order.
        Block i = adjoint(T_{exit_node_i → frame_b}).
    """
    # Build a direction map from the actual path.
    path = frame_graph.path_edges_between(frame_a, frame_b)
    direction_map: dict[str, bool] = {e.name: fwd for e, fwd in path}

    blocks: list[np.ndarray] = []
    for name in edge_names:
        edge = frame_graph._edges[name]
        is_forward = direction_map[name]
        exit_node = edge.child if is_forward else edge.parent
        # Sensitivity = Ad_{T_{frame_a → exit_node}}: the perturbation at exit_node
        # conjugated by the transform from frame_a to that node lands at the world/root frame.
        T_a_to_exit = frame_graph.nominal_transform_between(frame_a, exit_node)
        blocks.append(adjoint(T_a_to_exit))

    return np.hstack(blocks)
