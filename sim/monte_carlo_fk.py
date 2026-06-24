"""
Forward Monte Carlo FK engine.

RNG scheme (must not change — changes break result reproducibility):
    spawn_key = int.from_bytes(sha256(edge_name.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(SeedSequence([master_seed, spawn_key]))

This guarantees that adding, removing, or renaming *other* edges in the graph
never changes the random draws for a given edge at a fixed master_seed and n_trials.
A single shared sequential RNG consumed in topological order would not have this
property — any graph edit that changes topological ordering would shift all
downstream draws. The per-edge keyed scheme avoids this entirely.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from core.frame_graph import FrameGraph
from core.tolerance import apply_perturbation_batch


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_edge_rng(master_seed: int, edge_name: str) -> np.random.Generator:
    """Return a deterministic per-edge RNG derived from (master_seed, edge_name).

    Parameters
    ----------
    master_seed : int
        The run-level master seed, supplied by the caller to MonteCarloFKEngine.run().
    edge_name : str
        The unique edge name from FrameGraph (enforced unique at add_edge() time).

    Returns
    -------
    np.random.Generator
        A fresh generator whose stream depends only on (master_seed, edge_name).
    """
    spawn_key = int.from_bytes(
        hashlib.sha256(edge_name.encode()).digest()[:8], "little"
    )
    return np.random.default_rng(np.random.SeedSequence([master_seed, spawn_key]))


def _spawn_key(edge_name: str) -> int:
    """Deterministic integer spawn key for an edge name (used in edge_seed_log)."""
    return int.from_bytes(
        hashlib.sha256(edge_name.encode()).digest()[:8], "little"
    )


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class TrialData:
    """Per-trial pose data produced by MonteCarloFKEngine.run().

    Attributes
    ----------
    n_trials : int
    seed : int
        Master seed used for this run.
    frame_poses : dict[str, np.ndarray]
        frame_name -> (N,4,4) float64 absolute pose per trial, in the root frame.
        Root frames are tiled identity for every trial (no incoming edge, no perturbation).
        Relative transform between any two frames in the same component for trial i is
        ``np.linalg.inv(frame_poses[a][i]) @ frame_poses[b][i]`` — no graph re-traversal.
    nominal_poses : dict[str, np.ndarray]
        frame_name -> (4,4) float64 unperturbed reference pose.
    edge_seed_log : dict[str, int]
        edge_name -> spawn_key integer used to construct that edge's RNG.
        Purely for traceability/debugging — allows any edge's sample stream to be
        exactly reproduced offline by calling make_edge_rng(seed, edge_name).
    """

    n_trials: int
    seed: int
    frame_poses: dict[str, np.ndarray] = field(default_factory=dict)
    nominal_poses: dict[str, np.ndarray] = field(default_factory=dict)
    edge_seed_log: dict[str, int] = field(default_factory=dict)


# ── Engine ────────────────────────────────────────────────────────────────────

class MonteCarloFKEngine:
    """Forward tolerance verification engine (Mode 1, Section 3.1).

    Vectorized over trials — no Python loop over individual trial indices.
    """

    @staticmethod
    def run(frame_graph: FrameGraph, n_trials: int, seed: int) -> TrialData:
        """Sample every edge and compose poses for every Frame in every trial.

        Parameters
        ----------
        frame_graph : FrameGraph
            The system topology. validate_dag() is called internally as a precondition.
        n_trials : int
            Number of Monte Carlo trials.
        seed : int
            Master RNG seed. Combined with each edge's name to produce independent
            per-edge sub-streams — reproducible and robust to graph edits.

        Returns
        -------
        TrialData
        """
        # Precondition: raises ValueError on cycles or multi-parent frames.
        frame_graph.validate_dag()

        frame_poses: dict[str, np.ndarray] = {}
        nominal_poses: dict[str, np.ndarray] = {}
        edge_seed_log: dict[str, int] = {}

        # Root frames: anchored at identity for every trial.
        for root in frame_graph.root_frames():
            frame_poses[root] = np.tile(np.eye(4), (n_trials, 1, 1))
            nominal_poses[root] = np.eye(4)

        # Process edges in topological order (child always after parent).
        for edge_name in frame_graph.topological_edge_order():
            edge = frame_graph._edges[edge_name]

            key = _spawn_key(edge_name)
            edge_seed_log[edge_name] = key
            rng = make_edge_rng(seed, edge_name)

            delta_batch = edge.tolerance.sample(n_trials, rng)              # (N,6)
            perturbed = apply_perturbation_batch(edge.nominal, delta_batch) # (N,4,4)

            # Vectorized batched matrix multiply — no loop over range(n_trials).
            frame_poses[edge.child] = np.einsum(
                "nij,njk->nik", frame_poses[edge.parent], perturbed
            )
            nominal_poses[edge.child] = nominal_poses[edge.parent] @ edge.nominal.matrix

        return TrialData(
            n_trials=n_trials,
            seed=seed,
            frame_poses=frame_poses,
            nominal_poses=nominal_poses,
            edge_seed_log=edge_seed_log,
        )
