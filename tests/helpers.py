"""Shared non-fixture test helpers for the TolTransform test suite.

Import these directly in test files rather than redefining locally:
    from helpers import _FixedToleranceSpec6, _tol6, _uniform_spec
"""

from __future__ import annotations

import numpy as np

from persistence.schema import ToleranceSpec6Model, ToleranceSpecModel


class _FixedToleranceSpec6:
    """Returns the same (N,6) delta every call — for deterministic hand-checks.

    Duck-types ToleranceSpec6.sample() so it can be passed directly as a tolerance
    to FrameGraph.add_edge without monkeypatching.
    """

    def __init__(self, delta_1d):
        self._delta = np.asarray(delta_1d, dtype=float)  # (6,)

    def sample(self, n_trials: int, rng) -> np.ndarray:
        return np.tile(self._delta, (n_trials, 1))  # (N,6)


def _uniform_spec(bound: float = 0.001) -> ToleranceSpecModel:
    return ToleranceSpecModel(distribution="uniform", bound=bound)


def _tol6(bound: float = 0.001) -> ToleranceSpec6Model:
    s = _uniform_spec(bound)
    return ToleranceSpec6Model(dx=s, dy=s, dz=s, rx=s, ry=s, rz=s)
