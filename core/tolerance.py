"""
Tolerance specifications and the small-angle perturbation pipeline.

Perturbation convention (locked, Section 2.2.2):
    T_perturbed = T_nominal @ T_delta(delta)
    i.e., local-frame, right-multiplication.

IMPORTANT — locked flag and sampling:
    ToleranceSpec.sample() always samples every DoF regardless of locked=True.
    A locked tolerance still represents a real physical error source that contributes
    to FK propagation. The locked flag is consulted ONLY by the allocation engine
    (sim/allocation.py) when selecting free variables for the inverse solve.
    Do NOT add "if self.locked: return zeros" here — that would silently break FK mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core import sampling as _sampling
from core.transforms import HTM


# ── ToleranceSpec (single DoF) ───────────────────────────────────────────────

@dataclass
class ToleranceSpec:
    """Tolerance specification for one degree of freedom.

    Parameters
    ----------
    distribution : "uniform" or "normal"
    bound : float
        For "uniform": half-width of the distribution (samples in [-bound, +bound]).
        For "normal": the stated tolerance = sigma_level * sigma, so sigma = bound/sigma_level.
    sigma_level : float
        Default 3.0. Only meaningful when distribution="normal".
    locked : bool
        Default False. When True, this DoF is excluded from inverse allocation's
        free-variable set. Has NO effect on sampling — see module docstring.
    """

    distribution: Literal["uniform", "normal"]
    bound: float
    sigma_level: float = 3.0
    locked: bool = False

    def __post_init__(self) -> None:
        if self.distribution not in ("uniform", "normal"):
            raise ValueError(
                f"ToleranceSpec distribution must be 'uniform' or 'normal'; "
                f"got '{self.distribution}'."
            )
        if self.bound < 0:
            raise ValueError(
                f"ToleranceSpec bound must be >= 0; got {self.bound}."
            )

    def sample(self, n_trials: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n_trials samples. Always samples regardless of locked flag."""
        return _sampling.sample(
            self.distribution, self.bound, self.sigma_level, n_trials, rng
        )


# ── ToleranceSpec6 (per-edge, 6-DoF aggregate) ───────────────────────────────

class ToleranceSpec6:
    """Ordered container of exactly 6 ToleranceSpec instances.

    DoF order (fixed): [dx, dy, dz, rx, ry, rz]
    """

    def __init__(
        self,
        dx: ToleranceSpec,
        dy: ToleranceSpec,
        dz: ToleranceSpec,
        rx: ToleranceSpec,
        ry: ToleranceSpec,
        rz: ToleranceSpec,
    ) -> None:
        self._specs: list[ToleranceSpec] = [dx, dy, dz, rx, ry, rz]

    # Named properties for readability in calling code.
    @property
    def dx(self) -> ToleranceSpec: return self._specs[0]
    @property
    def dy(self) -> ToleranceSpec: return self._specs[1]
    @property
    def dz(self) -> ToleranceSpec: return self._specs[2]
    @property
    def rx(self) -> ToleranceSpec: return self._specs[3]
    @property
    def ry(self) -> ToleranceSpec: return self._specs[4]
    @property
    def rz(self) -> ToleranceSpec: return self._specs[5]

    def __getitem__(self, idx: int) -> ToleranceSpec:
        return self._specs[idx]

    def __len__(self) -> int:
        return 6

    def sample(self, n_trials: int, rng: np.random.Generator) -> np.ndarray:
        """Draw samples for all 6 DoFs.

        Returns
        -------
        np.ndarray, shape (n_trials, 6)
            Columns ordered [dx, dy, dz, rx, ry, rz].
        """
        return np.column_stack([spec.sample(n_trials, rng) for spec in self._specs])


# ── Perturbation pipeline ────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """Batched skew-symmetric matrix from a 3-vector or batch of 3-vectors.

    Parameters
    ----------
    v : np.ndarray, shape (..., 3)

    Returns
    -------
    np.ndarray, shape (..., 3, 3)
    """
    v = np.asarray(v, dtype=float)
    *batch, _ = v.shape
    S = np.zeros((*batch, 3, 3))
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] =  v[..., 1]
    S[..., 1, 0] =  v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] =  v[..., 0]
    return S


def small_angle_rotation_matrix_batch(rotvec_batch: np.ndarray) -> np.ndarray:
    """Build re-orthonormalized rotation matrices from small-angle rotation vectors.

    Parameters
    ----------
    rotvec_batch : np.ndarray, shape (N, 3)
        Each row is [rx, ry, rz] in radians (small-angle regime).

    Returns
    -------
    np.ndarray, shape (N, 3, 3)

    Notes
    -----
    First-order approximation R ≈ I + skew(rotvec) is not exactly orthonormal.
    SVD projection (U @ Vt) gives the nearest orthonormal matrix, which is required
    because HTM.inverse() and downstream FK composition assume valid rotation matrices.
    """
    rotvec_batch = np.asarray(rotvec_batch, dtype=float)
    N = rotvec_batch.shape[0]
    R_approx = np.eye(3)[np.newaxis, :, :] + skew(rotvec_batch)  # (N, 3, 3)
    U, _, Vt = np.linalg.svd(R_approx)
    return U @ Vt  # (N, 3, 3)


def delta_to_htm_batch(delta_batch: np.ndarray) -> np.ndarray:
    """Assemble perturbation HTMs from a batch of 6-DoF delta vectors.

    Parameters
    ----------
    delta_batch : np.ndarray, shape (N, 6)
        Columns ordered [dx, dy, dz, rx, ry, rz].

    Returns
    -------
    np.ndarray, shape (N, 4, 4)
    """
    delta_batch = np.asarray(delta_batch, dtype=float)
    N = delta_batch.shape[0]
    T = np.zeros((N, 4, 4))
    T[:, :3, :3] = small_angle_rotation_matrix_batch(delta_batch[:, 3:])
    T[:, :3, 3] = delta_batch[:, :3]
    T[:, 3, 3] = 1.0
    return T


def apply_perturbation_batch(
    nominal: HTM,
    delta_batch: np.ndarray,
) -> np.ndarray:
    """Apply batched perturbations to a nominal HTM.

    Implements the locked convention (Section 2.2.2):
        T_perturbed[i] = nominal.matrix @ T_delta[i]
    (local-frame, right-multiplication)

    Parameters
    ----------
    nominal : HTM
        The nominal transformation for one edge.
    delta_batch : np.ndarray, shape (N, 6)
        Sampled 6-DoF perturbations for N Monte Carlo trials.

    Returns
    -------
    np.ndarray, shape (N, 4, 4)
    """
    delta_htms = delta_to_htm_batch(delta_batch)  # (N, 4, 4)
    # Vectorized: no Python loop over trials.
    return np.einsum("ij,njk->nik", nominal.matrix, delta_htms)
