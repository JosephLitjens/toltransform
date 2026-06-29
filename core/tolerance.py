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
from core.transforms import HTM, skew


# ── ToleranceSpec (single DoF) ───────────────────────────────────────────────

@dataclass
class ToleranceSpec:
    """Tolerance specification for one degree of freedom.

    Two mutually exclusive modes:

    Symmetric (default):
        Set ``bound`` only. Samples from [-bound, +bound] (uniform) or
        N(0, bound/sigma_level) (normal).

    Asymmetric:
        Set ``lower`` and ``upper`` together (``bound`` is then auto-derived
        as max(|lower|, |upper|) for IK compatibility — do not pass both
        explicitly). Samples from [lower, upper] (uniform) or
        N((lower+upper)/2, (upper-lower)/(2·sigma_level)) (normal).

    Parameters
    ----------
    distribution : "uniform" or "normal"
    bound : float
        Symmetric half-width (>= 0). Used when lower/upper are None.
        Auto-computed as max(|lower|, |upper|) in asymmetric mode so that
        the allocation engine can read a conservative symmetric bound.
    sigma_level : float
        Default 3.0. Only meaningful when distribution="normal".
    locked : bool
        Default False. When True, this DoF is excluded from inverse allocation's
        free-variable set. Has NO effect on sampling — see module docstring.
    lower : float | None
        Lower bound for asymmetric mode. Must be set together with ``upper``.
    upper : float | None
        Upper bound for asymmetric mode. Must satisfy upper > lower.
    """

    distribution: Literal["uniform", "normal"]
    bound: float = 0.0
    sigma_level: float = 3.0
    locked: bool = False
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.distribution not in ("uniform", "normal"):
            raise ValueError(
                f"ToleranceSpec distribution must be 'uniform' or 'normal'; "
                f"got '{self.distribution}'."
            )

        has_lower = self.lower is not None
        has_upper = self.upper is not None

        if has_lower != has_upper:
            raise ValueError(
                "ToleranceSpec: lower and upper must be set together; "
                "got only one of the two."
            )

        if has_lower and has_upper:
            # Asymmetric mode
            if self.lower >= self.upper:  # type: ignore[operator]
                raise ValueError(
                    f"ToleranceSpec: lower ({self.lower}) must be < upper ({self.upper})."
                )
            # Derive a conservative symmetric bound for IK compatibility.
            self.bound = max(abs(self.lower), abs(self.upper))  # type: ignore[arg-type]
        else:
            # Symmetric mode
            if self.bound < 0:
                raise ValueError(
                    f"ToleranceSpec bound must be >= 0; got {self.bound}."
                )

    @property
    def is_asymmetric(self) -> bool:
        """True when lower/upper are set (asymmetric mode)."""
        return self.lower is not None and self.upper is not None

    @property
    def variance(self) -> float:
        """True second moment about zero for use in sensitivity/Pareto calculations.

        Symmetric uniform:   E[X²] = bound²/3
        Asymmetric uniform:  E[X²] = Var[X] + E[X]²
                             Var[X] = (upper-lower)²/12,  E[X] = (lower+upper)/2
        Symmetric normal:    E[X²] = Var[X] = (bound/sigma_level)²
        Asymmetric normal:   E[X²] = Var[X] + E[X]²
                             Var[X] = ((upper-lower)/(2·sigma_level))²
                             E[X]   = (lower+upper)/2
        """
        if self.is_asymmetric:
            lo, hi = self.lower, self.upper  # type: ignore[assignment]
            mean = (lo + hi) / 2.0
            half_width = (hi - lo) / 2.0
            if self.distribution == "uniform":
                var = (hi - lo) ** 2 / 12.0
            else:
                sigma = half_width / self.sigma_level
                var = sigma ** 2
            return var + mean ** 2
        else:
            if self.distribution == "uniform":
                return self.bound ** 2 / 3.0
            else:
                sigma = self.bound / self.sigma_level
                return sigma ** 2

    def sample(self, n_trials: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n_trials samples. Always samples regardless of locked flag."""
        if self.is_asymmetric:
            return _sampling.sample_asymmetric(
                self.distribution,
                self.lower,  # type: ignore[arg-type]
                self.upper,  # type: ignore[arg-type]
                self.sigma_level,
                n_trials,
                rng,
            )
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
