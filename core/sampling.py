"""
Pure distribution-sampling primitives. No project dependencies — numpy only.

Called exclusively by ToleranceSpec.sample() in core/tolerance.py.
All distribution-specific branching lives here so tolerance.py stays distribution-agnostic.
"""

import numpy as np


def sample_uniform(
    bound: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_trials samples from Uniform(-bound, +bound).

    Parameters
    ----------
    bound : float
        Half-width of the uniform distribution (>= 0). bound=0 returns zeros.
    n_trials : int
    rng : np.random.Generator
    """
    if bound == 0.0:
        return np.zeros(n_trials)
    return rng.uniform(-bound, bound, size=n_trials)


def sample_normal(
    bound: float,
    sigma_level: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_trials samples from Normal(0, bound/sigma_level).

    Parameters
    ----------
    bound : float
        The stated tolerance value (e.g., ±0.01 mm). Interpreted as sigma_level*sigma.
    sigma_level : float
        Number of standard deviations that bound represents (default in callers: 3.0).
    n_trials : int
    rng : np.random.Generator
    """
    if bound == 0.0:
        return np.zeros(n_trials)
    sigma = bound / sigma_level
    return rng.normal(0.0, sigma, size=n_trials)


def sample(
    distribution: str,
    bound: float,
    sigma_level: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the correct sampler based on distribution name.

    Parameters
    ----------
    distribution : str
        "uniform" or "normal".
    bound, sigma_level, n_trials, rng
        Forwarded to the appropriate sampler.
    """
    if distribution == "uniform":
        return sample_uniform(bound, n_trials, rng)
    elif distribution == "normal":
        return sample_normal(bound, sigma_level, n_trials, rng)
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Supported values: 'uniform', 'normal'."
        )
