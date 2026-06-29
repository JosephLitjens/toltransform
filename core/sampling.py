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


def sample_uniform_asymmetric(
    lower: float,
    upper: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_trials samples from Uniform(lower, upper).

    Parameters
    ----------
    lower : float
        Lower bound (may be negative).
    upper : float
        Upper bound; must satisfy upper > lower.
    n_trials : int
    rng : np.random.Generator
    """
    if lower == upper:
        return np.full(n_trials, lower)
    return rng.uniform(lower, upper, size=n_trials)


def sample_normal_asymmetric(
    lower: float,
    upper: float,
    sigma_level: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_trials samples from Normal(mean, sigma) fitted to [lower, upper].

    The interval [lower, upper] is treated as ±sigma_level·sigma about the
    midpoint: mean = (lower + upper) / 2, sigma = (upper - lower) / (2·sigma_level).

    Parameters
    ----------
    lower : float
        Lower bound of the stated tolerance interval.
    upper : float
        Upper bound of the stated tolerance interval; upper > lower required.
    sigma_level : float
        Number of standard deviations that the half-width represents.
    n_trials : int
    rng : np.random.Generator
    """
    mean = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    if half_width == 0.0:
        return np.full(n_trials, mean)
    sigma = half_width / sigma_level
    return rng.normal(mean, sigma, size=n_trials)


def sample(
    distribution: str,
    bound: float,
    sigma_level: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the correct symmetric sampler based on distribution name.

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


def sample_asymmetric(
    distribution: str,
    lower: float,
    upper: float,
    sigma_level: float,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the correct asymmetric sampler based on distribution name."""
    if distribution == "uniform":
        return sample_uniform_asymmetric(lower, upper, n_trials, rng)
    elif distribution == "normal":
        return sample_normal_asymmetric(lower, upper, sigma_level, n_trials, rng)
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Supported values: 'uniform', 'normal'."
        )
