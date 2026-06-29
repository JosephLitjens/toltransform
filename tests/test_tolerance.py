"""
Tests for core/tolerance.py and core/sampling.py.

All hand-calculable — explicit expected values with explanatory comments.
"""

import numpy as np
import pytest

from core.tolerance import (
    ToleranceSpec,
    ToleranceSpec6,
    apply_perturbation_batch,
    delta_to_htm_batch,
    skew,
    small_angle_rotation_matrix_batch,
)
from core.transforms import HTM


# ── ToleranceSpec construction and validation ────────────────────────────────

class TestToleranceSpecValidation:
    def test_valid_uniform(self):
        spec = ToleranceSpec("uniform", bound=0.01)
        assert spec.distribution == "uniform"
        assert spec.bound == 0.01
        assert spec.sigma_level == 3.0
        assert spec.locked is False

    def test_valid_normal(self):
        spec = ToleranceSpec("normal", bound=0.005, sigma_level=3.0)
        assert spec.distribution == "normal"

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            ToleranceSpec("gaussian", bound=0.01)

    def test_negative_bound_raises(self):
        with pytest.raises(ValueError, match="bound"):
            ToleranceSpec("uniform", bound=-0.001)

    def test_zero_bound_is_valid(self):
        spec = ToleranceSpec("uniform", bound=0.0)
        samples = spec.sample(100, np.random.default_rng(0))
        np.testing.assert_array_equal(samples, np.zeros(100))


# ── ToleranceSpec6 ───────────────────────────────────────────────────────────

class TestToleranceSpec6:
    def _make_spec6(self, bound=0.01, distribution="uniform"):
        s = ToleranceSpec(distribution, bound=bound)
        return ToleranceSpec6(s, s, s, s, s, s)

    def test_named_properties(self):
        dx = ToleranceSpec("uniform", bound=0.001)
        dz = ToleranceSpec("normal", bound=0.002)
        other = ToleranceSpec("uniform", bound=0.0)
        t6 = ToleranceSpec6(dx, other, dz, other, other, other)
        assert t6.dx is dx
        assert t6.dz is dz
        assert t6[0] is dx
        assert t6[2] is dz

    def test_len(self):
        assert len(self._make_spec6()) == 6

    def test_sample_shape(self):
        t6 = self._make_spec6()
        out = t6.sample(500, np.random.default_rng(1))
        assert out.shape == (500, 6)

    def test_uniform_samples_in_bounds(self):
        bound = 0.05
        t6 = self._make_spec6(bound=bound, distribution="uniform")
        samples = t6.sample(10_000, np.random.default_rng(2))
        assert np.all(samples >= -bound)
        assert np.all(samples <= bound)

    def test_normal_empirical_std(self):
        bound = 0.03
        sigma_level = 3.0
        expected_sigma = bound / sigma_level
        spec = ToleranceSpec("normal", bound=bound, sigma_level=sigma_level)
        t6 = ToleranceSpec6(spec, spec, spec, spec, spec, spec)
        samples = t6.sample(200_000, np.random.default_rng(3))
        empirical_sigma = np.std(samples, axis=0)
        # All 6 columns should be within 1% of the expected standard deviation.
        np.testing.assert_allclose(empirical_sigma, expected_sigma, rtol=0.01)


# ── Asymmetric ToleranceSpec ──────────────────────────────────────────────────

class TestAsymmetricToleranceSpec:
    def test_valid_asymmetric_construction(self):
        spec = ToleranceSpec("uniform", lower=-0.002, upper=0.005)
        assert spec.is_asymmetric
        assert spec.lower == -0.002
        assert spec.upper == 0.005
        # bound auto-derived as max(|lower|, |upper|)
        assert spec.bound == pytest.approx(0.005)

    def test_asymmetric_bound_derived_from_larger_abs(self):
        # |lower| > |upper|: bound should equal |lower|
        spec = ToleranceSpec("uniform", lower=-0.010, upper=0.003)
        assert spec.bound == pytest.approx(0.010)

    def test_only_lower_raises(self):
        with pytest.raises(ValueError, match="lower and upper must be set together"):
            ToleranceSpec("uniform", lower=-0.001)

    def test_only_upper_raises(self):
        with pytest.raises(ValueError, match="lower and upper must be set together"):
            ToleranceSpec("uniform", upper=0.001)

    def test_lower_equal_upper_raises(self):
        with pytest.raises(ValueError, match="lower.*<.*upper"):
            ToleranceSpec("uniform", lower=0.001, upper=0.001)

    def test_lower_greater_than_upper_raises(self):
        with pytest.raises(ValueError, match="lower.*<.*upper"):
            ToleranceSpec("uniform", lower=0.002, upper=0.001)

    def test_symmetric_spec_is_not_asymmetric(self):
        spec = ToleranceSpec("uniform", bound=0.005)
        assert not spec.is_asymmetric

    def test_asymmetric_uniform_samples_in_range(self):
        lo, hi = -0.002, 0.005
        spec = ToleranceSpec("uniform", lower=lo, upper=hi)
        samples = spec.sample(50_000, np.random.default_rng(20))
        assert np.all(samples >= lo)
        assert np.all(samples <= hi)

    def test_asymmetric_uniform_empirical_mean(self):
        lo, hi = -0.002, 0.005
        spec = ToleranceSpec("uniform", lower=lo, upper=hi)
        samples = spec.sample(200_000, np.random.default_rng(21))
        expected_mean = (lo + hi) / 2.0
        assert np.mean(samples) == pytest.approx(expected_mean, abs=1e-4)

    def test_asymmetric_normal_mean_at_midpoint(self):
        lo, hi = -0.001, 0.003
        spec = ToleranceSpec("normal", lower=lo, upper=hi, sigma_level=3.0)
        samples = spec.sample(200_000, np.random.default_rng(22))
        expected_mean = (lo + hi) / 2.0
        assert np.mean(samples) == pytest.approx(expected_mean, abs=1e-4)

    def test_asymmetric_normal_empirical_sigma(self):
        lo, hi = -0.001, 0.003
        sigma_level = 3.0
        expected_sigma = (hi - lo) / 2.0 / sigma_level
        spec = ToleranceSpec("normal", lower=lo, upper=hi, sigma_level=sigma_level)
        samples = spec.sample(200_000, np.random.default_rng(23))
        assert np.std(samples) == pytest.approx(expected_sigma, rel=0.02)


class TestToleranceSpecVariance:
    def test_symmetric_uniform_variance(self):
        b = 0.006
        spec = ToleranceSpec("uniform", bound=b)
        assert spec.variance == pytest.approx(b**2 / 3.0)

    def test_symmetric_normal_variance(self):
        b, k = 0.009, 3.0
        spec = ToleranceSpec("normal", bound=b, sigma_level=k)
        assert spec.variance == pytest.approx((b / k) ** 2)

    def test_asymmetric_uniform_variance_symmetric_case(self):
        # When lower=-b, upper=+b the asymmetric formula must match b²/3 exactly
        b = 0.004
        spec = ToleranceSpec("uniform", lower=-b, upper=b)
        assert spec.variance == pytest.approx(b**2 / 3.0)

    def test_asymmetric_uniform_variance_off_centre(self):
        # Uniform[-0.001, 0.003]: E[X] = 0.001, Var = (0.004)²/12
        lo, hi = -0.001, 0.003
        spec = ToleranceSpec("uniform", lower=lo, upper=hi)
        mean = (lo + hi) / 2.0
        var = (hi - lo) ** 2 / 12.0
        assert spec.variance == pytest.approx(var + mean**2)

    def test_asymmetric_normal_variance_off_centre(self):
        lo, hi, k = -0.001, 0.003, 3.0
        spec = ToleranceSpec("normal", lower=lo, upper=hi, sigma_level=k)
        mean = (lo + hi) / 2.0
        sigma = (hi - lo) / 2.0 / k
        assert spec.variance == pytest.approx(sigma**2 + mean**2)

    def test_zero_bound_variance_is_zero(self):
        spec = ToleranceSpec("uniform", bound=0.0)
        assert spec.variance == 0.0


# ── locked flag must NOT suppress sampling ────────────────────────────────────

class TestLockedFlagDoesNotSuppressSampling:
    def test_locked_spec_still_sampled(self):
        """Regression: locked=True must not zero out FK-mode samples."""
        spec = ToleranceSpec("uniform", bound=0.01, locked=True)
        samples = spec.sample(1_000, np.random.default_rng(7))
        # With bound=0.01 and 1000 draws, the probability of all zeros is negligible.
        assert np.any(samples != 0.0), (
            "locked=True spec returned all zeros — this breaks FK mode. "
            "The locked flag must only affect the allocation engine."
        )


# ── skew() ───────────────────────────────────────────────────────────────────

class TestSkew:
    def test_single_vector(self):
        # skew([1, 2, 3]) = [[0,-3,2],[3,0,-1],[-2,1,0]]
        S = skew(np.array([1.0, 2.0, 3.0]))
        expected = np.array([[0., -3., 2.],
                              [3.,  0., -1.],
                              [-2., 1.,  0.]])
        np.testing.assert_allclose(S, expected, atol=1e-12)

    def test_antisymmetric(self):
        v = np.array([0.1, -0.2, 0.3])
        S = skew(v)
        np.testing.assert_allclose(S, -S.T, atol=1e-12)

    def test_batched_shape(self):
        batch = np.random.default_rng(5).uniform(-1, 1, size=(50, 3))
        S = skew(batch)
        assert S.shape == (50, 3, 3)


# ── 1. Zero-delta case ────────────────────────────────────────────────────────

class TestZeroDelta:
    def test_zero_delta_returns_nominal(self):
        nominal = HTM.from_xyz_euler([1.0, 2.0, -0.5], [0.1, -0.2, 0.05])
        N = 200
        delta_batch = np.zeros((N, 6))
        result = apply_perturbation_batch(nominal, delta_batch)
        assert result.shape == (N, 4, 4)
        for i in range(N):
            np.testing.assert_allclose(result[i], nominal.matrix, atol=1e-12)


# ── 2. Known small-angle rx perturbation ─────────────────────────────────────

class TestKnownSmallAngle:
    def test_single_rx_perturbation(self):
        # With identity nominal, perturbing by rx should give R ≈ I + skew([rx,0,0]).
        nominal = HTM.from_matrix(np.eye(4))
        rx = 1e-4  # 0.1 mrad — well within small-angle regime
        delta_batch = np.array([[0.0, 0.0, 0.0, rx, 0.0, 0.0]])
        result = apply_perturbation_batch(nominal, delta_batch)

        R_result = result[0, :3, :3]
        # First-order approximation: R ≈ I + skew([rx, 0, 0])
        R_expected_approx = np.eye(3) + skew(np.array([rx, 0.0, 0.0]))
        # After SVD orthonormalization the result is extremely close to the
        # first-order form for sub-mrad angles.
        np.testing.assert_allclose(R_result, R_expected_approx, atol=1e-9)

    def test_translation_passthrough(self):
        # Translation component of delta should appear in the output translation.
        nominal = HTM.from_matrix(np.eye(4))
        dx = 0.005
        delta_batch = np.array([[dx, 0.0, 0.0, 0.0, 0.0, 0.0]])
        result = apply_perturbation_batch(nominal, delta_batch)
        np.testing.assert_allclose(result[0, :3, 3], [dx, 0.0, 0.0], atol=1e-12)


# ── 3. Re-orthonormalization sanity ──────────────────────────────────────────

class TestReOrthonormalization:
    def test_output_rotation_is_orthonormal(self):
        # Sub-degree perturbations; check R.T @ R ≈ I and det ≈ 1.
        rng = np.random.default_rng(9)
        # Angles up to ~0.5 deg (well within small-angle regime).
        rotvec_batch = rng.uniform(-0.01, 0.01, size=(500, 3))
        R_batch = small_angle_rotation_matrix_batch(rotvec_batch)

        # Compute R^T @ R for each matrix in the batch and compare to I.
        RtR = np.einsum("nij,nik->njk", R_batch, R_batch)
        eye_batch = np.tile(np.eye(3), (R_batch.shape[0], 1, 1))
        np.testing.assert_allclose(RtR, eye_batch, atol=1e-9)

        dets = np.linalg.det(R_batch)
        np.testing.assert_allclose(dets, 1.0, atol=1e-9)

    def test_orthonormalization_does_not_distort_small_angles(self):
        # SVD projects to the nearest orthonormal matrix, which for a pure X rotation
        # gives the exact rotation matrix (with cosine on the diagonal), not the
        # first-order approximation (which has 1.0 on the diagonal). The difference
        # between SVD output and first-order form is of order rx^2/2 ≈ 1.25e-7 for
        # rx = 5e-4. Verify instead that SVD gives the exact rotation matrix.
        rx = 5e-4  # 0.5 mrad
        rotvec = np.array([[rx, 0.0, 0.0]])
        R_svd = small_angle_rotation_matrix_batch(rotvec)[0]
        # Exact Rx(rx): [[1,0,0],[0,cos,-sin],[0,sin,cos]]
        R_exact = np.array([[1.0, 0.0, 0.0],
                             [0.0, np.cos(rx), -np.sin(rx)],
                             [0.0, np.sin(rx),  np.cos(rx)]])
        np.testing.assert_allclose(R_svd, R_exact, atol=1e-9)


# ── 4. apply_perturbation_batch output shape and bottom row ──────────────────

class TestApplyPerturbationBatch:
    def test_output_shape(self):
        nominal = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        N = 1_000
        spec = ToleranceSpec("uniform", bound=0.001)
        t6 = ToleranceSpec6(spec, spec, spec, spec, spec, spec)
        delta = t6.sample(N, np.random.default_rng(11))
        result = apply_perturbation_batch(nominal, delta)
        assert result.shape == (N, 4, 4)

    def test_bottom_row_preserved(self):
        nominal = HTM.from_xyz_euler([1.0, 0.0, 0.0], [0.1, 0.0, 0.0])
        N = 100
        delta = np.random.default_rng(12).uniform(-1e-3, 1e-3, size=(N, 6))
        result = apply_perturbation_batch(nominal, delta)
        expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(N):
            np.testing.assert_allclose(result[i, 3, :], expected_bottom, atol=1e-12)
