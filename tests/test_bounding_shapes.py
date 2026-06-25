"""
tests/test_bounding_shapes.py — Unit tests for postprocess/bounding_shapes.py.

Coverage:
  TestFitBoundingBox        — axis-aligned box on known point clouds
  TestFitBoundingSphere     — centroid+max-distance sphere
  TestFitBoundingEllipsoid  — PCA ellipsoid, worst-case and statistical modes
  TestFitRotationCone       — angle-norm extraction, mean axis, type-hardening
  TestFitRotationBox        — per-axis rotvec bounds, type-hardening
  TestConeConeBoxAgreement  — cone and box agree on isotropic inputs
"""
import numpy as np
import pytest

from postprocess.bounding_shapes import (
    fit_bounding_box,
    fit_bounding_sphere,
    fit_bounding_ellipsoid,
    fit_rotation_cone,
    fit_rotation_box,
)

# Numerical tolerance from conftest convention
DEFAULT_ATOL = 1e-9


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit_sphere_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n points uniformly distributed on the surface of the unit sphere."""
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _ellipsoid_surface_points(
    axes: np.ndarray, center: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Return n points on the surface of an axis-aligned ellipsoid."""
    sphere = _unit_sphere_points(n, rng)
    return sphere * axes[np.newaxis, :] + center[np.newaxis, :]


# ── TestFitBoundingBox ────────────────────────────────────────────────────────

class TestFitBoundingBox:
    def test_known_points(self):
        pts = np.array([
            [1.0, -2.0,  3.0],
            [-4.0, 5.0, -6.0],
            [0.0,  0.0,  0.0],
        ])
        result = fit_bounding_box(pts)
        np.testing.assert_allclose(result["min"], [-4.0, -2.0, -6.0], atol=DEFAULT_ATOL)
        np.testing.assert_allclose(result["max"], [ 1.0,  5.0,  3.0], atol=DEFAULT_ATOL)

    def test_single_point(self):
        pt = np.array([[1.0, 2.0, 3.0]])
        result = fit_bounding_box(pt)
        np.testing.assert_allclose(result["min"], result["max"], atol=DEFAULT_ATOL)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fit_bounding_box(np.zeros((5, 4)))

    def test_pose_array_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fit_bounding_box(np.zeros((10, 4, 4)))


# ── TestFitBoundingSphere ─────────────────────────────────────────────────────

class TestFitBoundingSphere:
    def test_sphere_surface_points(self):
        """fit_bounding_sphere returns the sample centroid and max-distance radius.

        The center is the sample mean (not the true geometric center), so for a
        finite point cloud on a sphere, center ≠ true_center. What we verify is:
          1. center == np.mean(pts)  (implementation invariant)
          2. all points are within radius of center  (enclosure guarantee)
        """
        rng = np.random.default_rng(0)
        R = 5.0
        true_center = np.array([1.0, -2.0, 3.0])
        pts = _unit_sphere_points(1000, rng) * R + true_center
        result = fit_bounding_sphere(pts)
        # Center must equal the sample mean (that's all the conservative sphere promises)
        np.testing.assert_allclose(result["center"], np.mean(pts, axis=0), atol=DEFAULT_ATOL)
        # Every point must be within radius of center
        dists = np.linalg.norm(pts - result["center"], axis=1)
        assert np.all(dists <= result["radius"] + DEFAULT_ATOL)

    def test_encloses_all_points(self):
        """Every point must be within radius of center."""
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((500, 3)) * 2.0
        result = fit_bounding_sphere(pts)
        dists = np.linalg.norm(pts - result["center"], axis=1)
        assert np.all(dists <= result["radius"] + DEFAULT_ATOL)

    def test_single_point_zero_radius(self):
        pt = np.array([[3.0, 4.0, 5.0]])
        result = fit_bounding_sphere(pt)
        assert result["radius"] == pytest.approx(0.0, abs=DEFAULT_ATOL)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fit_bounding_sphere(np.zeros((5, 6)))


# ── TestFitBoundingEllipsoid ──────────────────────────────────────────────────

class TestFitBoundingEllipsoid:
    def test_axis_aligned_worst_case(self):
        """Exact vertex points of known axis-aligned ellipsoid → exact recovery.

        Use 6 deterministic vertices (±a_i along each axis) rather than a random
        surface sample so the sample mean is exactly [0,0,0] and the covariance is
        exactly diagonal — no finite-sample bias to fight.
        """
        axes = np.array([3.0, 1.5, 0.5])
        # 6 points: ±a_i * e_i for each axis.  Sample mean is exactly [0,0,0].
        pts = np.vstack([np.diag(axes), np.diag(-axes)])   # shape (6,3)
        result = fit_bounding_ellipsoid(pts, coverage=1.0)
        np.testing.assert_allclose(result["center"], np.zeros(3), atol=DEFAULT_ATOL)
        np.testing.assert_allclose(result["axes_lengths"], axes, atol=1e-6)

    def test_worst_case_encloses_all(self):
        """Every point must lie within the coverage=1.0 ellipsoid."""
        rng = np.random.default_rng(3)
        pts = rng.standard_normal((300, 3)) * np.array([2.0, 1.0, 0.3])
        result = fit_bounding_ellipsoid(pts, coverage=1.0)
        center = result["center"]
        lengths = result["axes_lengths"]
        directions = result["axes_directions"]   # columns = principal axes
        # Ellipsoidal coordinate of each point: max over axes of |projection / length|
        X = pts - center
        projections = X @ directions             # (N,3)
        # Point i is inside if sum((proj[i,j]/lengths[j])^2) <= 1
        inside = np.sum((projections / lengths[np.newaxis, :]) ** 2, axis=1)
        assert np.all(inside <= 1.0 + 1e-9)

    def test_statistical_smaller_than_worst_case(self):
        """coverage=0.997 ellipsoid must be strictly smaller than coverage=1.0."""
        rng = np.random.default_rng(4)
        pts = rng.standard_normal((5000, 3))
        stat = fit_bounding_ellipsoid(pts, coverage=0.997)
        worst = fit_bounding_ellipsoid(pts, coverage=1.0)
        # At least one axis of the statistical ellipsoid must be shorter
        assert np.any(stat["axes_lengths"] < worst["axes_lengths"])

    def test_known_sphere_statistical(self):
        """Isotropic normal: statistical ellipsoid at coverage≈0.997 → radius ≈ 3σ."""
        rng = np.random.default_rng(5)
        sigma = 2.0
        pts = rng.standard_normal((100_000, 3)) * sigma
        result = fit_bounding_ellipsoid(pts, coverage=0.9973)
        # All three axes should be ≈ 3σ (chi2 scaling with df=3)
        from scipy.stats import chi2
        expected = sigma * np.sqrt(chi2.ppf(0.9973, df=3))
        np.testing.assert_allclose(result["axes_lengths"], expected, rtol=0.02)

    def test_invalid_coverage_raises(self):
        pts = np.ones((10, 3))
        with pytest.raises(ValueError, match="coverage"):
            fit_bounding_ellipsoid(pts, coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            fit_bounding_ellipsoid(pts, coverage=1.5)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fit_bounding_ellipsoid(np.zeros((5, 2)))


# ── TestFitRotationCone ───────────────────────────────────────────────────────

class TestFitRotationCone:
    def test_isotropic_max_angle(self):
        """Isotropic rotvec cloud: max_angle equals the largest norm in the batch."""
        rng = np.random.default_rng(6)
        n = 1000
        # Known angles, random axes
        angles = rng.uniform(0.0, 0.005, size=n)
        axes = _unit_sphere_points(n, rng)
        rotvecs = axes * angles[:, np.newaxis]
        result = fit_rotation_cone(rotvecs)
        np.testing.assert_allclose(result["max_angle"], np.max(angles), atol=DEFAULT_ATOL)

    def test_single_axis_known_angle(self):
        """Rotvecs all along +z with known angle → max_angle and mean_axis match."""
        theta = 0.003   # 3 mrad
        rotvecs = np.tile([0.0, 0.0, theta], (50, 1))
        result = fit_rotation_cone(rotvecs)
        assert result["max_angle"] == pytest.approx(theta, abs=DEFAULT_ATOL)
        np.testing.assert_allclose(result["mean_axis"], [0.0, 0.0, 1.0], atol=1e-6)

    def test_zero_rotvecs_no_error(self):
        """All-zero rotvecs: max_angle=0, mean_axis defaults to [0,0,1]."""
        rotvecs = np.zeros((20, 3))
        result = fit_rotation_cone(rotvecs)
        assert result["max_angle"] == pytest.approx(0.0, abs=DEFAULT_ATOL)
        np.testing.assert_allclose(result["mean_axis"], [0.0, 0.0, 1.0], atol=DEFAULT_ATOL)

    def test_mean_axis_is_unit_vector(self):
        """mean_axis must always be a unit vector."""
        rng = np.random.default_rng(7)
        rotvecs = rng.standard_normal((200, 3)) * 0.001
        result = fit_rotation_cone(rotvecs)
        assert np.linalg.norm(result["mean_axis"]) == pytest.approx(1.0, abs=1e-9)

    def test_rejects_pose_array(self):
        """Passing (N,4,4) must raise ValueError with a helpful message."""
        poses = np.zeros((10, 4, 4))
        with pytest.raises(ValueError, match=r"\(N,4,4\)"):
            fit_rotation_cone(poses)

    def test_rejects_wrong_columns(self):
        """Passing (N,6) must also raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            fit_rotation_cone(np.zeros((10, 6)))


# ── TestFitRotationBox ────────────────────────────────────────────────────────

class TestFitRotationBox:
    def test_known_rotvecs(self):
        """Per-axis min/max matches the known extremes of the input."""
        rotvecs = np.array([
            [ 0.001, -0.002,  0.003],
            [-0.004,  0.005, -0.006],
            [ 0.0,    0.0,    0.0  ],
        ])
        result = fit_rotation_box(rotvecs)
        np.testing.assert_allclose(result["min"], [-0.004, -0.002, -0.006], atol=DEFAULT_ATOL)
        np.testing.assert_allclose(result["max"], [ 0.001,  0.005,  0.003], atol=DEFAULT_ATOL)

    def test_rejects_pose_array(self):
        with pytest.raises(ValueError, match=r"\(N,4,4\)"):
            fit_rotation_box(np.zeros((10, 4, 4)))

    def test_rejects_wrong_columns(self):
        with pytest.raises(ValueError, match="shape"):
            fit_rotation_box(np.zeros((10, 6)))


# ── TestConeConeBoxAgreement ──────────────────────────────────────────────────

class TestConeConeBoxAgreement:
    def test_isotropic_agreement(self):
        """Isotropic rotvecs: cone max_angle ≈ max(|box.min|, |box.max|) per axis.

        For purely isotropic noise each individual axis has similar magnitude to
        the full-norm cone angle — the cone's max_angle should be >= the per-axis
        extremes but not dramatically larger (within sqrt(3) for isotropic noise).
        """
        rng = np.random.default_rng(8)
        n = 5000
        # Purely isotropic: same sigma on all three axes
        sigma = 0.001
        rotvecs = rng.normal(0, sigma, size=(n, 3))

        cone = fit_rotation_cone(rotvecs)
        box = fit_rotation_box(rotvecs)

        per_axis_max = np.max(np.abs(np.stack([box["min"], box["max"]])))

        # Cone must be >= any per-axis extreme (it's the full-norm)
        assert cone["max_angle"] >= per_axis_max - DEFAULT_ATOL
        # For isotropic noise, cone should be within sqrt(3) of the per-axis max
        assert cone["max_angle"] <= per_axis_max * np.sqrt(3) + 1e-6

    def test_single_axis_cone_matches_box(self):
        """Rotvecs along a single axis: cone max_angle == box max on that axis."""
        rotvecs = np.zeros((30, 3))
        rotvecs[:, 0] = np.linspace(-0.005, 0.005, 30)   # only rx varies
        cone = fit_rotation_cone(rotvecs)
        box = fit_rotation_box(rotvecs)
        # cone max_angle = 0.005 = abs(box max along axis 0)
        assert cone["max_angle"] == pytest.approx(0.005, abs=DEFAULT_ATOL)
        assert max(abs(box["min"][0]), abs(box["max"][0])) == pytest.approx(0.005, abs=DEFAULT_ATOL)
        # Other axes: box should be zero
        np.testing.assert_allclose(box["min"][1:], 0.0, atol=DEFAULT_ATOL)
        np.testing.assert_allclose(box["max"][1:], 0.0, atol=DEFAULT_ATOL)
