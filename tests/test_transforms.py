"""
Tests for core/transforms.py (and implicitly core/conversions.py).

All hand-calculable — no magic constants without an explanatory comment.
"""

import numpy as np
import pytest

from core.transforms import HTM

_ATOL = 1e-9  # tight tolerance for round-trips on well-conditioned inputs
_ATOL_LOOSE = 1e-6  # for gimbal-lock-adjacent cases


# ── 1. Pure translation ──────────────────────────────────────────────────────

class TestPureTranslation:
    def test_translation_stored_correctly(self):
        T = HTM.from_xyz_euler([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(T.matrix[:3, 3], [1.0, 2.0, 3.0], atol=_ATOL)

    def test_rotation_block_is_identity(self):
        T = HTM.from_xyz_euler([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(T.matrix[:3, :3], np.eye(3), atol=_ATOL)

    def test_bottom_row(self):
        T = HTM.from_xyz_euler([5.0, -1.0, 0.0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(T.matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=_ATOL)


# ── 2. Pure single-axis rotations ────────────────────────────────────────────

class TestPureRotation:
    def test_90deg_around_z(self):
        # Intrinsic ZYX: euler=[ez, ey, ex], so ez=pi/2 rotates 90° around Z.
        T = HTM.from_xyz_euler([0.0, 0.0, 0.0], [np.pi / 2, 0.0, 0.0])
        # R_z(90°) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        expected_R = np.array([[0.0, -1.0, 0.0],
                                [1.0,  0.0, 0.0],
                                [0.0,  0.0, 1.0]])
        np.testing.assert_allclose(T.matrix[:3, :3], expected_R, atol=_ATOL)

    def test_90deg_around_x(self):
        # ex=pi/2 with ez=ey=0 gives R_x(90°) = [[1,0,0],[0,0,-1],[0,1,0]]
        T = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2])
        expected_R = np.array([[1.0, 0.0,  0.0],
                                [0.0, 0.0, -1.0],
                                [0.0, 1.0,  0.0]])
        np.testing.assert_allclose(T.matrix[:3, :3], expected_R, atol=_ATOL)

    def test_180deg_around_y(self):
        # ey=pi with ez=ex=0 gives R_y(180°) = [[-1,0,0],[0,1,0],[0,0,-1]]
        T = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.0, np.pi, 0.0])
        expected_R = np.array([[-1.0, 0.0,  0.0],
                                [ 0.0, 1.0,  0.0],
                                [ 0.0, 0.0, -1.0]])
        np.testing.assert_allclose(T.matrix[:3, :3], expected_R, atol=_ATOL)


# ── 3. Round-trips for each constructor ─────────────────────────────────────

class TestRoundTrips:
    def test_xyz_euler_roundtrip(self):
        xyz_in = np.array([1.0, -2.0, 3.5])
        euler_in = np.array([0.3, -0.2, 0.1])  # [ez, ey, ex] radians
        T = HTM.from_xyz_euler(xyz_in, euler_in)
        xyz_out, euler_out = T.to_xyz_euler()
        np.testing.assert_allclose(xyz_out, xyz_in, atol=_ATOL)
        np.testing.assert_allclose(euler_out, euler_in, atol=_ATOL)

    def test_matrix_roundtrip(self):
        # Build a known HTM then round-trip through from_matrix
        T_ref = HTM.from_xyz_euler([2.0, 0.0, -1.0], [0.1, 0.2, -0.1])
        T2 = HTM.from_matrix(T_ref.matrix)
        assert T_ref.is_close(T2, atol=_ATOL)

    def test_quaternion_roundtrip(self):
        T_ref = HTM.from_xyz_euler([0.0, 1.0, -0.5], [0.5, -0.3, 0.2])
        q, xyz = T_ref.to_quaternion()
        T2 = HTM.from_quaternion(q, xyz)
        assert T_ref.is_close(T2, atol=_ATOL)

    def test_screw_roundtrip_rotation_and_translation(self):
        axis = np.array([0.0, 0.0, 1.0])  # Z axis
        angle = 0.4                         # radians
        translation = 0.75
        T = HTM.from_screw(axis, angle, translation)
        params = T.to_screw()
        np.testing.assert_allclose(abs(params["angle"]), abs(angle), atol=_ATOL)
        np.testing.assert_allclose(abs(params["translation_along_axis"]), abs(translation), atol=_ATOL)

    def test_screw_roundtrip_pure_translation(self):
        # angle=0: pure translation along Z
        axis = np.array([0.0, 0.0, 1.0])
        T = HTM.from_screw(axis, 0.0, 2.5)
        np.testing.assert_allclose(T.matrix[:3, 3], [0.0, 0.0, 2.5], atol=_ATOL)
        assert np.allclose(T.matrix[:3, :3], np.eye(3), atol=_ATOL)

    def test_screw_zero_angle_matrix_to_screw(self):
        axis = np.array([0.0, 0.0, 1.0])
        T = HTM.from_screw(axis, 0.0, 3.0)
        params = T.to_screw()
        assert abs(params["angle"]) < 1e-9
        np.testing.assert_allclose(params["translation_along_axis"], 3.0, atol=_ATOL)


# ── 4. Composition ───────────────────────────────────────────────────────────

class TestComposition:
    def test_compose_two_translations(self):
        T1 = HTM.from_xyz_euler([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        T2 = HTM.from_xyz_euler([0.0, 2.0, 0.0], [0.0, 0.0, 0.0])
        T12 = T1.compose(T2)
        np.testing.assert_allclose(T12.matrix[:3, 3], [1.0, 2.0, 0.0], atol=_ATOL)

    def test_compose_two_rotations_around_z(self):
        # 30° + 60° around Z = 90° around Z
        T1 = HTM.from_xyz_euler([0.0, 0.0, 0.0], [np.radians(30), 0.0, 0.0])
        T2 = HTM.from_xyz_euler([0.0, 0.0, 0.0], [np.radians(60), 0.0, 0.0])
        T12 = T1.compose(T2)
        T_expected = HTM.from_xyz_euler([0.0, 0.0, 0.0], [np.radians(90), 0.0, 0.0])
        assert T12.is_close(T_expected, atol=_ATOL)

    def test_compose_matches_matrix_product(self):
        T1 = HTM.from_xyz_euler([1.0, 0.0, 0.0], [np.pi / 4, 0.0, 0.0])
        T2 = HTM.from_xyz_euler([0.0, 1.0, 0.0], [0.0, np.pi / 6, 0.0])
        T12 = T1.compose(T2)
        expected_matrix = T1.matrix @ T2.matrix
        np.testing.assert_allclose(T12.matrix, expected_matrix, atol=_ATOL)


# ── 5. Inverse ───────────────────────────────────────────────────────────────

class TestInverse:
    def _identity(self) -> HTM:
        return HTM.from_matrix(np.eye(4))

    @pytest.mark.parametrize("xyz,euler", [
        ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.3, -0.2, 0.1]),
        ([1.5, -2.0, 0.7], [0.5, 0.3, -0.4]),
    ])
    def test_compose_with_inverse_is_identity(self, xyz, euler):
        T = HTM.from_xyz_euler(xyz, euler)
        result = T.compose(T.inverse())
        assert result.is_close(self._identity(), atol=_ATOL)

    def test_inverse_closed_form_matches_linalg_inv(self):
        T = HTM.from_xyz_euler([3.0, -1.0, 0.5], [0.2, -0.1, 0.3])
        T_inv_cf = T.inverse().matrix
        T_inv_np = np.linalg.inv(T.matrix)
        np.testing.assert_allclose(T_inv_cf, T_inv_np, atol=_ATOL)

    def test_inverse_of_pure_translation(self):
        T = HTM.from_xyz_euler([3.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        T_inv = T.inverse()
        np.testing.assert_allclose(T_inv.matrix[:3, 3], [-3.0, 0.0, 0.0], atol=_ATOL)


# ── 6. Validation errors ─────────────────────────────────────────────────────

class TestValidation:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            HTM.from_matrix(np.eye(3))

    def test_bad_bottom_row_raises(self):
        M = np.eye(4)
        M[3, 3] = 1.0001
        with pytest.raises(ValueError, match="bottom row"):
            HTM.from_matrix(M)

    def test_non_orthonormal_rotation_raises(self):
        M = np.eye(4)
        M[0, 0] = 2.0  # breaks orthonormality
        with pytest.raises(ValueError, match="orthonormal"):
            HTM.from_matrix(M)

    def test_reflection_raises(self):
        # det(R) = -1 → reflection, not rotation
        M = np.eye(4)
        M[0, 0] = -1.0  # flip X axis
        with pytest.raises(ValueError, match="determinant"):
            HTM.from_matrix(M)


# ── 7. Edge case: near-gimbal-lock ───────────────────────────────────────────

class TestGimbalLock:
    def test_near_gimbal_lock_no_exception(self):
        # Pitch (ey) of 89.9° is near gimbal lock for intrinsic ZYX.
        ey = np.radians(89.9)
        T = HTM.from_xyz_euler([0.0, 0.0, 0.0], [0.1, ey, 0.2])
        # Result must still be a valid HTM (validation runs in __init__).
        assert T.matrix.shape == (4, 4)
        assert np.allclose(T.matrix[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_near_gimbal_lock_round_trip_translation(self):
        # Translation round-trip should remain exact even near gimbal lock.
        xyz_in = np.array([1.0, 2.0, 3.0])
        T = HTM.from_xyz_euler(xyz_in, [0.1, np.radians(89.9), 0.2])
        xyz_out, _ = T.to_xyz_euler()
        np.testing.assert_allclose(xyz_out, xyz_in, atol=_ATOL)
