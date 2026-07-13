"""OBB collision geometry: SAT correctness against analytic cases.
Pure geometry — no engine state.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.worlds.obb import (
    Manifold,
    obb_vertices,
    obb_vs_obb,
    obb_vs_plane,
    obb_vs_sphere,
    quat_to_matrix,
)

IDENTITY = np.eye(3)


def _rot_z(angle: float) -> np.ndarray:
    half = angle / 2.0
    return quat_to_matrix(np.array([math.cos(half), 0.0, 0.0, math.sin(half)]))


def _rot_y(angle: float) -> np.ndarray:
    half = angle / 2.0
    return quat_to_matrix(np.array([math.cos(half), 0.0, math.sin(half), 0.0]))


def test_quat_to_matrix_is_rotation():
    for q in ([1, 0, 0, 0], [0.7071068, 0.7071068, 0, 0], [0.5, 0.5, 0.5, 0.5]):
        rotation = quat_to_matrix(np.array(q, dtype=np.float64))
        np.testing.assert_allclose(rotation @ rotation.T, IDENTITY, atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_vertices_of_rotated_cube():
    rotation = _rot_z(math.pi / 4.0)
    vertices = obb_vertices(np.zeros(3), rotation, np.array([1.0, 1.0, 1.0]))
    assert vertices.shape == (8, 3)
    # 45° about z: corner radius in xy becomes sqrt(2).
    xy_radii = np.linalg.norm(vertices[:, :2], axis=1)
    np.testing.assert_allclose(xy_radii, math.sqrt(2.0), atol=1e-9)


# ── plane contacts ───────────────────────────────────────────────

def test_flat_box_on_plane_gives_four_point_manifold():
    manifold = obb_vs_plane(
        np.array([0.0, 0.0, 0.45]), IDENTITY, np.array([0.5, 0.5, 0.5]), 0.0)
    assert manifold is not None and len(manifold.points) == 4
    for contact in manifold.points:
        assert contact.penetration == pytest.approx(0.05, abs=1e-12)
    np.testing.assert_allclose(manifold.normal, [0, 0, -1])


def test_edge_balanced_box_touches_on_two_corners():
    # Cube rotated 45° about y: an edge points down. Center height set so
    # the lowest edge dips 0.01 below the plane.
    rotation = _rot_y(math.pi / 4.0)
    lowest = math.sqrt(2.0) * 0.5  # corner depth below center
    manifold = obb_vs_plane(
        np.array([0.0, 0.0, lowest - 0.01]), rotation,
        np.array([0.5, 0.5, 0.5]), 0.0)
    assert manifold is not None
    assert 1 <= len(manifold.points) <= 4
    assert manifold.max_penetration == pytest.approx(0.01, abs=1e-9)
    deep = [c for c in manifold.points if c.penetration > 1e-12]
    assert len(deep) == 2  # exactly the two edge corners


def test_clear_box_has_no_plane_contact():
    assert obb_vs_plane(
        np.array([0.0, 0.0, 2.0]), _rot_y(0.3), np.array([0.5, 0.5, 0.5]), 0.0
    ) is None


# ── sphere contacts ──────────────────────────────────────────────

def test_sphere_face_contact_distance_is_analytic():
    manifold = obb_vs_sphere(
        np.zeros(3), IDENTITY, np.array([1.0, 1.0, 1.0]),
        np.array([1.3, 0.0, 0.0]), 0.5)
    assert manifold is not None
    assert manifold.max_penetration == pytest.approx(0.2, abs=1e-12)
    np.testing.assert_allclose(manifold.normal, [1, 0, 0], atol=1e-12)


def test_sphere_corner_contact_normal_is_diagonal():
    manifold = obb_vs_sphere(
        np.zeros(3), IDENTITY, np.array([1.0, 1.0, 1.0]),
        np.array([1.2, 1.2, 1.2]), 0.6)
    assert manifold is not None
    expected = np.ones(3) / math.sqrt(3.0)
    np.testing.assert_allclose(manifold.normal, expected, atol=1e-9)


def test_separated_sphere_misses():
    assert obb_vs_sphere(
        np.zeros(3), IDENTITY, np.array([1.0, 1.0, 1.0]),
        np.array([3.0, 0.0, 0.0]), 0.5) is None


# ── OBB vs OBB (SAT) ─────────────────────────────────────────────

def test_axis_aligned_overlap_depth_is_exact():
    manifold = obb_vs_obb(
        np.zeros(3), IDENTITY, np.array([1.0, 1.0, 1.0]),
        np.array([1.9, 0.0, 0.0]), IDENTITY, np.array([1.0, 1.0, 1.0]))
    assert manifold is not None
    np.testing.assert_allclose(manifold.normal, [1, 0, 0], atol=1e-9)
    assert manifold.max_penetration == pytest.approx(0.1, abs=1e-9)
    assert len(manifold.points) >= 1


def test_separated_boxes_report_none_even_when_aabbs_overlap():
    # Two long thin boxes rotated ±45°: AABBs overlap, SAT separates them.
    rotation_a = _rot_z(math.pi / 4.0)
    rotation_b = _rot_z(-math.pi / 4.0)
    manifold = obb_vs_obb(
        np.array([0.0, 1.6, 0.0]), rotation_a, np.array([2.0, 0.1, 0.1]),
        np.array([0.0, -1.6, 0.0]), rotation_b, np.array([2.0, 0.1, 0.1]))
    assert manifold is None


def test_rotated_overlap_detected_with_sane_normal():
    manifold = obb_vs_obb(
        np.zeros(3), IDENTITY, np.array([1.0, 1.0, 1.0]),
        np.array([1.5, 0.0, 0.0]), _rot_z(math.pi / 4.0),
        np.array([1.0, 1.0, 1.0]))
    assert manifold is not None
    # Normal points from A toward B and is dominated by +x.
    assert manifold.normal[0] > 0.7
    assert manifold.max_penetration > 0.0
    assert 1 <= len(manifold.points) <= 4


def test_sat_agrees_with_vertex_containment_probe():
    """Property check: whenever SAT reports no contact, no sampled point
    of either box lies inside the other."""
    rng = np.random.default_rng(7)
    for _ in range(60):
        center_b = rng.uniform(-3.0, 3.0, size=3)
        angle = float(rng.uniform(0.0, math.pi))
        rotation_b = _rot_z(angle) @ _rot_y(float(rng.uniform(0.0, math.pi)))
        half_b = rng.uniform(0.3, 1.2, size=3)
        manifold = obb_vs_obb(
            np.zeros(3), IDENTITY, np.array([1.0, 1.0, 1.0]),
            center_b, rotation_b, half_b)
        if manifold is None:
            # Probe: vertices of B must be outside A (and vice versa).
            for vertex in obb_vertices(center_b, rotation_b, half_b):
                assert np.any(np.abs(vertex) > 1.0 + 1e-9), (center_b, angle)
