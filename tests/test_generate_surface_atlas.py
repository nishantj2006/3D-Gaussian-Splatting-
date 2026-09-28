import numpy as np

from gsedit.generation.generate_surface_atlas import (accumulate_atlas, atlas_layout,
                                    backproject_plane, complete_atlas,
                                    sample_atlas, surface_basis)


def test_surface_basis_places_floor_and_perpendicular_wall():
    fit = {"origin": [0, 0, 0], "frame": np.eye(3).tolist(),
           "wall_normal_floor_xy": [0, 1], "wall_offset": 2}
    floor_origin, floor_axes, floor_normal = surface_basis(fit, "floor")
    wall_origin, wall_axes, wall_normal = surface_basis(fit, "wall")
    np.testing.assert_allclose(floor_origin, [0, 0, 0])
    np.testing.assert_allclose(wall_origin, [0, 2, 0])
    assert abs(np.dot(floor_normal, wall_normal)) < 1e-8
    assert abs(np.dot(wall_axes[:, 0], wall_normal)) < 1e-8


def test_backproject_hits_plane_and_rejects_parallel_ray():
    camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
              "width": 40, "height": 40, "fx": 20, "fy": 20}
    xyz, depth = backproject_plane(camera, np.array([[20, 20]]), (40, 40),
                                    np.array([0, 0, 5]), np.array([0, 0, 1]))
    np.testing.assert_allclose(xyz, [[0, 0, 5]])
    np.testing.assert_allclose(depth, [5])


def test_observed_atlas_pixels_are_preserved_and_reused_in_3d():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    origin = np.zeros(3)
    basis = np.eye(3)[:, :2]
    low, high, shape = atlas_layout(points, origin, basis, resolution=64, margin=0)
    sums = np.zeros((*shape, 3), float)
    counts = np.zeros(shape, np.int32)
    uv = points[:2, :2]
    assert accumulate_atlas(sums, counts, uv, np.array([[255, 0, 0],
                                                       [0, 255, 0]]), low, high) == 2
    observed = counts > 0
    rgb = np.zeros((*shape, 3), np.uint8)
    rgb[observed] = np.rint(sums[observed] / counts[observed, None]).astype(np.uint8)
    result = complete_atlas(rgb, observed, backend="opencv", pipe=None,
                            prompt="", steps=1, seed=0, radius=3)
    np.testing.assert_array_equal(result[observed], rgb[observed])
    sampled = sample_atlas(result, uv, low, high)
    assert sampled[0, 0] > .9 and sampled[1, 1] > .9
