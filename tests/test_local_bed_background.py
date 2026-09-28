"""Safety and determinism checks for preview-only bed refinement helpers."""

import numpy as np
import pytest

from gsedit.selection.attribute_revealed_bed import classify
from gsedit.reconstruction.refine_local_background import candidate_pool
from utils.ply_semantic_utils import read_vertices, write_vertices


def test_candidate_pool_excludes_removed_and_synthetic_points():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    selected = np.array([0])
    seed = np.array([[1., 0., 0.], [3., 0., 0.], [.1, 0., 0.]])
    assert candidate_pool(source, selected, seed, 2, 1.1).tolist() == [0]
    with pytest.raises(ValueError, match="Seed does not match"):
        candidate_pool(source, selected, seed, 1, 1.1)


def test_classification_preserves_ambiguous_boundary_and_is_repeatable():
    evidence = (np.array([9., 9., .1]), np.array([.2, 7., .01]),
                np.array([3, 3, 1]))
    first = classify(*evidence, min_inside=.5, min_views=2,
                     min_agreement=.85, boundary_min_agreement=.55)
    second = classify(*evidence, min_inside=.5, min_views=2,
                      min_agreement=.85, boundary_min_agreement=.55)
    assert first[0].tolist() == [True, False, False]
    assert first[1].tolist() == [False, True, False]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_ply_preserves_all_properties_and_128d_features(tmp_path):
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("opacity", "<f4")] + [(f"semantic_{i}", "<f4") for i in range(128)]
    vertices = np.zeros(3, dtype=fields)
    for i in range(128):
        vertices[f"semantic_{i}"] = i + np.arange(3)
    original = tmp_path / "original.ply"
    candidate = tmp_path / "candidate.ply"
    write_vertices(original, vertices)
    metadata, readback = read_vertices(original)
    changed = readback.copy()
    changed["opacity"][1] = -7
    write_vertices(candidate, changed, metadata, ["UNAPPROVED preview"])
    _, final = read_vertices(candidate)
    assert final.dtype == readback.dtype
    assert final.dtype.names == vertices.dtype.names
    for name in vertices.dtype.names:
        if name == "opacity":
            continue
        np.testing.assert_array_equal(final[name], readback[name])
    assert final["opacity"][1] == -7
    np.testing.assert_array_equal(read_vertices(original)[1], readback)
