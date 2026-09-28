"""Exact property preservation for reversible local opacity edits."""

import numpy as np
import pytest

from gsedit.reconstruction.materialize_local_gates import apply_gates


def test_gate_materialization_changes_only_requested_opacity():
    vertices = np.zeros(4, dtype=[("x", "<f4"), ("opacity", "<f4"),
                                   ("semantic_0", "<f4")])
    vertices["opacity"] = [0, 1, 2, 3]
    vertices["semantic_0"] = [4, 5, 6, 7]
    result = apply_gates(vertices, np.array([0, 1, 2]),
                         np.array([1., .5, .001]))
    assert result["opacity"][0] == vertices["opacity"][0]
    assert result["opacity"][3] == vertices["opacity"][3]
    assert result["opacity"][1] < vertices["opacity"][1]
    np.testing.assert_array_equal(result["semantic_0"], vertices["semantic_0"])
    again = apply_gates(vertices, np.array([0, 1, 2]), np.array([1., .5, .001]))
    np.testing.assert_array_equal(result, again)


@pytest.mark.parametrize("indices,gates", [
    (np.array([1, 1]), np.array([.5, .5])),
    (np.array([10]), np.array([.5])),
    (np.array([0]), np.array([np.nan])),
    (np.array([0]), np.array([1.5])),
])
def test_invalid_gates_are_rejected(indices, gates):
    vertices = np.zeros(2, dtype=[("opacity", "<f4")])
    with pytest.raises(ValueError):
        apply_gates(vertices, indices, gates)
