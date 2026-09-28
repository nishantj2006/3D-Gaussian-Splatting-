import numpy as np

from gsedit.selection.split_mixed_splats import split_record


def test_split_preserves_all_features_and_distributes_centers():
    dtype = [(k, "f4") for k in ("x", "y", "z", "scale_0", "scale_1",
                                     "scale_2", "rot_0", "rot_1", "rot_2",
                                     "rot_3", "opacity", "semantic_0", "f_dc_0")]
    point = np.zeros((), dtype=dtype)
    point["x"] = 2
    point["scale_0"] = np.log(2)
    point["rot_0"] = 1
    point["opacity"] = 0
    point["semantic_0"] = .32
    point["f_dc_0"] = -.21
    daughter = split_record(point)
    assert len(daughter) == 4
    assert np.allclose(np.column_stack([daughter[k] for k in ("x", "y", "z")]).mean(axis=0),
                       [2, 0, 0])
    assert len(set(zip(daughter["x"], daughter["y"], daughter["z"]))) == 4
    assert np.allclose(daughter["semantic_0"], .32)
    assert np.allclose(daughter["f_dc_0"], -.21)
