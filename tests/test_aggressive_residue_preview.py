import numpy as np

from gsedit.selection.aggressive_residue_preview import choose


def test_selection_uses_footprints_without_furniture_veto():
    inside = np.array([1., 1., .05, 1.])
    outside = np.array([0., 9., 0., 0.])
    support = np.array([2, 2, 3, 1])
    selected, agreement = choose(inside, outside, support, .15, 2, .2)
    assert selected.tolist() == [True, False, False, False]
    assert agreement[0] == 1.


def test_selection_is_repeatable():
    inside = np.array([.2, .3])
    outside = np.array([.4, .1])
    support = np.array([2, 3])
    first = choose(inside, outside, support, .1, 2, .2)
    second = choose(inside, outside, support, .1, 2, .2)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
