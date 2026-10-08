import numpy as np

from gsedit.selection.split_protected_residue_wall import choose_children, choose_parents


def test_parent_selection_uses_bed_and_dresser_support():
    ids = np.arange(4)
    chosen, _ = choose_parents(
        ids,
        inside=np.array([100., 90., 80., 70.]),
        support=np.array([4, 4, 2, 4]),
        protected_mass=np.array([10., 0., 20., 8.]),
        protected_support=np.array([3, 3, 3, 2]),
        max_scale=np.array([2., 3., 4., 1.]),
        min_bed=30., min_protected=1., min_bed_views=3,
        min_protected_views=2, min_scale=.3, max_parents=2)
    assert chosen.tolist() == [0, 3]


def test_child_cut_requires_low_dresser_contact():
    cut, agreement = choose_children(
        inside=np.array([2., 2., 2., 2.]),
        outside=np.array([0., 0., 1., 0.]),
        support=np.array([3, 3, 3, 2]),
        protected_mass=np.array([0., .1, 0., 0.]),
        protected_support=np.array([0, 1, 0, 0]),
        min_inside=.05, min_views=3, min_agreement=.85,
        max_protected_mass=.02, max_protected_support=0)
    assert cut.tolist() == [True, False, False, False]
    assert agreement[2] < .85
