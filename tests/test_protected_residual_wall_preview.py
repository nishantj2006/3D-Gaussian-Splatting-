import numpy as np

from gsedit.selection.protected_residual_wall_preview import select_residue
from gsedit.selection.restore_dresser_supported_splats import choose_restore


def test_selection_hard_vetoes_protected_floor_and_weak_dresser_guard():
    selected, _ = select_residue(
        inside=np.ones(5), outside=np.zeros(5), support=np.full(5, 4),
        protected_mass=np.array([0., 0., 0., .1, 0.]),
        protected_support=np.array([0, 0, 0, 1, 0]),
        protected_ids=np.array([False, True, False, False, False]),
        floor_like=np.array([False, False, True, False, False]),
        active=np.array([True, True, True, True, False]),
        min_inside=.2, min_views=3, min_agreement=.9,
        max_protected_mass=.02, max_protected_support=0)
    assert selected.tolist() == [True, False, False, False, False]


def test_selective_restoration_requires_repeated_dresser_dominance():
    chosen, fraction = choose_restore(
        np.arange(4),
        protected_mass=np.array([5., 5., 5., .5]),
        protected_support=np.array([3, 2, 3, 4]),
        bed_mass=np.array([2., 1., 10., .1]),
        min_protected_mass=1., min_protected_views=3,
        min_protected_fraction=.5)
    assert chosen.tolist() == [0]
    assert fraction[2] < .5
