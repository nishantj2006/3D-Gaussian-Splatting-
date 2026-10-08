import numpy as np

from gsedit.reconstruction.grow_speculative_wall import recursive_donors, supported_grid


def test_recursive_growth_labels_only_active_cells():
    active = np.zeros((5, 5), bool)
    active[1:4, 1:4] = True
    labels = recursive_donors(active, np.zeros(2), 1.,
                              np.array([[1.5, 1.5], [3.5, 3.5]]))
    assert (labels[active] >= 0).all()
    assert (labels[~active] == -1).all()
    assert set(labels[active]) == {0, 1}


def test_support_counts_unique_views_not_duplicate_rays():
    points = np.array([[i+.5, j+.5] for i in range(10) for j in range(10)])
    rays = {'a': np.repeat(points, 2, axis=0), 'b': points}
    core, active, votes = supported_grid(rays, np.zeros(2), (10, 10), 1., 2, 0)
    assert votes[0, 0] == 2
    assert core[0, 0] and active[0, 0]
