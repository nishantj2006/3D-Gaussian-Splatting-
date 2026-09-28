import numpy as np
from gsedit.evaluation.diagnose_residual_splats import categories


def test_protection_and_mixed_categories_are_distinct():
    group,_=categories(np.array([5,9,5,1,.001]),np.array([0,1,5,99,0]),
        np.array([3,3,3,3,1]),np.array([1,0,0,0,0],bool))
    assert group.tolist()==[0,1,2,3,4]


def test_single_view_is_not_declared_bed_only():
    group,_=categories(np.array([5]),np.array([0]),np.array([1]),np.array([0],bool))
    assert group.tolist()==[2]
