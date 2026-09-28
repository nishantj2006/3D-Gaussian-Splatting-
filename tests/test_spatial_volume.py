import numpy as np
from gsedit.selection.expand_spatial_instance import core_membership


def test_volume_requires_crossview_and_narrow_seed_band():
    result=core_membership(np.array([8,2,8,8]),np.array([8,8,8,8]),
        np.array([0,0,0,4]),np.array([.1,.1,3,.1]),.5,.85)
    assert result.tolist()==[True,False,False,False]
