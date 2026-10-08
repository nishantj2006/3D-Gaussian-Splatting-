import numpy as np
from gsedit.selection.guard_graphcut_additions import ambiguous_additions


def test_shared_revealed_furniture_requires_two_views():
    result=ambiguous_additions(np.array([.1,.1,.01]),np.array([2,1,4]))
    assert result.tolist()==[True,False,False]
