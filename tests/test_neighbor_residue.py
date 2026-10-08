"""Synthetic safety checks for deleted-neighbor residue scoring."""
import numpy as np
from scipy.spatial import cKDTree

from gsedit.selection.refine_neighbor_residue import (deleted_neighbor_fraction,
                                                      safe_candidates)


def test_gaussian_neighbor_score_is_local_not_global():
    points=np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.],[8.,0.,0.]])
    cov=np.tile(np.eye(3)[None]*.01,(len(points),1,1))
    bed=np.array([True,True,False,False])
    score=deleted_neighbor_fraction(points,cov,cKDTree(points),np.array([2,3]),bed,
                                    neighbors=3)
    assert score[0]>.9
    assert score[1]<.01


def test_neighborhood_alone_cannot_select_floor_or_protected():
    score=np.array([.95,.95,.95,.95])
    learned=np.array([[.1,.8,.1]]*4)
    inside=np.ones(4)
    outside=np.zeros(4)
    support=np.array([3,3,1,3])
    protected=np.array([False,True,False,False])
    floor=np.array([False,False,False,True])
    result=safe_candidates(score,learned,inside,outside,support,protected,floor,
                           threshold=.7)
    assert result.tolist()==[True,False,False,False]
