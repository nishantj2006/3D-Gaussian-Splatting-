import numpy as np
from gsedit.selection.covariance_instance_graph import overlap_links


def test_broad_splat_links_beyond_fixed_center_radius():
    xyz=np.array([[0.,0,0],[2.,0,0],[8.,0,0]])
    cov=np.array([np.eye(3)*.01,np.diag([1.,.01,.01]),np.eye(3)*.01])
    result=overlap_links(xyz,cov,np.array([0]),np.array([1,2]),sigma=3)
    assert result.tolist()==[False,True,False]


def test_thin_axis_is_not_treated_as_large_sphere():
    xyz=np.array([[0.,0,0],[0.,2,0]])
    cov=np.array([np.eye(3)*.01,np.diag([100.,.01,.01])])
    assert not overlap_links(xyz,cov,np.array([0]),np.array([1]),sigma=3)[1]
