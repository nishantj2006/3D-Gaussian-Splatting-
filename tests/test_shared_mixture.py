import numpy as np
from gsedit.selection.refine_shared_splats import split_mixture,choose_children
from gsedit.selection.covariance_instance_graph import covariances


def test_mixture_preserves_covariance_and_features():
    dtype=[(k,'f8') for k in ['x','y','z','opacity']+
           [f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+['semantic_0']]
    record=np.zeros((),dtype=dtype);record['rot_0']=1
    record['scale_0']=np.log(3);record['scale_1']=np.log(2);record['semantic_0']=.42
    children=split_mixture(record)
    centers=np.column_stack([children[k] for k in ['x','y','z']])
    combined=covariances(children).mean(axis=0)+centers.T@centers/4
    assert np.allclose(combined,np.diag([9,4,1]))
    assert np.all(children['semantic_0']==.42)


def test_protected_daughter_cannot_be_cut():
    selected=choose_children(np.array([10,10]),np.array([0,0]),np.array([4,4]),
                             np.array([.1,0]))
    assert selected.tolist()==[False,True]
