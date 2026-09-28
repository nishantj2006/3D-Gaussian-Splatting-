import numpy as np
from gsedit.selection.spatial_instance_removal import seeded_group, occupied_envelope


def test_disconnected_dresser_is_not_bed():
    xyz=np.array([[0,0,0],[.1,0,0],[.2,0,0],[2,0,0],[2.1,0,0]])
    chosen=seeded_group(xyz,np.ones(5,bool),[0],np.zeros(5,bool),radius=.15)
    assert chosen.tolist()==[True,True,True,False,False]


def test_protection_blocks_growth_even_when_touching():
    xyz=np.array([[0,0,0],[.1,0,0],[.2,0,0],[.3,0,0]])
    chosen=seeded_group(xyz,np.ones(4,bool),[0],np.array([0,0,1,1],bool),radius=.15)
    assert chosen.tolist()==[True,True,False,False]


def test_envelope_is_search_constraint_not_deletion():
    rng=np.random.default_rng(4)
    xyz=rng.normal(size=(40,3))
    inside,_=occupied_envelope(xyz,np.arange(20))
    result=seeded_group(xyz,inside,[],np.zeros(40,bool),radius=2)
    assert not result.any()


def test_group_repeatable():
    xyz=np.arange(60).reshape(20,3)*.01
    a=seeded_group(xyz,np.ones(20,bool),[0],np.zeros(20,bool),radius=.1)
    b=seeded_group(xyz,np.ones(20,bool),[0],np.zeros(20,bool),radius=.1)
    assert np.array_equal(a,b)
