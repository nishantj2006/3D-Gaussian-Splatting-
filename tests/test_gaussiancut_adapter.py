import numpy as np
import pytest
from gsedit.selection.benchmark_gaussiancut import terminal_evidence


def test_protected_negative_overrides_high_bed_agreement():
    inside=np.array([10.]*8+[0.]*8)
    outside=np.array([0.]*8+[10.]*8)
    support=np.full(16,5)
    protected=np.zeros(16,bool);protected[0]=True
    ws,wt,pos,neg=terminal_evidence(inside,outside,support,protected)
    assert not pos[0] and neg[0]
    assert ws[0]==0 and wt[0]==1e6
    assert pos.sum()==7


def test_insufficient_evidence_is_rejected():
    with pytest.raises(ValueError,match='Insufficient'):
        terminal_evidence(np.ones(4),np.zeros(4),np.ones(4),np.zeros(4,bool))
