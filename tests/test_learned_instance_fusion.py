"""Safety tests for auxiliary learned-instance evidence."""
import numpy as np
import pytest

from gsedit.selection.fuse_instance_scores import eligible_learned_bed


class Evidence:
    files = ('inside', 'outside', 'support', 'silhouette', 'contradictions')

    def __init__(self, **values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]


def example():
    probabilities = np.array([
        [.01, .98, .01], [.01, .98, .01], [.01, .98, .01],
        [.01, .98, .01], [.01, .98, .01], [.01, .98, .01],
    ], dtype=np.float32)
    evidence = Evidence(
        inside=np.array([1., 1., .01, 1., 1., 1.]),
        outside=np.array([.1, .1, 1., .1, .1, .1]),
        support=np.array([3, 3, 3, 1, 3, 3]),
        silhouette=np.array([3, 3, 3, 3, 3, 3]),
        contradictions=np.array([0, 0, 0, 0, 4, 0]),
    )
    protected = np.array([False, True, False, False, False, False])
    labels = np.array([0, 0, 0, 0, 0, 1])
    return probabilities, evidence, protected, labels


def test_probability_requires_footprint_and_protection():
    p, e, protected, labels = example()
    assert eligible_learned_bed(p, e, protected, labels).tolist() == [
        True, False, False, False, False, False]


def test_frame_conflict_and_uncertain_probability_are_rejected():
    p, e, protected, labels = example()
    p[0] = [.01, .58, .41]
    assert not eligible_learned_bed(p, e, protected, labels).any()


def test_source_alignment_and_bad_scores_fail_closed():
    p, e, protected, labels = example()
    with pytest.raises(ValueError):
        eligible_learned_bed(p[:-1], e, protected, labels)
    p[0, 1] = np.nan
    with pytest.raises(ValueError):
        eligible_learned_bed(p, e, protected, labels)
