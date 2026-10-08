import pytest
import torch

from gsedit.evaluation.pointmae_knn_compat import KNN


def test_knn_indices_and_layout():
    reference = torch.tensor([[[0., 0, 0], [2., 0, 0], [5., 0, 0]]])
    query = torch.tensor([[[1.9, 0, 0], [4.8, 0, 0]]])
    distances, indices = KNN(2, True)(reference, query)
    assert indices.tolist() == [[[1, 0], [2, 1]]]
    d2, i2 = KNN(2, False)(reference.transpose(1, 2), query.transpose(1, 2))
    torch.testing.assert_close(d2.transpose(1, 2), distances)
    torch.testing.assert_close(i2.transpose(1, 2), indices)


def test_knn_refuses_insufficient_reference():
    with pytest.raises(ValueError):
        KNN(4, True)(torch.zeros(1, 3, 3), torch.zeros(1, 1, 3))
