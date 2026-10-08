"""Exact PyTorch nearest-neighbor indices for Point-MAE's unavailable KNN plugin.

Point-MAE consumes only the returned indices. Distances here are Euclidean;
no equivalence claim is made about the unavailable plugin's distance units.
"""
class KNN:
    def __init__(self, k, transpose_mode=False):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.transpose_mode = transpose_mode

    def __call__(self, reference, query):
        import torch
        if not self.transpose_mode:
            reference, query = reference.transpose(1, 2), query.transpose(1, 2)
        if reference.ndim != 3 or query.ndim != 3 or reference.shape[0] != query.shape[0]:
            raise ValueError("Need matching batched point arrays")
        if reference.shape[1] < self.k:
            raise ValueError("Too few reference points")
        distances = torch.cdist(query, reference)
        values, indices = distances.topk(self.k, largest=False, sorted=True)
        if not self.transpose_mode:
            values, indices = values.transpose(1, 2), indices.transpose(1, 2)
        return values, indices
