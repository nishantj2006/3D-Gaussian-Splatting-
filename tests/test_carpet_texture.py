import numpy as np
import pytest

from gsedit.reconstruction.carpet_texture import transfer_texture
from gsedit.reconstruction.wall_color_field import predict_wall_colors


def test_texture_transfer_preserves_evidence_and_repeats():
    y, x = np.indices((256, 256))
    carpet = np.stack((120 + 9 * (x % 8 < 4),
                       108 + 9 * (y % 8 < 4),
                       90 + 6 * ((x + y) % 8 < 4)), axis=-1).astype(np.uint8)
    known = np.zeros((256, 256), bool)
    known[:140, :140] = True
    observed = np.zeros_like(carpet)
    observed[known] = carpet[known]
    a = transfer_texture(observed, known)
    b = transfer_texture(observed, known)
    assert np.array_equal(a, b)
    assert np.array_equal(a[known], carpet[known])
    assert np.std(a[180:230, 180:230]) > 1


def test_texture_transfer_rejects_sparse_or_flat_donors():
    rgb = np.zeros((256, 256, 3), np.uint8)
    known = np.zeros((256, 256), bool)
    known[:10, :10] = True
    with pytest.raises(ValueError, match="donor"):
        transfer_texture(rgb, known)


def test_wall_field_reconstructs_unseen_gradient():
    y, x = np.indices((100, 120))
    photo = np.stack((60 + x // 3, 40 + y // 4, 20 + x // 6), axis=-1).astype(np.uint8)
    known = y < 40
    predicted = predict_wall_colors(photo, known, np.array([70]), np.array([80]))
    assert np.max(np.abs(predicted[0].astype(int)-photo[70, 80].astype(int))) <= 2
