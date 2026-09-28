import numpy as np
from PIL import Image
import pytest

from gsedit.generation.inpaint_key_views import prepare_mask, resized_size


def test_resize_uses_model_compatible_shape():
    image = Image.new("RGB", (1080, 1920))
    assert resized_size(image, 384) == (384, 680)


def test_prepare_mask_keeps_separate_foreground_and_rejects_broad_mask(tmp_path):
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[18:22, 18:22] = 0  # Separately protected foreground object.
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    result = prepare_mask(path, (40, 40), close_px=0, dilate_px=0)
    assert not result[19, 19]
    assert result[15, 15]
    Image.fromarray(np.full((40, 40), 255, np.uint8)).save(path)
    with pytest.raises(ValueError, match="unsafe"):
        prepare_mask(path, (40, 40), close_px=0, dilate_px=0)
