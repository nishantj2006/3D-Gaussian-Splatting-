import numpy as np
from PIL import Image
import pytest
import torch

from gsedit.generation.generate_background_views import inpaint_diffusion


@pytest.mark.skipif(not torch.cuda.is_available(), reason="helper uses CUDA generator")
def test_inpainting_requests_portrait_dimensions_and_preserves_exterior():
    class FakePipeline:
        def __call__(self, **kwargs):
            assert (kwargs["height"], kwargs["width"]) == (80, 48)
            return type("Result", (), {"images": [Image.new("RGB", (48, 80), "red")]})()

    photo = np.full((80, 48, 3), 130, np.uint8)
    mask = np.zeros((80, 48), bool)
    mask[20:60, 10:38] = True
    result = inpaint_diffusion(FakePipeline(), photo, mask, "room", "bed", 1, 0)
    assert result.shape == photo.shape
    np.testing.assert_array_equal(result[~mask], photo[~mask])
    assert result[mask, 0].mean() == 255
