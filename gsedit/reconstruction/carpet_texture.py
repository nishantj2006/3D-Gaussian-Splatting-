"""Transfer observed planar texture detail into unseen atlas texels.

This is a plausible texture synthesis, not recovery of hidden carpet.  It
preserves every observed texel and keeps one shared atlas across all views.
"""

import cv2
import numpy as np


def transfer_texture(observed_rgb, known, *, patch_size=128,
                     min_patch_coverage=.95, fade_pixels=12, detail_strength=.5):
    if observed_rgb.shape[:2] != known.shape or observed_rgb.shape[2] != 3:
        raise ValueError("Texture atlas and evidence mask shapes differ")
    height, width = known.shape
    size = min(patch_size, height, width)
    if size < 32:
        raise ValueError("Atlas is too small for texture transfer")
    coverage = cv2.boxFilter(known.astype(np.float32), -1, (size, size))
    y, x = np.unravel_index(int(np.argmax(coverage)), coverage.shape)
    if coverage[y, x] < min_patch_coverage:
        raise ValueError("No sufficiently observed donor texture patch")
    y0 = int(np.clip(y-size//2, 0, height-size))
    x0 = int(np.clip(x-size//2, 0, width-size))
    donor = observed_rgb[y0:y0+size, x0:x0+size].astype(np.float32)
    if np.std(donor) < 2:
        raise ValueError("Donor patch has too little texture")
    low_frequency = cv2.inpaint(observed_rgb, (~known).astype(np.uint8),
                                7, cv2.INPAINT_TELEA).astype(np.float32)
    detail = donor-cv2.GaussianBlur(donor, (0, 0), 8)
    tiled = np.tile(detail, (int(np.ceil(height/size)),
                             int(np.ceil(width/size)), 1))[:height, :width]
    distance = cv2.distanceTransform((~known).astype(np.uint8),
                                      cv2.DIST_L2, 3)
    blend = np.minimum(distance/max(fade_pixels, 1), 1)[..., None]
    result = np.clip(low_frequency+detail_strength*tiled*blend, 0, 255).astype(np.uint8)
    result[known] = observed_rgb[known]
    return result
