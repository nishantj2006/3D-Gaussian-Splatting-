"""Robust low-frequency wall appearance from visible photo pixels."""

import numpy as np


def predict_wall_colors(photo, observed_mask, y, x):
    sy, sx = np.where(observed_mask)
    if len(sx) < 100:
        raise ValueError("Too few visible wall pixels")
    if len(sx) > 20000:
        chosen = np.linspace(0, len(sx)-1, 20000).astype(int)
        sx, sy = sx[chosen], sy[chosen]
    height, width = photo.shape[:2]

    def design(px, py):
        return np.column_stack((np.ones(len(px)),
            (px-width/2)/width, (py-height/2)/height))

    matrix = design(sx, sy)
    color = photo[sy, sx].astype(float)
    weights = np.ones(len(sx))
    for _ in range(4):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(matrix*root[:, None],
                                       color*root[:, None], rcond=None)[0]
        error = np.linalg.norm(matrix @ coefficients-color, axis=1)
        scale = max(float(np.median(error)), 1.)
        weights = np.minimum(1., 2.5*scale/np.maximum(error, 1e-6))
    return np.clip(design(x, y) @ coefficients, 0, 255).astype(np.uint8)
