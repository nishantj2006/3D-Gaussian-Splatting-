"""Cross-view background evidence utilities.

Only pixels outside independently labeled foreground masks become observed
evidence.  Generated colors and completed depths must use a separate channel.
"""

import numpy as np


def backproject(camera, xy, depth, image_shape):
    """Backproject pixel centers using the camera convention used by the renderer."""
    height, width = image_shape
    xy = np.asarray(xy, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or depth.shape != (len(xy),):
        raise ValueError("Invalid pixels or depths")
    if np.any(~np.isfinite(depth)) or np.any(depth <= 0):
        raise ValueError("Depth must be positive and finite")
    local = np.column_stack(((xy[:, 0] - width / 2) * depth /
                             (camera["fx"] * width / camera["width"]),
                             (xy[:, 1] - height / 2) * depth /
                             (camera["fy"] * height / camera["height"]), depth))
    return np.asarray(camera["position"], dtype=np.float64) + local @ np.asarray(
        camera["rotation"], dtype=np.float64).T


def project(camera, points, image_shape):
    height, width = image_shape
    local = (np.asarray(points, dtype=np.float64) - np.asarray(camera["position"],
             dtype=np.float64)) @ np.asarray(camera["rotation"], dtype=np.float64)
    z = local[:, 2]
    safe_z = np.where(z > 0, z, 1)
    x = local[:, 0] / safe_z * camera["fx"] * width / camera["width"] + width / 2
    y = local[:, 1] / safe_z * camera["fy"] * height / camera["height"] + height / 2
    valid = (np.isfinite(x) & np.isfinite(y) & (z > 0) & (x >= 0) &
             (x < width) & (y >= 0) & (y < height))
    return np.column_stack((x, y)), z, valid


def warp_observed(source_camera, target_camera, source_rgb, source_depth,
                  source_clean, target_hole, *, depth_tolerance=0.05):
    """Warp clean observed pixels; keep nearest sample per target pixel.

    Returns RGB, depth and observed mask. Occlusion in the *source* view is
    already handled by its rendered depth and clean semantic mask. Target-view
    object depth is deliberately not used to reject background behind it.
    """
    shape = source_depth.shape
    if source_rgb.shape != (*shape, 3) or source_clean.shape != shape:
        raise ValueError("Source image/depth/mask dimensions differ")
    ty, tx = target_hole.shape
    yy, xx = np.where(source_clean & np.isfinite(source_depth) &
                      (source_depth > 0))
    rgb = np.zeros((ty, tx, 3), source_rgb.dtype)
    depth = np.full((ty, tx), np.nan, np.float32)
    observed = np.zeros((ty, tx), bool)
    if not len(xx):
        return rgb, depth, observed
    xyz = backproject(source_camera, np.column_stack((xx, yy)),
                      source_depth[yy, xx], shape)
    xy, zz, valid = project(target_camera, xyz, (ty, tx))
    ix = np.rint(xy[:, 0]).astype(int)
    iy = np.rint(xy[:, 1]).astype(int)
    valid &= (ix >= 0) & (ix < tx) & (iy >= 0) & (iy < ty)
    valid &= target_hole[np.clip(iy, 0, ty - 1), np.clip(ix, 0, tx - 1)]
    indices = np.flatnonzero(valid)
    if len(indices):
        ordered = indices[np.argsort(zz[indices], kind="stable")]
        flat = iy[ordered] * tx + ix[ordered]
        _, first = np.unique(flat, return_index=True)
        chosen = ordered[first]
        rgb[iy[chosen], ix[chosen]] = source_rgb[yy[chosen], xx[chosen]]
        depth[iy[chosen], ix[chosen]] = zz[chosen]
        observed[iy[chosen], ix[chosen]] = True
    return rgb, depth, observed


def calibrate_prior(prior, observed_depth, observed_mask, *, min_samples=100):
    """Robust affine alignment of a monocular depth prior to scene depth.

    The prior is never accepted as evidence in observed pixels. It may only
    initialize missing depths after cross-view/geometry validation.
    """
    good = observed_mask & np.isfinite(prior) & np.isfinite(observed_depth) & (
        prior > 0) & (observed_depth > 0)
    if good.sum() < min_samples:
        raise ValueError("Too few observed depths to calibrate prior")
    x = prior[good].astype(np.float64)
    y = observed_depth[good].astype(np.float64)
    a = np.column_stack((x, np.ones_like(x)))
    weight = np.ones(len(x))
    coeff = np.array([1., 0.])
    for _ in range(8):
        coeff = np.linalg.lstsq(a * weight[:, None], y * weight, rcond=None)[0]
        residual = y - a @ coeff
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-6)
        weight = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), 1e-6))
    if coeff[0] <= 0:
        raise ValueError("Depth prior cannot be aligned monotonically")
    aligned = prior * coeff[0] + coeff[1]
    return aligned, {"scale": float(coeff[0]), "shift": float(coeff[1]),
                     "observed_samples": int(good.sum()),
                     "median_observed_error": float(np.median(np.abs(y - a @ coeff)))}
