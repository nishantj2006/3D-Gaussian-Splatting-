# Image-first bed removal experiment (unapproved)

This is an isolated test of generating bed-free photos before updating a splat. It used existing object masks, camera poses and source photos; it did not alter the 5k scene or create a replacement PLY. All images are 384×680 diagnostics from non-held-out `frame_0138` and `frame_0144`.

`inpaint_key_views.py` supports a local Diffusers inpainting checkpoint, explicit text/negative prompts, an optional guide-image directory, seed and denoising strength. It writes source, mask, optional guide, result and peak-resource report into a new output directory. `build_imagefirst_guides.py` makes a 2D wall/carpet guide: the tentative wall is moved behind *observed* nearby carpet rays, while dark or distant 3D carpet donors are replaced with nearby observed photo carpet. This wall offset is a hypothesis for image generation only. `evaluate_keyview_consistency.py` tests projected RGB agreement for the two generated views against the same tentative surfaces; it cannot validate geometry.

## Results on the Orin

| Experiment | `frame_0138` | `frame_0144` | Wall RGB L1 | Floor RGB L1 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| OpenCV only | 0.49 s | 0.40 s | — | — | Obvious gray smear |
| Unguided SD 1.5, 20 steps | 7.48 s | 5.99 s | 0.168 | 0.121 | Invented bed/furniture in both views |
| Guided SD 1.5, strength 0.5 | 8.68 s | 7.01 s | 0.042 | 0.152 | Bed body mostly gone; panels and pale seam |
| Guided SD 1.5, strength 0.3 | 7.15 s | 5.18 s | 0.036 | 0.116 | Best 2D preview; pale seam remains; image-consistency gate fails |

RGB L1 is the mean absolute per-channel difference normalized to 0–1 for projected image pixels matched within 0.15 scene units. The reported wall/floor numbers are conditioned on the *speculative* guide wall plane, so even a low error would not prove correct 3D geometry. The best run peaked at 5,996 MiB process RAM and 2,539 MiB GPU allocation (CUDA allocation, not total unified RAM).

The visible wall fit supported only 19.61% of its training samples. To avoid visibly contradicting observed carpet, the 2D guide pushed its wall offset from 8.13 to 12.96 scene units. Floor donor distances were also large (q90 7.71 and 3.99 scene units), requiring image-space carpet donors. Those changes make a plausible *picture* but do not establish where a wall should be in 3D. The existing photos also still contain the foreground water bottle, so a full retrain against these edited photos would risk bringing the bottle back and would require updated semantic feature maps.

Best two-view images: `output/bed-ops/bed-imagefirst-guided-sd15-v2/frame_0138-inpainted.png` and `frame_0144-inpainted.png`. The final numeric check is `output/bed-ops/bed-imagefirst-consistency-guided-v2-gated/report.json` (`image_consistent: false`, `geometry_supported: false`, `approved: false`). No 3D training was started because the 2D supervision and wall geometry fail the gates. A next iteration should use a shared generated floor atlas and independently validate wall depth, or acquire more views of the uncovered wall/floor; blindly training from these two images would create another view-dependent artifact.
