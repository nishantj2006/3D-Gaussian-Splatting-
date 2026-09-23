import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruct_flat import (build_candidate, choose_donor, commit, exclude_original_records,
                              extend_target_from_view_mask, fit_local_color_correction,
                              fit_plane, make_target, removed_indices, sha256)


class ReconstructFlatTests(unittest.TestCase):
    def test_local_color_fit_recovers_gradient_and_rejects_sparse_boundary(self):
        rng = np.random.default_rng(13)
        ring_uv = rng.uniform(-0.6, 0.6, (500, 2))
        donor = np.full((500, 3), 0.55)
        true_offset = np.array([0.04, -0.03, 0.02])
        true_slope = np.array([[0.035, 0.015, -0.02],
                               [-0.025, 0.01, 0.025]])
        target = donor + true_offset + ring_uv @ true_slope
        target[:30] += 0.4  # Outliers should not determine the correction.
        fill_uv = rng.uniform(-0.4, 0.4, (100, 2))
        correction, coefficients = fit_local_color_correction(
            ring_uv, target, donor, fill_uv, np.zeros(2))
        expected = true_offset + fill_uv @ true_slope
        self.assertLess(np.mean(np.abs(correction - expected)), 0.02)
        self.assertEqual(coefficients.shape, (3, 3))
        with self.assertRaisesRegex(ValueError, "Not enough clean carpet"):
            fit_local_color_correction(
                ring_uv[:50], target[:50], donor[:50], fill_uv, np.zeros(2))

    def test_reviewed_residual_exclusion_preserves_fill_and_rejects_invalid_rows(self):
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("semantic_0", "<f4")])
        original = np.zeros(3, dtype=dtype)
        original["x"] = [1, 2, 3]
        original["semantic_0"] = [0.1, 0.2, 0.3]
        candidate = np.concatenate((original, original[:1].copy()))
        candidate[-1]["x"] = 4
        result, indices = exclude_original_records(original, candidate, 1, [1])
        self.assertEqual(indices, [1])
        np.testing.assert_array_equal(result["x"], [1, 3, 4])
        np.testing.assert_allclose(result["semantic_0"], [0.1, 0.3, 0.1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            exclude_original_records(original, candidate, 1, [1, 1])
        with self.assertRaisesRegex(ValueError, "out of range"):
            exclude_original_records(original, candidate, 1, [3])
        with self.assertRaisesRegex(ValueError, "already be removed"):
            exclude_original_records(original, candidate[[0, 2, 3]], 1, [1])

    def test_ordered_subset_and_schema_validation(self):
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("semantic_0", "f4")])
        original = np.zeros(4, dtype=dtype)
        original["x"] = [0, 1, 2, 3]
        kept = original[[0, 2]]
        np.testing.assert_array_equal(removed_indices(original, kept), [False, True, False, True])
        with self.assertRaises(ValueError):
            removed_indices(original, original[[2, 0]])
        with self.assertRaises(ValueError):
            removed_indices(original, kept[["x", "y", "z"]])

    def test_nonplanar_surface_is_rejected(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(5000, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        with self.assertRaisesRegex(ValueError, "No dominant flat surface"):
            fit_plane(points, points[:500])

    def test_zero_repair_margin_and_view_mask_projection(self):
        rng = np.random.default_rng(11)
        uv = rng.normal(0, 0.03, (300, 2))
        target, low, cell, _ = make_target(uv, np.zeros(len(uv)), repair_margin=0)
        self.assertLess(target.sum(), 500)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = {"img_name": "floor", "width": 512, "height": 512,
                      "fx": 300, "fy": 300, "position": [0, 0, -3],
                      "rotation": np.eye(3).tolist()}
            cameras_path = root / "cameras.json"
            cameras_path.write_text(json.dumps([camera]))
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[260:280, 275:295] = 255
            mask_path = root / "mask.png"
            Image.fromarray(mask).save(mask_path)
            enlarged, _, extra = extend_target_from_view_mask(
                target, low, cell, mask_path, "floor", cameras_path,
                np.zeros(3), np.array([0, 0, 1]),
                np.array([1, 0, 0]), np.array([0, 1, 0]))
            self.assertGreater(extra, 0)
            self.assertGreater(enlarged.sum(), target.sum())

    def test_donor_poor_surface_is_rejected(self):
        vertices = np.zeros(100, dtype=[("f_dc_0", "f4"), ("f_dc_1", "f4"),
                                       ("f_dc_2", "f4")])
        uv = np.zeros((100, 2))
        distance = np.zeros(100)
        target = np.ones((5, 5), dtype=bool)
        with self.assertRaisesRegex(ValueError, "Not enough clean nearby donor"):
            choose_donor(uv, distance, vertices, target, np.array([-0.05, -0.05]), 0.025)

    def test_flat_fill_preserves_ply_schema_semantics_and_is_deterministic(self):
        rng = np.random.default_rng(7)
        n_floor, n_object = 30000, 500
        dtype = np.dtype(
            [(name, "<f4") for name in
             ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
              "opacity", "semantic_0", "semantic_1", "object_id")]
        )
        original = np.zeros(n_floor + n_object, dtype=dtype)
        original["x"][:n_floor] = rng.uniform(-2, 2, n_floor)
        original["y"][:n_floor] = rng.uniform(-2, 2, n_floor)
        original["z"][:n_floor] = rng.normal(0, 0.002, n_floor)
        original["x"][n_floor:] = rng.uniform(-0.2, 0.2, n_object)
        original["y"][n_floor:] = rng.uniform(-0.2, 0.2, n_object)
        original["z"][n_floor:] = rng.uniform(0.15, 0.8, n_object)
        for i in range(3):
            original[f"f_dc_{i}"] = (0.7 - 0.5) / 0.2820947918
            original[f"scale_{i}"] = np.log(0.01)
        original["rot_0"] = 1
        original["opacity"] = 1
        original["semantic_0"][:n_floor] = 0.8
        original["semantic_1"][:n_floor] = 0.2
        original["object_id"][n_floor:] = 5
        missing = np.zeros(len(original), dtype=bool)
        missing[n_floor:] = True
        missing[:n_floor] = ((original["x"][:n_floor] ** 2 +
                              original["y"][:n_floor] ** 2) < 0.3 ** 2)
        pruned = original[~missing]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = {"img_name": "floor", "width": 512, "height": 512,
                      "fx": 300, "fy": 300, "position": [0, 0, -3],
                      "rotation": np.eye(3).tolist()}
            cameras_path = root / "cameras.json"
            cameras_path.write_text(json.dumps([camera]))
            Image.new("RGB", (512, 512), (179, 179, 179)).save(root / "floor.png")
            first, details = build_candidate(original, pruned, missing, cameras_path, root,
                                             seed=3, cell=0.025, grid_step=0.01,
                                             repair_margin=0)
            second, _ = build_candidate(original, pruned, missing, cameras_path, root,
                                        seed=3, cell=0.025, grid_step=0.01,
                                        repair_margin=0)
            blended, blended_details = build_candidate(
                original, pruned, missing, cameras_path, root,
                seed=3, cell=0.025, grid_step=0.01, repair_margin=0,
                local_color_match=True, seam_blend_width=0.05)
            supported, supported_details = build_candidate(
                original, pruned, missing, cameras_path, root,
                seed=3, cell=0.025, grid_step=0.01, repair_margin=0,
                distance_support_scale=0.02)
            with self.assertRaisesRegex(ValueError, "at least twice"):
                build_candidate(original, pruned, missing, cameras_path, root,
                                seed=3, cell=0.025, grid_step=0.01, repair_margin=0,
                                distance_support_scale=0.015)
            with self.assertRaisesRegex(ValueError, "align to --cell"):
                build_candidate(original, pruned, missing, cameras_path, root,
                                seed=3, cell=0.025, grid_step=0.01, repair_margin=0,
                                donor_shift=(0.012, 0.0))
        self.assertEqual(first.dtype, original.dtype)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(len(blended), len(first))
        self.assertGreater(blended_details["local_color_samples"], 100)
        self.assertEqual(blended.dtype, original.dtype)
        self.assertEqual(supported.dtype, original.dtype)
        self.assertGreater(supported_details["distance_support_gaussians"], 0)
        self.assertTrue(np.array_equal(supported[:len(first)], first))
        support = supported[-supported_details["distance_support_gaussians"]:]
        self.assertTrue(np.allclose(np.exp(support["scale_0"]), 0.02))
        self.assertTrue(np.all(support["semantic_0"] == 0.8))
        self.assertTrue(np.all(support["semantic_1"] == 0.2))
        self.assertGreater(details["added_donor_gaussians"], 500)
        clones = first[-details["added_donor_gaussians"]:]
        self.assertTrue(np.all(clones["object_id"] == 0))
        self.assertTrue(np.all(clones["semantic_0"] == 0.8))
        self.assertTrue(np.all(clones["semantic_1"] == 0.2))

    def test_commit_requires_review_and_unchanged_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.ply"
            candidate.write_bytes(b"preview")
            (root / "view.png").write_bytes(b"render")
            (root / "preview.json").write_text(json.dumps({
                "views": ["view"], "candidate_sha256": sha256(candidate),
                "original": str(root / "original.ply"), "pruned": str(root / "pruned.ply")
            }))
            args = Namespace(preview_dir=str(root), output=str(root / "approved.ply"), approve=False)
            with self.assertRaisesRegex(ValueError, "--approve"):
                commit(args)
            args.approve = True
            candidate.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after preview"):
                commit(args)
            candidate.write_bytes(b"preview")
            commit(args)
            self.assertEqual((root / "approved.ply").read_bytes(), b"preview")


if __name__ == "__main__":
    unittest.main()
