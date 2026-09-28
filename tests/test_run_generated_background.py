"""Workflow argument and held-out metric checks without GPU rendering."""

import numpy as np
from PIL import Image

from gsedit.pipelines.run_generated_background import cli_options, masked_metrics


def test_cli_options_supports_list_bool_and_snake_case():
    assert cli_options({"holdout_views": ["a", "b"], "frame_manifest": None,
                        "dry_run": True, "width": 512}) == [
        "--holdout-views", "a", "b", "--dry-run", "--width", "512"]


def test_heldout_metrics_separate_inside_and_outside(tmp_path):
    before = np.full((8, 8, 3), 100, dtype=np.uint8)
    after = before.copy()
    after[:4] = 125
    mask = np.zeros((8, 8), np.uint8)
    mask[:4] = 255
    Image.fromarray(before).save(tmp_path / "before.png")
    Image.fromarray(after).save(tmp_path / "after.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    report = masked_metrics(tmp_path / "before.png", tmp_path / "after.png",
                            tmp_path / "mask.png")
    assert report["inside_l1_change"] > 0
    assert report["outside_l1_change"] == 0
    assert report["inside_near_black_fraction"] == 0
