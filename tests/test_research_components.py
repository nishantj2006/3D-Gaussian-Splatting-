import subprocess
import sys

import numpy as np

from gsedit.evaluation.benchmark_research_components import HOLDOUT, label_instances


def test_instance_overlap_is_unknown_not_background():
    bed = np.array([[True, True], [False, False]])
    frame = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(label_instances(bed, frame), [[1, 255], [2, 0]])


def test_locked_holdouts():
    assert HOLDOUT == {"frame_0134", "frame_0141", "frame_0143"}


def test_existing_preview_rejected_before_gpu_import(tmp_path):
    existing = tmp_path / "preview"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("unchanged")
    cmd = [sys.executable, "-m", "gsedit.evaluation.benchmark_research_components", "--method", "GPGS"]
    for flag in ["upstream-dir", "source-ply", "cameras", "images", "manifest", "protected-ids"]:
        cmd += ["--" + flag, str(tmp_path / "missing")]
    cmd += ["--output-dir", str(existing)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert sentinel.read_text() == "unchanged"
