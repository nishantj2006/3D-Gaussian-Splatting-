import numpy as np
import pytest

from gsedit.pipelines.run_autonomous_edit import diverse_holdouts, reconcile_selection, stage


def test_holdouts_are_diverse_and_only_accepted():
    cameras = {f"v{i}": {"position": [i, 0, 0]} for i in range(10)}
    manifest = {"views": {f"v{i}": {"accepted": i != 9} for i in range(10)}}
    chosen = diverse_holdouts(cameras, manifest)
    assert len(chosen) == 3 and len(set(chosen)) == 3
    assert "v9" not in chosen


def test_protected_instances_are_not_removed_and_large_conflict_rejects(tmp_path):
    initial = tmp_path/"initial.npy"
    protected = tmp_path/"protected.npy"
    np.save(initial, np.arange(100))
    np.save(protected, np.array([0]))
    report = reconcile_selection(initial, protected, tmp_path/"safe.npy")
    assert report == {"selected": 99, "protected_conflicts": 1}
    assert 0 not in np.load(tmp_path/"safe.npy")
    np.save(protected, np.arange(50))
    with pytest.raises(ValueError, match="conflicts"):
        reconcile_selection(initial, protected, tmp_path/"unsafe.npy")
    assert not (tmp_path/"unsafe.npy").exists()


def test_failed_stage_retains_log_and_status(tmp_path):
    report = {"output": str(tmp_path), "stages": {}}
    with pytest.raises(RuntimeError, match="missing-script"):
        stage("missing-script.py", {"output_dir": str(tmp_path/"stage")},
              report, "missing-script")
    assert report["stages"]["missing-script"]["exit_code"] != 0
    assert (tmp_path/"missing-script.log").exists()

