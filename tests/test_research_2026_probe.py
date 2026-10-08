import sys
from pathlib import Path

import pytest

from gsedit.evaluation.probe_2026_methods import main, probe


@pytest.mark.parametrize("code, status", [(0, "startup_passed_not_inference"),
                                          (1, "startup_blocked")])
def test_probe_records_only_startup(tmp_path, monkeypatch, code, status):
    repo = tmp_path / "upstream"
    repo.mkdir()
    entry = repo / "entry.py"
    entry.write_text("import sys\nprint('RuntimeError: test failure')\nsys.exit(" + str(code) + ")\n")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "test-commit\n")
    result = probe(repo, entry.name, output, sys.executable)
    assert result["status"] == status
    assert result["commit"] == "test-commit"
    assert result["exit_code"] == code
    assert result["startup_peak_rss_mib"] > 0
    assert not result["inference_run"]
    assert not result["visual_quality_measured"]
    assert result["command"][-1] == "--help"
    assert Path(result["log"]).exists()


def test_existing_output_rejected_before_launch(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "candidate.ply"
    marker.write_bytes(b"unchanged")
    monkeypatch.setattr(sys, "argv", ["probe", "--upstream-root", str(tmp_path),
                                     "--output-dir", str(output)])
    with pytest.raises(FileExistsError):
        main()
    assert marker.read_bytes() == b"unchanged"
