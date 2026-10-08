import sys

import pytest

from gsedit.evaluation.smoke_research_rasterizer import main


def test_existing_smoke_report_is_not_overwritten(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_bytes(b"original")
    monkeypatch.setattr(sys, "argv", ["smoke", "--method", "GPGS", "--output", str(report)])
    with pytest.raises(FileExistsError):
        main()
    assert report.read_bytes() == b"original"
