import json

from gsedit.generation.generate_surface_atlas import surface_label


def test_surface_label_uses_only_accepted_training_views(tmp_path):
    manifest = {"views": {
        "train": {"accepted": True, "selected_candidate": 1,
                  "candidates": [{"label": "carpet."}]},
        "heldout": {"accepted": True, "selected_candidate": 1,
                    "candidates": [{"label": "wood"}]},
    }}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert surface_label(path, ["train"], "floor") == "carpet"
    assert surface_label(path, ["missing"], "floor") == "floor"
