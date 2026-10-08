import subprocess
import sys

import pytest
import torch

from gsedit.evaluation.checkpoint_metadata import load_lama_tensor_state


class UntrustedMetadata:
    def __reduce__(self):
        return eval, ("1 + 1",)


def test_tensor_only_generator_state(tmp_path):
    path = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {"generator.layer.weight": torch.tensor([2.]), "discriminator.weight": torch.tensor([1.])}}, path)
    state = load_lama_tensor_state(path)
    assert list(state) == ["layer.weight"]
    torch.testing.assert_close(state["layer.weight"], torch.tensor([2.]))


def test_unknown_checkpoint_global_is_rejected(tmp_path):
    path = tmp_path / "unknown.ckpt"
    torch.save({"state_dict": {"generator.weight": torch.tensor([1.])}, "metadata": UntrustedMetadata()}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint metadata"):
        load_lama_tensor_state(path)


def test_holdout_cannot_be_a_generation_input(tmp_path):
    cmd = [sys.executable, "-m", "gsedit.evaluation.benchmark_inpainting_components", "--provider", "lama"]
    for flag in ["checkpoint", "images", "manifest", "protected-manifest"]:
        cmd += ["--"+flag, str(tmp_path / "missing")]
    out = tmp_path / "output"
    cmd += ["--output-dir", str(out), "--views", "frame_0141"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Held-out photos" in result.stderr
    assert not out.exists()
