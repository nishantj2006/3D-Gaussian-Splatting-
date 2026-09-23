import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from add_asset import (alignment_target, generate_nano_image, local_image_command,
                       run_stage, validate_generation_requirements)


class AssetPipelineTests(unittest.TestCase):
    def test_target_modes(self):
        args = argparse.Namespace(replace_original=Path("before.ply"),
                                  replace_pruned=Path("after.ply"), target_bbox=None,
                                  target_center=None, target_size=None)
        self.assertEqual(alignment_target(args),
                         ["--replace-original", "before.ply", "--replace-pruned", "after.ply"])
        args.replace_original = args.replace_pruned = None
        args.target_center = [1, 2, 3]
        args.target_size = [4, 5, 6]
        self.assertEqual(alignment_target(args),
                         ["--target-center", "1", "2", "3", "--target-size", "4", "5", "6"])
        args.target_center = args.target_size = None
        with self.assertRaises(ValueError):
            alignment_target(args)

    def test_missing_key_stops_before_api_call(self):
        with patch.dict("os.environ", {}, clear=True), tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                generate_nano_image("red bottle", Path(temp) / "image.png", "test-model")

    def test_local_provider_needs_no_api_key_or_model_load(self):
        args = argparse.Namespace(mesh=None, image=None, image_provider="local-sdxl",
                                  image_python=Path(sys.executable), local_steps=25,
                                  local_width=1024, local_height=1024, local_model="sdxl",
                                  prompt="blue bottle", seed=7)
        with patch.dict("os.environ", {}, clear=True):
            validate_generation_requirements(args)
        command = local_image_command(args, Path("/tmp/local-sdxl.png"))
        self.assertEqual(command[:2], [str(Path(sys.executable)), str(
            Path(__file__).resolve().parents[1] / "generate_local_image.py")])
        self.assertIn("blue bottle", command)
        self.assertEqual(command[-2:], ["--seed", "7"])

    def test_local_provider_rejects_bad_size_before_run(self):
        args = argparse.Namespace(mesh=None, image=None, image_provider="local-sdxl",
                                  image_python=Path(sys.executable), local_steps=25,
                                  local_width=999, local_height=1024)
        with self.assertRaisesRegex(ValueError, "local-width"):
            validate_generation_requirements(args)
        args.image = Path("supplied.png")
        validate_generation_requirements(args)

    def test_stage_requires_a_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp, patch("subprocess.run") as run:
            expected = Path(temp) / "mesh.obj"
            with self.assertRaisesRegex(RuntimeError, "missing"):
                run_stage(["echo", "okay"], expected)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
