"""Validate the optional local image stage without loading model weights."""

import argparse
import tempfile
import unittest
from pathlib import Path

from gsedit.generation.generate_local_image import object_prompt, validate_args


class LocalImageTests(unittest.TestCase):
    def test_object_prompt_is_isolated(self):
        prompt = object_prompt("blue reusable water bottle")
        self.assertIn("blue reusable water bottle", prompt)
        self.assertIn("plain white background", prompt)

    def test_rejects_bad_dimensions_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asset.png"
            args = argparse.Namespace(output=path, steps=25, width=1024,
                                      height=1024, seed=0)
            validate_args(args)
            self.assertFalse(path.exists())
            args.width = 511
            with self.assertRaisesRegex(ValueError, "width"):
                validate_args(args)
            args.width = 1024
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                validate_args(args)


if __name__ == "__main__":
    unittest.main()
