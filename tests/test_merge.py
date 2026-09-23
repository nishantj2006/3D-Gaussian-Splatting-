import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge import merge
from utils.ply_semantic_utils import read_vertices, write_vertices


class MergeTests(unittest.TestCase):
    def test_preserves_existing_object_ids_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dtype = [(name, "<f4") for name in
                     ("x", "y", "z", "scale_0", "scale_1", "scale_2", "object_id")]
            scene = np.zeros(2, dtype=dtype)
            scene["object_id"] = [0, 7]
            asset = np.zeros(1, dtype=dtype)
            write_vertices(root / "scene.ply", scene)
            write_vertices(root / "asset.ply", asset)
            args = argparse.Namespace(scene=root / "scene.ply", asset=root / "asset.ply",
                                      output=root / "merged.ply", scale=1.0,
                                      translate=(0, 0, 0), object_id=8, label="test object")
            merge(args)
            _, merged = read_vertices(args.output)
            np.testing.assert_array_equal(merged["object_id"], [0, 7, 8])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                merge(args)
            args.output = root / "another.ply"
            args.object_id = 7
            with self.assertRaisesRegex(ValueError, "already contains"):
                merge(args)


if __name__ == "__main__":
    unittest.main()
