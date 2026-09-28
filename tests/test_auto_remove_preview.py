"""The automated workflow refuses to overwrite a previous candidate."""

import tempfile
import unittest
from types import SimpleNamespace

from gsedit.pipelines.auto_remove_preview import run


class AutoRemovePreviewTests(unittest.TestCase):
    def test_existing_preview_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(SimpleNamespace(output_dir=folder))


if __name__ == "__main__":
    unittest.main()
