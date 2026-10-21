import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from agents.write_router import write_router


class WriteRouterTests(unittest.TestCase):
    def test_existing_router_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "router.py"
            original_content = "keep this router\n"
            output_path.write_text(original_content, encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_router(output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                original_content,
            )

    def test_missing_router_can_be_generated_with_safe_missed_dose_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = write_router(Path(temporary_directory) / "router.py")
            generated_content = output_path.read_text(encoding="utf-8")

        compile(generated_content, str(output_path), "exec")
        self.assertIn("ما نجمش نحددلك وقتاش تاخذ الجرعة الناقصة", generated_content)
        self.assertNotIn("خذ الجرعة التالية في وقتها العادي", generated_content)
