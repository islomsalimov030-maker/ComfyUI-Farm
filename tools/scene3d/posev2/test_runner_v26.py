"""Regression: step 9 diagnostics survive summary export."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2


class SummaryTests(unittest.TestCase):
    def test_summary_includes_failure_height_and_root_clamps(self):
        failure = {"check": "back_contact", "excess": .47, "unit": "m", "ok": False}
        record = {"status": "contact_fail", "validation": {"first_failure": failure},
                  "height_adjust": {"shift_z_m": .02}, "refine_root_clamps": 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / run_v2.M / "Q08"
            out.mkdir(parents=True)
            (out / "fit.json").write_text(json.dumps(record), encoding="utf-8")
            with patch.object(run_v2.pr, "ROUND", root), \
                 patch.object(run_v2.pr, "prompts", return_value=[{"id": "Q08"}]), \
                 patch("builtins.print") as printed:
                run_v2.do_summary()
            summary = json.loads((root / run_v2.M / "summary.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(summary["validation"]["first_failure"], failure)
            self.assertEqual(summary["height_adjust"], record["height_adjust"])
            self.assertEqual(summary["refine_root_clamps"], 1)
            self.assertIn("back_contact +0.47 m", printed.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
