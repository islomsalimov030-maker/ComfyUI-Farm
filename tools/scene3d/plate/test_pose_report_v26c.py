"""No GPU/network: step9c report covers every gold id and missing metrics."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_report_v26c as report


class ReportTests(unittest.TestCase):
    def test_reports_gold_missing_shots_without_invented_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(report, "read", return_value={"specs": {"Q03_test": {}}}):
                result = report.tables(tmp)
        self.assertIn("Q03_test", result)
        self.assertIn("отсутствует", result)
        self.assertIn("| D1 | 0 | — |", result)

    def test_numbers_do_not_print_non_finite(self):
        self.assertEqual(report.number(float("inf")), "—")
        self.assertEqual(report.number(None), "—")
        self.assertEqual(report.number(.85), "0.85")


if __name__ == "__main__":
    unittest.main()
