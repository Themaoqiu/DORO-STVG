import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


class TemporalEvalTests(unittest.TestCase):
    def test_parse_temporal_response_json_span(self):
        from temporal_prompts import parse_temporal_response

        parsed = parse_temporal_response('{"temporal_span": [13, 24]}')

        self.assertEqual(parsed["temporal_span"], (13, 24))

    def test_parse_temporal_response_seconds_to_two_fps_frames(self):
        from temporal_prompts import parse_temporal_response

        parsed = parse_temporal_response("Time range: 6.5s - 12.0s")

        self.assertEqual(parsed["temporal_span"], (13, 24))

    def test_compute_temporal_metrics_only_reports_temporal_scores(self):
        from utils.temporal_metrics import compute_temporal_metrics

        metrics = compute_temporal_metrics((10, 30), (20, 40))

        self.assertEqual(metrics["tIoU"], 10 / 30)
        self.assertEqual(metrics["m_tIoU"], 10 / 30)
        self.assertEqual(metrics["tIoU@0.3"], 1.0)
        self.assertEqual(metrics["tIoU@0.5"], 0.0)
        self.assertNotIn("m_vIoU", metrics)


if __name__ == "__main__":
    unittest.main()
