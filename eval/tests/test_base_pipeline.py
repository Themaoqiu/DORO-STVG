import sys
import tempfile
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


class _DummyModel:
    def predict_batch(self, **kwargs):
        return ['{"temporal_span": [0, 1], "spatial_bboxes": {"0": [0, 0, 1, 1]}}']


class _DummyPipeline:
    pass


class BasePipelineReturnTests(unittest.TestCase):
    def test_run_evaluation_returns_metrics_only(self):
        from pipelines.base_pipeline import BasePipeline

        class DummyPipeline(BasePipeline):
            def load_data(self):
                return [
                    {
                        "video_name": "demo.mp4",
                        "video_path": "demo.mp4",
                        "query": "find target",
                        "gt_temporal_sampled": [0, 1],
                        "gt_bboxes_sampled": {"0": [0, 0, 1, 1]},
                        "metadata": {},
                    }
                ]

            def get_dataset_name(self):
                return "Dummy"

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = DummyPipeline(
                model=_DummyModel(),
                model_name="dummy-model",
                data_name="dummy",
                annotation_path=str(Path(tmp_dir) / "anno.jsonl"),
                video_dir=tmp_dir,
                output_dir=tmp_dir,
                batch_size=1,
            )

            result = pipeline.run_evaluation()

        self.assertIsInstance(result, dict)
        self.assertIn("m_tIoU", result)
        self.assertNotIn("raw_response", result)


if __name__ == "__main__":
    unittest.main()
