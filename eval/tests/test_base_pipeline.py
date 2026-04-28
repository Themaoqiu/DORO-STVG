import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from pipelines.base_pipeline import BasePipeline  # noqa: E402


class DummyLlavaModel:
    prompt_style = "llava_st"

    def __init__(self) -> None:
        self.last_raw_responses: List[str] = []
        self.last_video_frame_indices: List[List[int]] = []
        self.queries: List[str] = []

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        del video_paths, system_prompt
        self.queries = list(queries)
        output = "Object bounding box: {<TEMP-005>: [<WIDTH-010><HEIGHT-020><WIDTH-030><HEIGHT-040>]}"
        self.last_raw_responses = [output]
        self.last_video_frame_indices = [list(range(30, 130))]
        return [output]


class DummyPipeline(BasePipeline):
    def __init__(self, samples: List[Dict[str, Any]] | None = None, *args, **kwargs) -> None:
        self.samples = samples or []
        super().__init__(*args, **kwargs)

    def load_data(self) -> List[Dict[str, Any]]:
        return self.samples

    def get_dataset_name(self) -> str:
        return "dummy"


class BasePipelineTests(unittest.TestCase):
    def test_llava_st_pipeline_uses_official_prompt_and_frame_index_mapping(self) -> None:
        model = DummyLlavaModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = DummyPipeline(
                model=model,
                model_name="llava-st-qwen2",
                data_name="dummy",
                annotation_path="unused.jsonl",
                video_dir=tmpdir,
                output_dir=tmpdir,
                batch_size=1,
            )
            result = pipeline._process_batch(
                [
                    {
                        "video_name": "video.mp4",
                        "video_path": "/tmp/video.mp4",
                        "query": "a target",
                        "gt_temporal_sampled": (35, 35),
                        "gt_bboxes_sampled": {35: [0.1, 0.2, 0.3, 0.4]},
                        "metadata": {},
                    }
                ]
            )[0]

        self.assertIn("Please firstly give the timestamps", model.queries[0])
        self.assertEqual(result["prediction"][0]["spatial_bboxes"][35], [10 / 99.0, 20 / 99.0, 30 / 99.0, 40 / 99.0])


if __name__ == "__main__":
    unittest.main()
