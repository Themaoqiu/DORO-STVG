import json
import tempfile
import unittest
from pathlib import Path

from eval.prompts import SYSTEM_PROMPT, format_prompt
from graph_generator.utils.format_sft import main


class FormatSFTTests(unittest.TestCase):
    def test_eval_json_prompt_style_from_query_raw(self) -> None:
        row = {
            "query_id": "sample_1",
            "query": "well groomed",
            "video_path": "/dataset/video/clip.mp4",
            "video_width": 640,
            "video_height": 360,
            "target_members": [
                {
                    "object_id": "dog_1",
                    "boxes": {
                        "15": [64.0, 36.0, 320.0, 180.0],
                        "16": [128.0, 72.0, 384.0, 216.0],
                    },
                }
            ],
            "per_target_queries": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "query_raw.jsonl"
            output_path = Path(tmp_dir) / "sharegpt.jsonl"
            input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            main(
                input=str(input_path),
                output=str(output_path),
                media_dir="/dataset/video",
                path_mode="relative",
                include_system=True,
                prompt_style="eval_json",
            )

            sample = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(sample["videos"], ["clip.mp4"])
        self.assertEqual(sample["system"], SYSTEM_PROMPT)
        self.assertEqual(sample["messages"][0], {"role": "user", "content": f"<video>{format_prompt('well groomed')}"})

        assistant_payload = json.loads(sample["messages"][1]["content"])
        self.assertEqual(
            assistant_payload,
            {
                "well groomed": {
                    "15": [0.1, 0.1, 0.5, 0.5],
                    "16": [0.2, 0.2, 0.6, 0.6],
                }
            },
        )

    def test_eval_json_prompt_style_keeps_multiple_targets(self) -> None:
        row = {
            "query_id": "sample_2",
            "query": "the two people",
            "video_path": "/dataset/video/pair.mp4",
            "video_width": 100,
            "video_height": 50,
            "target_members": [
                {"object_id": "person_1", "boxes": {"0": [10, 5, 20, 15]}},
                {"object_id": "person_2", "boxes": {"0": [40, 10, 60, 20]}},
            ],
            "per_target_queries": {"target 1": "the left person", "target 2": "the right person"},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "query_raw.jsonl"
            output_path = Path(tmp_dir) / "sharegpt.jsonl"
            input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            main(
                input=str(input_path),
                output=str(output_path),
                media_dir="/dataset/video",
                path_mode="relative",
                include_system=False,
                prompt_style="eval_json",
            )

            sample = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(sample["system"], "")
        assistant_payload = json.loads(sample["messages"][1]["content"])
        self.assertEqual(
            assistant_payload,
            {
                "the left person": {"0": [0.1, 0.1, 0.2, 0.3]},
                "the right person": {"0": [0.4, 0.2, 0.6, 0.4]},
            },
        )


if __name__ == "__main__":
    unittest.main()
