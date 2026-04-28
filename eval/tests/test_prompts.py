import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from prompts import format_prompt, parse_response  # noqa: E402


class PromptParsingTests(unittest.TestCase):
    def test_llava_st_prompt_preserves_task_text_with_official_output_instruction(self) -> None:
        prompt = format_prompt("a person opens the door", prompt_style="llava_st")

        self.assertIn("Where does a person opens the door occur in the video?", prompt)
        self.assertIn("Videos are sampled at 2 fps.", prompt)
        self.assertIn("Please firstly give the timestamps", prompt)
        self.assertNotIn("Output a strict JSON object", prompt)

    def test_parse_llava_st_official_tokens_maps_sample_position_to_frame_id(self) -> None:
        raw = (
            "The event happens during {<TEMP-010><TEMP-020>}. "
            "Object bounding box: {<TEMP-010>: [<WIDTH-010><HEIGHT-020><WIDTH-030><HEIGHT-040>]}"
        )

        parsed = parse_response(
            raw,
            query="a person opens the door",
            sampled_indices=list(range(100, 200)),
            prompt_style="llava_st",
        )

        self.assertEqual(parsed["temporal_span"], (110, 120))
        self.assertEqual(
            parsed["spatial_bboxes"],
            {110: [10 / 99.0, 20 / 99.0, 30 / 99.0, 40 / 99.0]},
        )
        self.assertEqual(parsed["objects"][0]["description"], "a person opens the door")


if __name__ == "__main__":
    unittest.main()
