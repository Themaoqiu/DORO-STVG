import tempfile
import unittest
from pathlib import Path

from utils.frames_to_video import _build_ffmpeg_pattern, _detect_frame_extension


class FramesToVideoTests(unittest.TestCase):
    def test_ignores_files_with_leading_underscore_when_detecting_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = Path(tmp_dir)
            (frame_dir / "_00000.jpg").write_bytes(b"fake")
            (frame_dir / "00000.jpg").write_bytes(b"fake")
            (frame_dir / "00001.jpg").write_bytes(b"fake")

            extension = _detect_frame_extension(frame_dir)
            pattern = _build_ffmpeg_pattern(frame_dir, extension)

            self.assertEqual(extension, ".jpg")
            self.assertEqual(pattern, str(frame_dir / "%05d.jpg"))

    def test_raises_when_only_leading_underscore_images_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = Path(tmp_dir)
            (frame_dir / "_00000.jpg").write_bytes(b"fake")

            with self.assertRaisesRegex(ValueError, "No supported image frames found"):
                _detect_frame_extension(frame_dir)


if __name__ == "__main__":
    unittest.main()
