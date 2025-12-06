"""
Run a quick test of graph_generator.scene_detector.SceneDetector.
If dependencies are missing it prints install instructions.
"""
import sys
import traceback
from pathlib import Path

TEST_VIDEO = Path(__file__).resolve().parents[1] / "anno_videos" / "50_TM5MPJIq1Is_annotated_100frames.mp4"


def main():
    try:
        from graph_generator.scene_detector import SceneDetector
    except ImportError as e:
        print("ImportError:", e)
        print()
        print("Please install required packages:")
        print("  pip install scenedetect opencv-python")
        print("Or use the project's environment manager (uv):")
        print("  uv add scenedetect opencv-python")
        return 1

    if not TEST_VIDEO.exists():
        print(f"Test video not found: {TEST_VIDEO}")
        return 2

    try:
        det = SceneDetector(str(TEST_VIDEO), detector_type="adaptive", threshold=3.0, min_scene_duration=0.5)
        clips = det.detect()
        print(f"Detected {len(clips)} clips:")
        for c in clips:
            print(f"  id={c.clip_id} start={c.start_time:.3f}s end={c.end_time:.3f}s dur={c.duration:.3f}s frames={c.num_frames}")
        return 0
    except Exception:
        print("Error while running detection:\n")
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
