"""
Test SceneDetector with JSONL save functionality.
"""
import sys
import json
import traceback
from pathlib import Path

TEST_VIDEO = Path(__file__).resolve().parents[1] / "anno_videos" / "50_TM5MPJIq1Is_annotated_100frames.mp4"
OUTPUT_JSONL = Path(__file__).resolve().parent / "scene_results.jsonl"


def main():
    try:
        from graph_generator.scene_detector import SceneDetector, SceneClip
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
        print(f"Testing SceneDetector on {TEST_VIDEO.name}...\n")
        
        det = SceneDetector(
            str(TEST_VIDEO),
            detector_type="adaptive",
            threshold=3.0,
            min_scene_duration=0.5
        )
        
        print(f"Detecting scenes...")
        clips = det.detect()
        
        print(f"Detected {len(clips)} clips:\n")
        for c in clips:
            print(f"  {c}")
        
        print(f"\nSaving to JSONL: {OUTPUT_JSONL}")
        det.save_to_jsonl(str(OUTPUT_JSONL), clips)
        
        print(f"\nJSONL content:")
        with open(OUTPUT_JSONL, 'r') as f:
            data = json.loads(f.readline())
            print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return 0
    except Exception:
        print("Error while running detection:\n")
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
