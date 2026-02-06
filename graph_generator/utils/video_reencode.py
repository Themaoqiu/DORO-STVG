import argparse
from pathlib import Path
from typing import Optional

import cv2


def _choose_fourcc(output_path: str, fourcc: Optional[str]) -> int:
    if fourcc:
        return cv2.VideoWriter_fourcc(*fourcc)

    suffix = Path(output_path).suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        return cv2.VideoWriter_fourcc(*"avc1")
    if suffix in {".avi"}:
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def reencode_video_fps(
    input_path: str,
    output_path: str,
    target_fps: float = 2.0,
    fourcc: Optional[str] = None,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read frames from video: {input_path}")

    height, width = first_frame.shape[:2]

    step = 1
    if src_fps > 0 and target_fps > 0:
        step = max(1, int(round(src_fps / target_fps)))

    output_path = str(output_path)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        _choose_fourcc(output_path, fourcc),
        float(target_fps),
        (width, height),
        True,
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError("Cannot open video writer. Try a different --fourcc.")

    frame_idx = 0
    written = 0
    frame = first_frame
    while True:
        if frame_idx % step == 0:
            writer.write(frame)
            written += 1
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    writer.release()
    if written == 0:
        raise ValueError("No frames written. Check input video and target fps.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode a video to a target FPS.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS")
    parser.add_argument("--fourcc", default=None, help="Override fourcc (e.g. avc1, mp4v, XVID)")
    args = parser.parse_args()

    reencode_video_fps(args.input, args.output, args.fps, args.fourcc)


if __name__ == "__main__":
    main()
