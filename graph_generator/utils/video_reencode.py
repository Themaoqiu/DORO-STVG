import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv"}


def reencode_video_fps(
    input_path: str,
    output_path: str,
    target_fps: float = 2.0,
    codec: Optional[str] = None,
) -> None:
    input_path = str(input_path)
    output_path = str(output_path)

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if codec is None:
        suffix = Path(output_path).suffix.lower()
        if suffix in {".mp4", ".m4v"}:
            codec = "libx264"
        elif suffix in {".webm"}:
            codec = "libvpx-vp9"
        elif suffix in {".avi"}:
            codec = "mpeg4"
        else:
            codec = "libx264"

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", str(target_fps),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", codec,
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]

    print("Running ffmpeg command:", flush=True)
    print(" ".join(cmd), flush=True)

    try:
        subprocess.run(
            cmd,
            check=True,
        )
        print(f"✅ Video re-encoded successfully: {output_path}", flush=True)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed with exit code {e.returncode}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode a video to a target FPS using ffmpeg.")
    parser.add_argument("--input", required=True, help="Input video path or folder path")
    parser.add_argument("--output", default=None, help="Output video path or output folder path")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS (default: 2.0)")
    parser.add_argument(
        "--codec",
        default=None,
        help="Video codec (default: auto-detect, e.g., libx264, libx265, mpeg4)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if args.output is None:
            raise ValueError("--output is required when --input is a video file.")
        reencode_video_fps(str(input_path), args.output, args.fps, args.codec)
        return

    output_root = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_reencoded"
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(iter_video_files(input_path))
    if not video_files:
        print(f"⚠️ No video files found in folder: {input_path}", flush=True)
        return

    print(f"Found {len(video_files)} videos in {input_path}. Output folder: {output_root}", flush=True)
    failed = []
    for idx, src in enumerate(video_files, 1):
        rel = src.relative_to(input_path)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(video_files)}] Re-encoding: {src} -> {dst}", flush=True)
        try:
            reencode_video_fps(str(src), str(dst), args.fps, args.codec)
        except Exception as e:  # noqa: BLE001
            failed.append((src, str(e)))
            print(f"⚠️ Skip failed video: {src}\n   Reason: {e}", flush=True)

    print(
        f"Done. Success: {len(video_files) - len(failed)}, Failed: {len(failed)}, Total: {len(video_files)}",
        flush=True,
    )
    if failed:
        print("Failed videos:", flush=True)
        for src, reason in failed:
            print(f"- {src}: {reason}", flush=True)


if __name__ == "__main__":
    main()
