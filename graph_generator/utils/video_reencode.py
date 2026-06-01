import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple

# Example:
#   python3 video_reencode.py \
#     --input /path/to/input.mp4 \
#     --output /path/to/output_1fps.mp4 \
#     --fps 1 \
#     --overwrite
#
#   python3 video_reencode.py \
#     --input /path/to/video_folder \
#     --output /path/to/video_folder_1fps \
#     --fps 1 \
#     --workers 8 \
#     --overwrite


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv"}
FRAME_MAP_SUFFIX = ".frame_map.tsv"


def get_frame_map_path(output_path: str) -> Path:
    output = Path(output_path)
    return output.with_suffix(f"{output.suffix}{FRAME_MAP_SUFFIX}")


def _select_codec(output_path: str, codec: Optional[str]) -> str:
    if codec is not None:
        return codec

    suffix = Path(output_path).suffix.lower()
    if suffix in {".mp4", ".m4v"}:
        return "libx264"
    if suffix in {".webm"}:
        return "libvpx-vp9"
    if suffix in {".avi"}:
        return "mpeg4"
    return "libx264"


def _build_ffmpeg_command(input_path: str, output_path: str, target_fps: float, codec: Optional[str]) -> list[str]:
    filter_chain = f"fps={target_fps},scale=trunc(iw/2)*2:trunc(ih/2)*2"
    frame_map_path = get_frame_map_path(output_path)
    selected_codec = _select_codec(output_path, codec)
    return [
        "ffmpeg",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-vf",
        filter_chain,
        "-an",
        "-stats_mux_pre",
        str(frame_map_path),
        "-stats_mux_pre_fmt",
        "{n}\t{ni}\t{pts}\t{ptsi}\t{tb}\t{tbi}",
        "-c:v",
        selected_codec,
        "-preset",
        "medium",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-y",
        output_path,
    ]


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
    cmd = _build_ffmpeg_command(input_path, output_path, target_fps, codec)

    print("Running ffmpeg command:", flush=True)
    print(" ".join(cmd), flush=True)

    try:
        subprocess.run(
            cmd,
            check=True,
        )
        print(f"✅ Video re-encoded successfully: {output_path}", flush=True)
        print(f"✅ Frame map written to: {get_frame_map_path(output_path)}", flush=True)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed with exit code {e.returncode}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )


def _reencode_one(
    src: str,
    dst: str,
    target_fps: float,
    codec: Optional[str],
    overwrite: bool,
) -> Tuple[str, bool, str]:
    src_path = Path(src)
    dst_path = Path(dst)
    frame_map_path = get_frame_map_path(dst)
    if dst_path.exists() and frame_map_path.exists() and not overwrite:
        return str(src_path), True, f"skip existing: {dst_path}"

    try:
        reencode_video_fps(src, dst, target_fps, codec)
        return str(src_path), True, str(dst_path)
    except Exception as e:  # noqa: BLE001
        return str(src_path), False, str(e)


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def run_folder(
    input_root: Path,
    output_root: Path,
    target_fps: float,
    codec: Optional[str],
    workers: int,
    overwrite: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    video_files = sorted(iter_video_files(input_root))
    if not video_files:
        print(f"⚠️ No video files found in folder: {input_root}", flush=True)
        return

    print(f"Found {len(video_files)} videos in {input_root}. Output folder: {output_root}", flush=True)
    jobs = []
    for src in video_files:
        rel = src.relative_to(input_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(src), str(dst)))

    print(f"Using {workers} parallel workers", flush=True)
    failed = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _reencode_one,
                src,
                dst,
                target_fps,
                codec,
                overwrite,
            ): (src, dst)
            for src, dst in jobs
        }
        for future in as_completed(futures):
            src, dst = futures[future]
            completed += 1
            ok_src, ok, message = future.result()
            if ok:
                print(f"[{completed}/{len(video_files)}] Done: {ok_src} -> {message}", flush=True)
            else:
                failed.append((src, message))
                print(f"[{completed}/{len(video_files)}] Failed: {src}\n   Reason: {message}", flush=True)

    print(
        f"Done. Success: {len(video_files) - len(failed)}, Failed: {len(failed)}, Total: {len(video_files)}",
        flush=True,
    )
    if failed:
        print("Failed videos:", flush=True)
        for src, reason in failed:
            print(f"- {src}: {reason}", flush=True)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel ffmpeg workers when --input is a folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
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
    run_folder(input_path, output_root, args.fps, args.codec, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
