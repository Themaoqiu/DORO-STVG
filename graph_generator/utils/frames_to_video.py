import subprocess
from pathlib import Path
from typing import Iterable

import fire


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _detect_frame_extension(frame_dir: Path) -> str:
    extensions = {
        path.suffix.lower()
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not extensions:
        raise ValueError(f"No supported image frames found in folder: {frame_dir}")
    if len(extensions) > 1:
        raise ValueError(f"Mixed frame extensions found in folder: {frame_dir} -> {sorted(extensions)}")
    return extensions.pop()


def _build_ffmpeg_pattern(frame_dir: Path, extension: str) -> str:
    sample_names = {path.name for path in frame_dir.iterdir() if path.is_file()}
    numbered_patterns = [
        ("00000", "%05d"),
        ("000000", "%06d"),
        ("0000000", "%07d"),
        ("00000000", "%08d"),
        ("0000", "%04d"),
        ("000", "%03d"),
        ("0", "%d"),
        ("1", "%d"),
        ("00001", "%05d"),
        ("000001", "%06d"),
    ]
    for prefix, pattern in numbered_patterns:
        if f"{prefix}{extension}" in sample_names:
            return str(frame_dir / f"{pattern}{extension}")
    raise ValueError(
        f"Could not infer ffmpeg numeric filename pattern in folder: {frame_dir}. "
        "Expected names like 00000.jpg or 00001.png."
    )


def frames_to_video(
    frame_dir: str,
    output_path: str,
    fps: float,
    codec: str = "libx264",
) -> None:
    frame_dir_path = Path(frame_dir)
    if not frame_dir_path.exists() or not frame_dir_path.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir_path}")

    extension = _detect_frame_extension(frame_dir_path)
    input_pattern = _build_ffmpeg_pattern(frame_dir_path, extension)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        codec,
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path_obj),
    ]

    print("Running ffmpeg command:", flush=True)
    print(" ".join(cmd), flush=True)

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Video created successfully: {output_path_obj}", flush=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed with exit code {exc.returncode}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg not found. Please install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        ) from exc


def iter_frame_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def run(
    input: str,
    output: str,
    fps: float,
    codec: str = "libx264",
) -> None:
    input_root = Path(input)
    output_root = Path(output)
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    frame_dirs = list(iter_frame_dirs(input_root))
    if not frame_dirs:
        raise ValueError(f"No frame subdirectories found in: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    failed = []
    for idx, frame_dir in enumerate(frame_dirs, start=1):
        output_path = output_root / f"{frame_dir.name}.mp4"
        print(f"[{idx}/{len(frame_dirs)}] {frame_dir} -> {output_path}", flush=True)
        try:
            frames_to_video(
                frame_dir=str(frame_dir),
                output_path=str(output_path),
                fps=fps,
                codec=codec,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((frame_dir, str(exc)))
            print(f"⚠️ Skip failed folder: {frame_dir}\n   Reason: {exc}", flush=True)

    print(
        f"Done. Success: {len(frame_dirs) - len(failed)}, Failed: {len(failed)}, Total: {len(frame_dirs)}",
        flush=True,
    )
    if failed:
        print("Failed folders:", flush=True)
        for frame_dir, reason in failed:
            print(f"- {frame_dir}: {reason}", flush=True)


if __name__ == "__main__":
    fire.Fire(run)
