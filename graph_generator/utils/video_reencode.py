import argparse
import subprocess
from pathlib import Path
from typing import Optional


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
            codec = "libx264"  # 默认使用h264
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", str(target_fps),  # 设置输出帧率
        "-c:v", codec,          # 视频编码器
        "-preset", "medium",     # 编码速度预设
        "-crf", "23",           # 质量控制 (18-28,越小质量越好)
        "-c:a", "aac",          # 音频编码器
        "-b:a", "128k",         # 音频比特率
        "-movflags", "+faststart",  # 优化mp4流式播放
        "-y",                   # 覆盖输出文件
        output_path
    ]
    
    print("Running ffmpeg command:")
    print(" ".join(cmd))
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ Video re-encoded successfully: {output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error:\n{e.stderr}")
        raise RuntimeError(f"FFmpeg failed with exit code {e.returncode}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode a video to a target FPS using ffmpeg.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS (default: 2.0)")
    parser.add_argument(
        "--codec", 
        default=None, 
        help="Video codec (default: auto-detect, e.g., libx264, libx265, mpeg4)"
    )
    args = parser.parse_args()

    reencode_video_fps(args.input, args.output, args.fps, args.codec)


if __name__ == "__main__":
    main()
