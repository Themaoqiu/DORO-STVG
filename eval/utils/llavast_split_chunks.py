import argparse
import math
from pathlib import Path


def split_jsonl(input_path: Path, output_dir: Path, chunks: int) -> None:
    if chunks < 1:
        raise ValueError("chunks must be >= 1")

    lines = input_path.read_text(encoding="utf-8").splitlines()
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_len = math.ceil(len(lines) / chunks) if lines else 0

    for chunk_id in range(chunks):
        start = chunk_id * slice_len
        end = start + slice_len
        chunk_lines = lines[start:end] if slice_len else []
        output_path = output_dir / f"chunk_{chunk_id:03d}.jsonl"
        output_path.write_text("\n".join(chunk_lines) + ("\n" if chunk_lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=int)
    args = parser.parse_args()

    split_jsonl(args.input, args.output_dir, args.chunks)


if __name__ == "__main__":
    main()
