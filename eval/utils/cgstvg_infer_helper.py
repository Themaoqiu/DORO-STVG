from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _probe_frame_count(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return 0

    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        return 0

    for key in ("nb_read_frames", "nb_frames"):
        value = streams[0].get(key)
        try:
            frame_count = int(value)
        except (TypeError, ValueError):
            continue
        if frame_count > 0:
            return frame_count
    return 0


def _write_dummy_doro_manifest(rows: List[Dict[str, Any]], path: Path, num_clip_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            video_path = Path(str(row.get("video_path") or row.get("video_input_path") or ""))
            frame_count = _probe_frame_count(video_path.expanduser())
            if frame_count <= 0:
                frame_count = max(4, int(num_clip_frames))

            width = int(row.get("width") or 1)
            height = int(row.get("height") or 1)
            payload = {
                "video_path": video_path.name,
                "video_input_path": str(video_path),
                "query": str(row.get("query") or ""),
                "queryid": row.get("queryid") or f"cgstvg_{idx}",
                "Width": max(width, 1),
                "Height": max(height, 1),
                "box": json.dumps(
                    {
                        "target": {
                            str(frame_idx): [0.0, 0.0, 1.0, 1.0]
                            for frame_idx in range(frame_count)
                        }
                    },
                    ensure_ascii=False,
                ),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.stdout.strip():
        print(proc.stdout, end="")
    if proc.stderr.strip():
        print(proc.stderr, end="", file=os.sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def _prediction_key_for_row(
    row: Dict[str, Any],
    row_idx: int,
    predictions: Dict[str, Any],
    vid2names: Dict[str, Any],
) -> str | None:
    video_stem = Path(str(row.get("video_path") or row.get("video_input_path") or "")).stem
    for key in [video_stem, f"{video_stem}.mp4", str(row_idx)]:
        if key in predictions:
            return key

    for key, value in vid2names.items():
        if str(value) in {video_stem, f"{video_stem}.mp4"} and key in predictions:
            return key

    keys = list(predictions.keys())
    if row_idx < len(keys):
        return keys[row_idx]
    return None


def _normalize_predictions(result_path: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions") or {}
    video_predictions = payload.get("video_predictions") or {}
    vid2names = payload.get("vid2names") or {}
    outputs: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        pred_key = _prediction_key_for_row(row, row_idx, predictions, vid2names)
        pred = predictions.get(pred_key) if pred_key is not None else {}
        video_pred = video_predictions.get(pred_key) if pred_key is not None else {}
        sted = video_pred.get("sted") if isinstance(video_pred, dict) else None
        width = max(float(row.get("width") or 1), 1.0)
        height = max(float(row.get("height") or 1), 1.0)

        target: Dict[str, List[float]] = {}
        for frame_idx, boxes in pred.items():
            frame_id = int(frame_idx)
            if isinstance(sted, list) and len(sted) == 2 and not (int(sted[0]) <= frame_id < int(sted[1])):
                continue

            box = boxes[0] if isinstance(boxes, list) and boxes and isinstance(boxes[0], list) else boxes
            if not isinstance(box, list) or len(box) != 4:
                continue

            x1, y1, x2, y2 = [float(value) for value in box]
            target[str(frame_id)] = [
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            ]

        outputs.append({"target": target} if target else {})

    return outputs


def _build_test_net_command(
    *,
    python_bin: str,
    config_file: str,
    model_weight: str,
    data_dir: str,
    output_dir: str,
    num_workers: str | None,
    input_resolution: str | None,
    num_clip_frames: str | None,
    extra_opts: Iterable[str] | None = None,
    test_module: str = "dependence.cgstvg.scripts.test_net",
) -> List[str]:
    cmd = [
        python_bin,
        "-m",
        test_module,
        "--config-file",
        config_file,
        "MODEL.WEIGHT",
        model_weight,
        "DATA_DIR",
        data_dir,
        "OUTPUT_DIR",
        output_dir,
    ]
    if num_workers is not None:
        cmd.extend(["DATALOADER.NUM_WORKERS", str(num_workers)])
    if input_resolution is not None:
        cmd.extend(["INPUT.RESOLUTION", str(input_resolution)])
    if num_clip_frames is not None:
        cmd.extend(["DATASET.NUM_CLIP_FRAMES", str(num_clip_frames)])
    if extra_opts is not None:
        cmd.extend(list(extra_opts))
    return cmd


def _split_extra_opts(extra_opts: str | None) -> List[str]:
    if not extra_opts:
        return []
    return extra_opts.split()


def _create_runtime_dir(base_dir: Path, prefix: str) -> tempfile.TemporaryDirectory[str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=str(base_dir))


def run_official_vidstg_helper(
    *,
    manifest: str,
    output: str,
    model_weight: str,
    config_file: str = "dependence/cgstvg/experiments/vidstg.yaml",
    test_module: str = "dependence.cgstvg.scripts.test_net",
    result_file: str = "test_results.json",
    input_resolution: str | None = None,
    num_clip_frames: str | None = None,
    num_workers: str | None = None,
    extra_opts: str | None = None,
    env_prefix: str = "CGSTVG",
    model_label: str = "CG-STVG",
) -> None:
    rows = _load_jsonl(Path(manifest))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    runtime_base_dir = project_root / "res" / "cgstvg_runtime"
    converter = project_root / "utils" / "convert_doro_to_cgstvg_vidstg.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing converter: {converter}")

    with _create_runtime_dir(runtime_base_dir, prefix="cgstvg_official_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        tmp_repo = tmp_dir / "repo_view"
        data_root = tmp_repo / "data" / "vidstg"
        official_output = tmp_dir / "outputs"
        doro_jsonl = tmp_dir / "manifest_as_doro.jsonl"

        clip_frames = int(num_clip_frames) if num_clip_frames is not None else 32
        _write_dummy_doro_manifest(rows, doro_jsonl, num_clip_frames=clip_frames)
        _run(
            [
                shutil.which("python3") or shutil.which("python") or "python3",
                str(converter),
                "--annotation-path",
                str(doro_jsonl),
                "--output-dir",
                str(tmp_repo),
                "--fps",
                "2.0",
            ],
            cwd=project_root,
            env=os.environ.copy(),
        )

        videos_dir = data_root / "videos"
        for link_path in videos_dir.glob("*.mp4"):
            target = next(
                (
                    Path(str(row.get("video_path") or row.get("video_input_path"))).expanduser().resolve()
                    for row in rows
                    if Path(str(row.get("video_path") or row.get("video_input_path") or "")).name == link_path.name
                ),
                None,
            )
            if target is not None:
                link_path.unlink(missing_ok=True)
                link_path.symlink_to(target)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        official_python = os.getenv(f"{env_prefix}_PYTHON") or "python3"
        cmd = _build_test_net_command(
            python_bin=official_python,
            test_module=test_module,
            config_file=config_file,
            model_weight=str(Path(model_weight).expanduser().resolve()),
            data_dir=str(data_root),
            output_dir=str(official_output),
            num_workers=num_workers,
            input_resolution=input_resolution,
            num_clip_frames=num_clip_frames,
            extra_opts=_split_extra_opts(extra_opts),
        )
        _run(cmd, cwd=project_root, env=env)

        test_results = official_output / result_file
        if not test_results.exists():
            raise FileNotFoundError(f"Official {model_label} did not write predictions: {test_results}")

        normalized = _normalize_predictions(test_results, rows)

    with output_path.open("w", encoding="utf-8") as handle:
        for item in normalized:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vendored official CG-STVG inference for DORO-STVG manifest rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-weight", required=True)
    parser.add_argument("--config-file", default="dependence/cgstvg/experiments/vidstg.yaml")
    parser.add_argument("--test-module", default="dependence.cgstvg.scripts.test_net")
    parser.add_argument("--result-file", default="test_results.json")
    parser.add_argument("--input-resolution")
    parser.add_argument("--num-clip-frames")
    parser.add_argument("--num-workers")
    parser.add_argument("--official-extra-opts")
    args = parser.parse_args()

    run_official_vidstg_helper(
        manifest=args.manifest,
        output=args.output,
        model_weight=args.model_weight,
        config_file=args.config_file,
        test_module=args.test_module,
        result_file=args.result_file,
        input_resolution=args.input_resolution,
        num_clip_frames=args.num_clip_frames,
        num_workers=args.num_workers,
        extra_opts=args.official_extra_opts,
        env_prefix="CGSTVG",
        model_label="CG-STVG",
    )


if __name__ == "__main__":
    main()
