from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import fire


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return 0
    streams = payload.get("streams") or []
    if not streams:
        return 0
    stream = streams[0]
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        try:
            frame_count = int(value)
        except (TypeError, ValueError):
            continue
        if frame_count > 0:
            return frame_count
    return 0


def _write_doro_jsonl(rows: List[Dict[str, Any]], path: Path, num_clip_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    for key in [video_stem, video_stem + ".mp4", str(row_idx)]:
        if key in predictions:
            return key

    for key, value in vid2names.items():
        if str(value) in {video_stem, video_stem + ".mp4"} and key in predictions:
            return key

    keys = list(predictions.keys())
    if row_idx < len(keys):
        return keys[row_idx]
    return None


def _normalize_predictions(test_results: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = json.loads(test_results.read_text(encoding="utf-8"))
    predictions = payload.get("predictions") or {}
    video_predictions = payload.get("video_predictions") or {}
    vid2names = payload.get("vid2names") or {}
    outputs: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        pred_key = _prediction_key_for_row(row, row_idx, predictions, vid2names)
        pred = predictions.get(pred_key) if pred_key is not None else {}
        video_pred = video_predictions.get(pred_key) if pred_key is not None else {}
        sted = video_pred.get("sted") if isinstance(video_pred, dict) else None
        target: Dict[str, List[float]] = {}
        width = float(row.get("width") or 1)
        height = float(row.get("height") or 1)
        if width <= 0:
            width = 1.0
        if height <= 0:
            height = 1.0

        for frame_idx, boxes in pred.items():
            frame_id = int(frame_idx)
            if isinstance(sted, list) and len(sted) == 2:
                if not (int(sted[0]) <= frame_id < int(sted[1])):
                    continue
            box = boxes[0] if isinstance(boxes, list) and boxes and isinstance(boxes[0], list) else boxes
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            target[str(frame_id)] = [
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            ]

        outputs.append({"target": target} if target else {})

    return outputs


def run_official_vidstg_util(
    *,
    manifest: str,
    output: str,
    repo_dir: str,
    model_weight: str,
    input_resolution: str = "224",
    num_clip_frames: str = "16",
    num_workers: str = "0",
    env_prefix: str = "CGSTVG",
    model_label: str = "CG-STVG",
) -> None:
    official_repo = Path(repo_dir).expanduser().resolve()
    rows = _load_jsonl(Path(manifest))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    converter = project_root / "eval" / "utils" / "convert_doro_to_cgstvg_vidstg.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing converter: {converter}")

    with tempfile.TemporaryDirectory(prefix="cgstvg_official_") as tmp:
        tmp_dir = Path(tmp)
        tmp_repo = tmp_dir / "repo_view"
        data_root = tmp_repo / "data" / "vidstg"
        official_output = tmp_dir / "outputs"
        doro_jsonl = tmp_dir / "manifest_as_doro.jsonl"

        _write_doro_jsonl(rows, doro_jsonl, num_clip_frames=int(num_clip_frames))
        _run(
            [
                shutil.which("python") or "python",
                str(converter),
                "--annotation-path",
                str(doro_jsonl),
                "--output-dir",
                str(tmp_repo),
                "--fps",
                "1.0",
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
            if target and target.exists():
                link_path.unlink(missing_ok=True)
                link_path.symlink_to(target)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(official_repo)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        official_python = os.getenv(f"{env_prefix}_PYTHON") or str(official_repo / ".venv" / "bin" / "python")
        _run(
            [
                official_python,
                "scripts/test_net.py",
                "--config-file",
                "experiments/vidstg.yaml",
                "MODEL.WEIGHT",
                str(Path(model_weight).expanduser().resolve()),
                "DATA_DIR",
                str(data_root),
                "OUTPUT_DIR",
                str(official_output),
                "DATALOADER.NUM_WORKERS",
                str(num_workers),
                "INPUT.RESOLUTION",
                str(input_resolution),
                "DATASET.NUM_CLIP_FRAMES",
                str(num_clip_frames),
            ],
            cwd=official_repo,
            env=env,
        )

        test_results = official_output / "test_results.json"
        if not test_results.exists():
            raise FileNotFoundError(f"Official {model_label} did not write predictions: {test_results}")

        normalized = _normalize_predictions(test_results, rows)

    with output_path.open("w", encoding="utf-8") as f:
        for item in normalized:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(
    manifest: str,
    output: str,
    cgstvg_dir: str,
    model_weight: str,
    input_resolution: str = "224",
    num_clip_frames: str = "16",
    num_workers: str = "0",
) -> None:
    run_official_vidstg_util(
        manifest=manifest,
        output=output,
        repo_dir=cgstvg_dir,
        model_weight=model_weight,
        input_resolution=input_resolution,
        num_clip_frames=num_clip_frames,
        num_workers=num_workers,
        env_prefix="CGSTVG",
        model_label="CG-STVG",
    )


if __name__ == "__main__":
    fire.Fire(main)
