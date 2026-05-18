from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


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
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames", "-of", "json", str(video_path),
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
    for key in ("nb_read_frames", "nb_frames"):
        try:
            frame_count = int(streams[0].get(key))
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
                "queryid": row.get("queryid") or f"tastvg_{idx}",
                "Width": max(width, 1),
                "Height": max(height, 1),
                "box": json.dumps({
                    "target": {
                        str(frame_idx): [0.0, 0.0, 1.0, 1.0]
                        for frame_idx in range(frame_count)
                    }
                }, ensure_ascii=False),
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


def _prediction_key_for_row(row: Dict[str, Any], row_idx: int, predictions: Dict[str, Any], vid2names: Dict[str, Any]) -> Any:
    video_stem = Path(str(row.get("video_path") or row.get("video_input_path") or "")).stem
    for key in [video_stem, video_stem + ".mp4", str(row_idx), row_idx]:
        if key in predictions:
            return key
    for key, value in vid2names.items():
        if str(value) in {video_stem, video_stem + ".mp4", str(row_idx)} and key in predictions:
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
        width = max(float(row.get("width") or 1), 1.0)
        height = max(float(row.get("height") or 1), 1.0)

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


def _subject_for_row(row: Dict[str, Any]) -> str:
    query = str(row.get("query") or "").strip()
    return query or "target"


def _tastvg_label(subject: str) -> Dict[str, Any]:
    return {
        "sub": subject,
        "verb_index_list": [],
        "adj_index_list": [],
        "noun_index_list": [],
        "obj_index_list": [],
    }


def _write_tastvg_label_files(annos_dir: Path, rows: List[Dict[str, Any]], generated_pairs: Path | None = None) -> None:
    annos_dir.mkdir(parents=True, exist_ok=True)
    label_map: Dict[str, Dict[str, str]] = {}
    if generated_pairs and generated_pairs.exists():
        try:
            pairs = json.loads(generated_pairs.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pairs = []
        if isinstance(pairs, list):
            for pair_idx, pair in enumerate(pairs):
                if not isinstance(pair, dict):
                    continue
                item_id = pair.get("item_id")
                if item_id is None:
                    continue
                row = rows[pair_idx] if pair_idx < len(rows) else {}
                label_map[str(item_id)] = _tastvg_label(_subject_for_row(row))

    if not label_map:
        for idx, row in enumerate(rows):
            subject = _subject_for_row(row)
            for key in {str(idx), str(idx + 1), str(row.get("queryid") or "")}:
                if key:
                    label_map[key] = _tastvg_label(subject)

    for split_name in ("test.json", "train.json"):
        (annos_dir / split_name).write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_tastvg_cache(
    official_repo: Path,
    data_root: Path,
    env: Dict[str, str],
    official_python: str,
    rows: List[Dict[str, Any]],
) -> None:
    annos_dir = data_root / "annos"
    annos_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_root / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for cache_name in ("vidstd-test-input.cache", "vidstd-test-anno.cache"):
        (cache_dir / cache_name).unlink(missing_ok=True)

    official_test = annos_dir / "test.json"
    official_train = annos_dir / "train.json"
    if not official_test.exists():
        _write_tastvg_label_files(annos_dir, rows)
    if not official_train.exists():
        _write_tastvg_label_files(annos_dir, rows)

    tmp_official = annos_dir / "test.json.tmp_doro"
    generated_pairs = annos_dir / "test.json.doro_pairs"
    tmp_official.unlink(missing_ok=True)
    generated_pairs.unlink(missing_ok=True)

    official_test.rename(tmp_official)
    try:
        build_cache_code = (
            "from config import cfg\n"
            "from datasets.vidstg import VidSTGDataset\n"
            "cfg.merge_from_file('experiments/vidstg.yaml')\n"
            f"cfg.DATA_DIR = {str(data_root)!r}\n"
            "cfg.DATA_TRUNK = None\n"
            "ds = VidSTGDataset(cfg, 'test')\n"
            "print(f'dataset_len = {len(ds)}')\n"
        )
        _run([official_python, "-c", build_cache_code], cwd=official_repo, env=env)
        generated_test = annos_dir / "test.json"
        if generated_test.exists():
            generated_test.rename(generated_pairs)
    finally:
        if tmp_official.exists():
            tmp_official.rename(official_test)
    _write_tastvg_label_files(annos_dir, rows, generated_pairs)


def run_tastvg_helper(
    *,
    manifest: str,
    output: str,
    tastvg_dir: str,
    model_weight: str,
    input_resolution: str,
    num_clip_frames: str,
    num_workers: str,
    config_file: str,
    test_script: str,
    result_file: str,
    prepare_cache: bool,
    official_extra_opts: List[str],
) -> None:
    official_repo = Path(tastvg_dir).expanduser().resolve()
    rows = _load_jsonl(Path(manifest))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    converter = project_root / "eval" / "utils" / "convert_doro_to_tastvg_vidstg.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing converter: {converter}")

    with tempfile.TemporaryDirectory(prefix="tastvg_official_") as tmp:
        tmp_dir = Path(tmp)
        tmp_repo = tmp_dir / "repo_view"
        data_root = tmp_repo / "data" / "vidstg"
        official_output = tmp_dir / "outputs"
        doro_jsonl = tmp_dir / "manifest_as_doro.jsonl"

        _write_doro_jsonl(rows, doro_jsonl, num_clip_frames=int(num_clip_frames))
        _run([
            shutil.which("python") or "python",
            str(converter),
            "--annotation-path", str(doro_jsonl),
            "--output-dir", str(tmp_repo),
            "--fps", "2.0",
        ], cwd=project_root, env=os.environ.copy())

        videos_dir = data_root / "videos"
        for link_path in videos_dir.glob("*.mp4"):
            target = next((
                Path(str(row.get("video_path") or row.get("video_input_path"))).expanduser().resolve()
                for row in rows
                if Path(str(row.get("video_path") or row.get("video_input_path") or "")).name == link_path.name
            ), None)
            if target and target.exists():
                link_path.unlink(missing_ok=True)
                link_path.symlink_to(target)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(official_repo)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        official_python = os.getenv("TASTVG_PYTHON") or str(official_repo / ".venv" / "bin" / "python")
        _write_tastvg_label_files(data_root / "annos", rows)
        if prepare_cache or os.getenv("TASTVG_PREPARE_CACHE", "0").lower() in {"1", "true", "yes", "on"}:
            _prepare_tastvg_cache(official_repo, data_root, env, official_python, rows)

        cmd = [
            official_python,
            test_script,
            "--config-file", config_file,
            "MODEL.WEIGHT", str(Path(model_weight).expanduser().resolve()),
            "DATA_DIR", str(data_root),
            "OUTPUT_DIR", str(official_output),
            "DATALOADER.NUM_WORKERS", str(num_workers),
            "INPUT.RESOLUTION", str(input_resolution),
            "DATASET.NUM_CLIP_FRAMES", str(num_clip_frames),
        ]
        cmd.extend(official_extra_opts)
        _run(cmd, cwd=official_repo, env=env)

        test_results = official_output / result_file
        if not test_results.exists():
            raise FileNotFoundError(f"Official TA-STVG did not write predictions: {test_results}")
        normalized = _normalize_predictions(test_results, rows)

    with output_path.open("w", encoding="utf-8") as f:
        for item in normalized:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official TA-STVG inference for DORO-STVG manifest rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tastvg-dir", required=True)
    parser.add_argument("--model-weight", required=True)
    parser.add_argument("--input-resolution", default="224")
    parser.add_argument("--num-clip-frames", default="16")
    parser.add_argument("--num-workers", default="0")
    parser.add_argument("--config-file", default="experiments/vidstg.yaml")
    parser.add_argument("--test-script", default="scripts/test_net.py")
    parser.add_argument("--result-file", default="test_results.json")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--official-extra-opts", default="")
    args = parser.parse_args()

    run_tastvg_helper(
        manifest=args.manifest,
        output=args.output,
        tastvg_dir=args.tastvg_dir,
        model_weight=args.model_weight,
        input_resolution=args.input_resolution,
        num_clip_frames=args.num_clip_frames,
        num_workers=args.num_workers,
        config_file=args.config_file,
        test_script=args.test_script,
        result_file=args.result_file,
        prepare_cache=args.prepare_cache,
        official_extra_opts=shlex.split(args.official_extra_opts),
    )


if __name__ == "__main__":
    main()
