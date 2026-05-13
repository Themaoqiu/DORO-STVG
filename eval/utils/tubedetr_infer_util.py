import json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import ffmpeg
import numpy as np
import torch


def _parse_video_input_spec(video_input: str) -> Tuple[str, int]:
    marker = "::split="
    if marker not in video_input:
        return video_input, 0
    video_path, split_spec = video_input.split(marker, 1)
    start_text, _ = split_spec.split(":", 1)
    return video_path, int(start_text)


def _build_tubedetr_args(
    repo_dir: Path,
    checkpoint: str,
    dataset_config: str,
    combine_datasets: List[str],
    combine_datasets_val: List[str],
    resolution: int,
    fps: int,
    device: str,
    output_dir: str,
):
    sys.path.insert(0, str(repo_dir))
    from main import get_args_parser
    parser = get_args_parser()
    cli = [
        "--dataset_config", dataset_config,
        "--combine_datasets", *combine_datasets,
        "--combine_datasets_val", *combine_datasets_val,
        "--load", checkpoint,
        "--device", device,
        "--resolution", str(resolution),
        "--fps", str(fps),
        "--output-dir", output_dir,
    ]
    return parser.parse_args(cli)


def _load_model_and_postprocessors(args):
    from models import build_model
    from models.postprocessors import PostProcess, PostProcessSTVG
    model, _, _ = build_model(args)
    model.to(args.device)
    model.eval()
    checkpoint = torch.load(args.load, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_ema") or checkpoint.get("model") or checkpoint
    if args.num_queries < 100 and "query_embed.weight" in state_dict:
        state_dict["query_embed.weight"] = state_dict["query_embed.weight"][:args.num_queries]
    if "transformer.time_embed.te" in state_dict:
        del state_dict["transformer.time_embed.te"]
    model.load_state_dict(state_dict, strict=False)
    return model, {"vidstg": PostProcessSTVG(), "bbox": PostProcess()}


def _predict_one(model, postprocessors, args, query: str, video_input_path: str) -> str:
    from datasets.video_transforms import make_video_transforms, prepare
    from util.misc import NestedTensor

    vid_path, split_start = _parse_video_input_spec(video_input_path)
    probe = ffmpeg.probe(vid_path)
    video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if video_stream is None:
        return "{}"
    clip_start = float(video_stream.get("start_time", 0.0))
    clip_duration = float(video_stream["duration"])
    extracted_fps = min((args.fps * clip_duration), args.video_max_len) / clip_duration
    out, _ = ffmpeg.input(vid_path, ss=clip_start, t=clip_duration).filter("fps", fps=extracted_fps).output("pipe:", format="rawvideo", pix_fmt="rgb24").run(capture_stdout=True, quiet=True)

    width, height = int(video_stream["width"]), int(video_stream["height"])
    images_list = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    if len(images_list) == 0:
        return "{}"

    image_ids = [[i for i in range(len(images_list))]]
    placeholder = prepare(width, height, [])
    transforms = make_video_transforms("test", cautious=True, resolution=args.resolution)
    images, targets = transforms(images_list, [placeholder] * len(images_list))

    samples = NestedTensor.from_tensor_list([images], False)
    if args.stride:
        samples_fast = samples.to(args.device)
        samples = NestedTensor.from_tensor_list([images[:, ::args.stride]], False).to(args.device)
    else:
        samples_fast = None

    with torch.no_grad():
        memory_cache = model(samples, [len(targets)], [query], encode_and_save=True, samples_fast=samples_fast)
        outputs = model(samples, [len(targets)], [query], encode_and_save=False, memory_cache=memory_cache)
        pred_steds = postprocessors["vidstg"](outputs, image_ids, video_ids=[0])[0]
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(args.device)
        results = postprocessors["bbox"](outputs, orig_sizes)

    frame_map: Dict[str, List[float]] = {}
    start_idx, end_idx = int(pred_steds[0]), int(pred_steds[1])
    for img_id, result in zip(image_ids[0], results):
        if not (start_idx <= img_id < end_idx):
            continue
        box = result["boxes"].detach().cpu().tolist()
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = box
        elif isinstance(box, list) and box and isinstance(box[0], list) and len(box[0]) == 4:
            x1, y1, x2, y2 = box[0]
        else:
            continue
        frame_map[str(split_start + int(img_id))] = [
            max(0.0, min(1.0, float(x1) / width)),
            max(0.0, min(1.0, float(y1) / height)),
            max(0.0, min(1.0, float(x2) / width)),
            max(0.0, min(1.0, float(y2) / height)),
        ]
    return json.dumps({"target": frame_map}, ensure_ascii=False) if frame_map else "{}"


def main(
    manifest: str,
    output: str,
    tubedetr_dir: str,
    checkpoint: str,
    dataset_config: str = "config/vidstg.json",
    combine_datasets: List[str] | None = None,
    combine_datasets_val: List[str] | None = None,
    resolution: int = 224,
    fps: int = 5,
    device: str = "cuda",
    tmp_output_dir: str = "",
) -> None:
    repo_dir = Path(tubedetr_dir).expanduser().resolve()
    os.chdir(repo_dir)
    tube_args = _build_tubedetr_args(
        repo_dir,
        checkpoint,
        dataset_config,
        combine_datasets or ["vidstg"],
        combine_datasets_val or ["vidstg"],
        resolution,
        fps,
        device,
        tmp_output_dir or str(repo_dir / "_util_outputs"),
    )
    model, postprocessors = _load_model_and_postprocessors(tube_args)

    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            query = str(row.get("query") or "")
            video_input_path = str(row.get("video_input_path") or row.get("video_path") or "")
            raw_response = _predict_one(model, postprocessors, tube_args, query, video_input_path)
            f.write(json.dumps({"query": query, "video_path": video_input_path, "raw_response": raw_response}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
