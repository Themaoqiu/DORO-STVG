"""Persistent worker for DeViL inference.

Uses the bundled DeViL package at eval/dependence/devil/devil.
Reads JSON requests from stdin, writes responses framed with the
__DORO_DEVIL__ marker.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


PROTO = "__DORO_DEVIL__"


def _emit(msg):
    sys.stdout.write(PROTO + " " + json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _xyxy_from_cxcywh(box):
    cx, cy, w, h = [float(v) for v in box]
    x1 = max(0.0, min(1.0, cx - w / 2))
    y1 = max(0.0, min(1.0, cy - h / 2))
    x2 = max(0.0, min(1.0, cx + w / 2))
    y2 = max(0.0, min(1.0, cy + h / 2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _parse_time_range(text):
    import re
    s = (text or "").replace("–", "-")
    for pat in (
        r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)",
        r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
    ):
        m = re.search(pat, s, flags=re.IGNORECASE)
        if not m:
            continue
        start, end = float(m.group(1)), float(m.group(2))
        if end < start:
            start, end = end, start
        return start, end
    return None


def _run_video_request(model, processor, video_path, query, max_new_tokens, do_sample):
    import torch
    from dependence.devil import mm_infer, load_video_new

    frames, timestamps, _imags = load_video_new(video_path, make_pil=True)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": query},
            ],
        }
    ]
    inputs = processor(
        images=[frames],
        text=conversation,
        merge_size=2,
        return_tensors="pt",
    )
    inputs["images"] = [_imags]

    output_text = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=do_sample,
        modal="video",
        max_new_tokens=max_new_tokens,
    )

    boxes_per_frame = {}
    dino_output = getattr(model, "infer_dino_output", None)
    if isinstance(dino_output, dict) and "pred_logits" in dino_output and "pred_boxes" in dino_output:
        try:
            pred_logits = dino_output["pred_logits"][0]  # [T, Q, C]
            pred_boxes = dino_output["pred_boxes"][0]    # [T, Q, 4]
            if pred_logits.ndim == 3 and pred_boxes.ndim == 3 and pred_logits.shape[1] > 0:
                T = pred_logits.shape[0]
                time_range = _parse_time_range(output_text)
                if time_range is not None:
                    start_sec, end_sec = time_range
                    relevant = [i for i, ts in enumerate(timestamps) if start_sec <= ts <= end_sec and i < T]
                else:
                    relevant = list(range(T))
                if relevant:
                    with torch.no_grad():
                        all_scores = pred_logits.sigmoid()
                        max_scores, _ = all_scores.max(-1)
                        tube_scores = max_scores[relevant].mean(dim=0)
                        best_q = int(tube_scores.argmax().item())
                    for i in relevant:
                        box_cxcywh = pred_boxes[i][best_q].detach().cpu().tolist()
                        boxes_per_frame[str(i)] = _xyxy_from_cxcywh(box_cxcywh)
        except Exception:
            pass

    return {
        "text": str(output_text),
        "boxes": boxes_per_frame,
        "timestamps": list(timestamps),
        "time_range": _parse_time_range(output_text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":")[-1] if ":" in args.device else "0")

    try:
        from dependence.devil import disable_torch_init, model_init
    except Exception as exc:
        _emit({"type": "error", "stage": "import", "message": str(exc), "trace": traceback.format_exc()})
        return 1

    try:
        disable_torch_init()
        model, processor = model_init(args.model_path)
    except Exception as exc:
        _emit({"type": "error", "stage": "load", "message": str(exc), "trace": traceback.format_exc()})
        return 1

    _emit({"type": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"type": "error", "stage": "decode", "message": str(exc)})
            continue

        if payload.get("type") == "shutdown":
            _emit({"type": "bye"})
            return 0
        if payload.get("type") != "request":
            continue

        request_id = payload.get("id", "")
        try:
            result = _run_video_request(
                model=model,
                processor=processor,
                video_path=payload["video_path"],
                query=payload["query"],
                max_new_tokens=int(payload.get("max_new_tokens", 1024)),
                do_sample=bool(payload.get("do_sample", False)),
            )
        except Exception as exc:
            _emit({"type": "error", "id": request_id, "message": str(exc), "trace": traceback.format_exc()})
            continue

        _emit({"type": "response", "id": request_id, **result})

    return 0


if __name__ == "__main__":
    sys.exit(main())
