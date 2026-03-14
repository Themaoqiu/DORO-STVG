from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import sys

import cv2
import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

GRAPH_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
MMACTION2_ROOT = GRAPH_GENERATOR_ROOT / "dependence" / "mmaction2"
if str(MMACTION2_ROOT) not in sys.path:
    sys.path.insert(0, str(MMACTION2_ROOT))

from mmaction.evaluation import read_labelmap
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import get_str_type


HUMAN_CLASSES = {"person", "man", "woman", "boy", "girl", "people"}


@dataclass
class ActionResult:
    object_id: str
    frame_idx: int
    timestamp: float
    actions: List[Tuple[str, float]]


class VideoMAEActionDetector:
    """SpatioTemporal action detector aligned with MMAction2 demo flow.

    It uses pre-computed person boxes as proposals and runs action detection on
    clips sampled from the original video.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        label_map_path: Optional[str] = None,
        short_side: int = 256,
    ):
        self.config_path = config_path
        self.device = device
        self.short_side = int(short_side)

        self.cfg = Config.fromfile(config_path)
        init_default_scope(self.cfg.get("default_scope", "mmaction"))

        # Keep official demo behavior for STDET inference.
        try:
            self.cfg["model"]["test_cfg"]["rcnn"] = dict(action_thr=0)
        except Exception:
            pass
        try:
            self.cfg.model.backbone.pretrained = None
        except Exception:
            pass

        self.model = MODELS.build(self.cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()

        self.clip_len, self.model_frame_interval = self._read_clip_params()
        self.label_map = self._load_label_map(label_map_path)
        self.img_norm_cfg = self._read_img_norm_cfg()

    def _read_clip_params(self) -> Tuple[int, int]:
        val_pipeline = self.cfg.get("val_pipeline", None) or self.cfg.get("test_pipeline", [])
        for step in val_pipeline:
            if get_str_type(step.get("type")) == "SampleAVAFrames":
                return int(step.get("clip_len", 16)), int(step.get("frame_interval", 4))
        return 16, 4

    def _read_img_norm_cfg(self) -> Dict[str, object]:
        pre = self.cfg.get("model", {}).get("data_preprocessor", {})
        mean = np.array(pre.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
        std = np.array(pre.get("std", [58.395, 57.12, 57.375]), dtype=np.float32)
        to_rgb = bool(pre.get("to_rgb", False))
        return {"mean": mean, "std": std, "to_rgb": to_rgb}

    def _load_label_map(self, label_map_path: Optional[str]) -> Dict[int, str]:
        if label_map_path is None:
            raise ValueError("label_map_path is required for action decoding.")

        label_map_path = str(label_map_path)
        path_obj = Path(label_map_path)
        if not path_obj.is_absolute():
            config_dir = Path(self.config_path).resolve().parent
            candidate = (config_dir / path_obj).resolve()
            if candidate.exists():
                label_map_path = str(candidate)

        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map, _ = read_labelmap(f)
            if label_map:
                return {int(item["id"]): item["name"] for item in label_map}

            # Fallback for simple "id: label" text format.
            f.seek(0)
            mapping: Dict[int, str] = {}
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                raw_id, name = line.split(":", 1)
                mapping[int(raw_id.strip())] = name.strip()
            return mapping

    @staticmethod
    def _load_video(video_path: str) -> Tuple[List[np.ndarray], float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No frames decoded from video: {video_path}")
        return frames, float(fps)

    def _resize_frames(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], float, float]:
        h, w = frames[0].shape[:2]
        new_w, new_h = mmcv.rescale_size((w, h), (self.short_side, np.inf))
        resized = [mmcv.imresize(img, (new_w, new_h)) for img in frames]
        w_ratio = float(new_w) / float(w)
        h_ratio = float(new_h) / float(h)
        return resized, w_ratio, h_ratio

    def _clip_indices(self, center_idx: int, total_frames: int) -> np.ndarray:
        window_size = self.clip_len * self.model_frame_interval
        if self.clip_len % 2 == 0:
            start = center_idx - (self.clip_len // 2 - 1) * self.model_frame_interval
        else:
            start = center_idx - (self.clip_len // 2) * self.model_frame_interval

        frame_inds = start + np.arange(0, window_size, self.model_frame_interval)
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)
        return frame_inds.astype(np.int64)

    def _predict_scores(
        self,
        resized_frames: List[np.ndarray],
        center_idx: int,
        proposal_xyxy: np.ndarray,
    ) -> np.ndarray:
        frame_inds = self._clip_indices(center_idx, len(resized_frames))
        imgs = [resized_frames[i].astype(np.float32).copy() for i in frame_inds]
        _ = [mmcv.imnormalize_(img, **self.img_norm_cfg) for img in imgs]

        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(self.device)

        proposal_tensor = torch.from_numpy(proposal_xyxy.astype(np.float32)).to(self.device)
        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal_tensor)
        datasample.set_metainfo(dict(img_shape=resized_frames[0].shape[:2]))

        with torch.no_grad():
            result = self.model(input_tensor, [datasample], mode="predict")
        return result[0].pred_instances.scores.detach().cpu().numpy()

    def _decode_actions(
        self,
        score_vec: np.ndarray,
        score_thr: float,
        topk: int,
    ) -> List[Tuple[str, float]]:
        if score_vec.size == 0:
            return []

        ranked = np.argsort(score_vec)[::-1]
        out: List[Tuple[str, float]] = []
        for cls_idx in ranked:
            score = float(score_vec[cls_idx])
            if score < score_thr:
                break
            if int(cls_idx) == 0:
                continue  # background class

            label = self.label_map.get(int(cls_idx), str(int(cls_idx)))
            out.append((label, score))
            if topk > 0 and len(out) >= topk:
                break
        return out

    def infer(
        self,
        video_path: str,
        boxes_by_frame: Dict[int, List[Tuple[str, List[float]]]],
        fps: Optional[float] = None,
        score_thr: float = 0.2,
        topk: int = 3,
        debug_raw: bool = False,
    ) -> List[ActionResult]:
        frames, video_fps = self._load_video(video_path)
        resized_frames, w_ratio, h_ratio = self._resize_frames(frames)
        fps = fps or video_fps

        results: List[ActionResult] = []
        for frame_idx, items in sorted(boxes_by_frame.items()):
            if not items:
                continue
            if frame_idx < 0 or frame_idx >= len(frames):
                continue

            object_ids = [item[0] for item in items]
            boxes = [item[1] for item in items]
            proposal = np.array(boxes, dtype=np.float32)
            proposal[:, [0, 2]] *= w_ratio
            proposal[:, [1, 3]] *= h_ratio

            score_mat = self._predict_scores(
                resized_frames=resized_frames,
                center_idx=frame_idx,
                proposal_xyxy=proposal,
            )

            for obj_id, score_vec in zip(object_ids, score_mat):
                if debug_raw:
                    raw_idx = np.argsort(score_vec)[::-1][: max(3, topk)]
                    raw_items = [(int(i), float(score_vec[i])) for i in raw_idx]
                    print(f"[raw] frame={frame_idx} obj={obj_id} topk_idx_score={raw_items}")

                decoded = self._decode_actions(score_vec, score_thr=score_thr, topk=topk)
                if debug_raw:
                    print(f"[mapped] frame={frame_idx} obj={obj_id} actions={decoded}")

                results.append(
                    ActionResult(
                        object_id=obj_id,
                        frame_idx=frame_idx,
                        timestamp=float(frame_idx) / float(fps),
                        actions=decoded,
                    )
                )
        return results


def _is_human_object(obj_node: Dict) -> bool:
    cls = str(obj_node.get("object_class", "")).strip().lower()
    if cls in HUMAN_CLASSES:
        return True
    return "person" in cls


def _aggregate_action_segments(
    frame_results: List[ActionResult],
    max_gap: int = 1,
) -> List[Dict[str, object]]:
    # key: (object_id, action_label) -> [(frame_idx, score), ...]
    buckets: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    for item in frame_results:
        for action_label, score in item.actions:
            key = (item.object_id, action_label)
            buckets.setdefault(key, []).append((int(item.frame_idx), float(score)))

    segments: List[Dict[str, object]] = []
    for (obj_id, label), entries in buckets.items():
        entries.sort(key=lambda x: x[0])
        seg_start = entries[0][0]
        seg_end = entries[0][0]
        seg_scores = [entries[0][1]]
        seg_peak_score = entries[0][1]
        seg_peak_frame = entries[0][0]

        for frame_idx, score in entries[1:]:
            if frame_idx - seg_end <= max_gap:
                seg_end = frame_idx
                seg_scores.append(score)
                if score > seg_peak_score:
                    seg_peak_score = score
                    seg_peak_frame = frame_idx
                continue

            segments.append(
                {
                    "object_node_id": obj_id,
                    "action_label": label,
                    "start_frame": int(seg_start),
                    "end_frame": int(seg_end),
                    "frame_idx": int(seg_peak_frame),
                    "score": float(np.mean(seg_scores)),
                    "max_score": float(seg_peak_score),
                    "num_observations": int(len(seg_scores)),
                }
            )

            seg_start = frame_idx
            seg_end = frame_idx
            seg_scores = [score]
            seg_peak_score = score
            seg_peak_frame = frame_idx

        segments.append(
            {
                "object_node_id": obj_id,
                "action_label": label,
                "start_frame": int(seg_start),
                "end_frame": int(seg_end),
                "frame_idx": int(seg_peak_frame),
                "score": float(np.mean(seg_scores)),
                "max_score": float(seg_peak_score),
                "num_observations": int(len(seg_scores)),
            }
        )

    segments.sort(key=lambda x: (x["object_node_id"], x["start_frame"], x["action_label"]))
    return segments


def add_actions_to_graph(
    graph: Dict,
    video_path: str,
    detector: VideoMAEActionDetector,
    fps: Optional[float] = None,
    frame_interval: int = 1,
    score_thr: float = 0.2,
    topk: int = 3,
    debug_raw: bool = False,
) -> Dict:
    boxes_by_frame: Dict[int, List[Tuple[str, List[float]]]] = {}
    for obj_node in graph.get("object_nodes", []):
        if not _is_human_object(obj_node):
            continue

        node_id = obj_node.get("node_id")
        if not node_id:
            continue

        bboxes = obj_node.get("bboxes", {})
        for frame_key, box in bboxes.items():
            frame_idx = int(frame_key)
            if frame_interval > 1 and frame_idx % frame_interval != 0:
                continue
            boxes_by_frame.setdefault(frame_idx, []).append((node_id, box))

    frame_results = detector.infer(
        video_path=video_path,
        boxes_by_frame=boxes_by_frame,
        fps=fps,
        score_thr=score_thr,
        topk=topk,
        debug_raw=debug_raw,
    )

    segments = _aggregate_action_segments(
        frame_results=frame_results,
        max_gap=max(1, int(frame_interval)),
    )

    # Drop short-lived action segments that are likely noise.
    segments = [seg for seg in segments if int(seg.get("num_observations", 0)) >= 3]

    # Save actions grouped by person/object instead of one JSON record per action.
    grouped: Dict[str, List[Dict]] = {}
    for seg in segments:
        obj_id = seg["object_node_id"]
        grouped.setdefault(obj_id, []).append(
            {
                "action_label": seg["action_label"],
                "frame_idx": seg["frame_idx"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
            }
        )

    action_nodes: List[Dict] = []
    for idx, (obj_id, actions) in enumerate(sorted(grouped.items(), key=lambda x: x[0])):
        actions.sort(key=lambda x: (x["start_frame"], x["end_frame"], x["action_label"]))
        action_nodes.append(
            {
                "node_id": f"action_group_{idx}",
                "object_node_id": obj_id,
                "actions": actions,
            }
        )

    graph["action_nodes"] = action_nodes
    return graph


def _load_graph(jsonl_path: str, video_path: str) -> Dict:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("video_path") == video_path or Path(data.get("video_path", "")).stem == Path(video_path).stem:
                return data
    raise ValueError(f"No graph found for video {video_path}")


def _match_video(graph: Dict, video_path: str) -> bool:
    graph_video = graph.get("video_path", "")
    if graph_video == video_path:
        return True
    return Path(graph_video).stem == Path(video_path).stem


def _update_jsonl_inplace(jsonl_path: str, video_path: str, graph: Dict) -> None:
    input_path = Path(jsonl_path)
    temp_path = input_path.with_suffix(input_path.suffix + ".tmp")

    replaced = False
    with open(input_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            if _match_video(data, video_path):
                f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")
                replaced = True
            else:
                f_out.write(line)

    if not replaced:
        with open(temp_path, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")

    os.replace(temp_path, input_path)


def main(
    config: str,
    checkpoint: str,
    jsonl: str,
    video: str,
    output: Optional[str] = None,
    label_map: Optional[str] = None,
    fps: Optional[float] = None,
    frame_interval: int = 1,
    score_thr: float = 0.5,
    topk: int = 3,
    debug_raw: bool = False,
    device: str = "cuda:0",
    short_side: int = 256,
) -> None:
    detector = VideoMAEActionDetector(
        config_path=config,
        checkpoint_path=checkpoint,
        label_map_path=label_map,
        device=device,
        short_side=short_side,
    )
    graph = _load_graph(jsonl, video)
    graph = add_actions_to_graph(
        graph,
        video_path=video,
        detector=detector,
        fps=fps,
        frame_interval=frame_interval,
        score_thr=score_thr,
        topk=topk,
        debug_raw=debug_raw,
    )
    if output and output != jsonl:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(graph, ensure_ascii=False) + "\n")
    else:
        _update_jsonl_inplace(jsonl, video, graph)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
