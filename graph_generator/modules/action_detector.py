from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope

from mmaction.apis import init_recognizer
from mmaction.evaluation import read_labelmap


@dataclass
class ActionResult:
    object_id: str
    frame_idx: int
    timestamp: float
    window: Tuple[int, int]
    box: List[float]
    actions: List[Tuple[str, float]]


class VideoMAEActionDetector:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        label_map_path: Optional[str] = None,
    ):
        self.config_path = config_path
        self.cfg = Config.fromfile(config_path)
        init_default_scope(self.cfg.get("default_scope", "mmaction"))
        self.model = init_recognizer(self.cfg, checkpoint_path, device=device)
        self.pipeline = Compose(self._build_pipeline(self.cfg))
        self.label_map, self.label_map_ids = self._load_label_map(label_map_path)
        self.clip_len, self.frame_interval = self._read_clip_params()
        self.num_classes = self._read_num_classes()

    def _build_pipeline(self, cfg: Config) -> List[dict]:
        pipeline_cfg = cfg.get("val_pipeline", None) or cfg.get("test_pipeline")
        if pipeline_cfg is None:
            raise ValueError("Config must define val_pipeline or test_pipeline.")
        pipeline_cfg = [step.copy() for step in pipeline_cfg]

        has_decord = any(step.get("type") == "DecordInit" for step in pipeline_cfg)
        if not has_decord:
            pipeline_cfg.insert(0, dict(type="DecordInit", io_backend="disk"))

        for step in pipeline_cfg:
            if step.get("type") == "RawFrameDecode":
                step["type"] = "DecordDecode"

        return pipeline_cfg

    def _load_label_map(self, label_map_path: Optional[str]) -> Tuple[Dict[int, str], List[int]]:
        if label_map_path is None:
            label_map_path = self.cfg.get("label_file")
        if label_map_path is None:
            default_path = Path(__file__).resolve().parents[1] / "mmaction2/tools/data/ava/label_map.txt"
            if default_path.exists():
                label_map_path = str(default_path)
        if label_map_path is not None:
            label_map_path = str(label_map_path)
            path_obj = Path(label_map_path)
            if not path_obj.is_absolute():
                config_dir = Path(self.config_path).resolve().parent
                candidate = (config_dir / path_obj).resolve()
                if candidate.exists():
                    label_map_path = str(candidate)
        if label_map_path is None:
            return {}, []
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map, _ = read_labelmap(f)
            if not label_map:
                f.seek(0)
                label_map = []
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    raw_id, name = line.split(":", 1)
                    label_map.append({"id": int(raw_id.strip()), "name": name.strip()})
        label_map_dict = {item["id"]: item["name"] for item in label_map}
        label_ids = [item["id"] for item in label_map]
        return label_map_dict, label_ids

    def _read_clip_params(self) -> Tuple[int, int]:
        for step in self.cfg.get("val_pipeline", []):
            if step.get("type") == "SampleAVAFrames":
                return step.get("clip_len", 16), step.get("frame_interval", 4)
        return 16, 4

    def _read_num_classes(self) -> int:
        bbox_head = self.cfg.get("model", {}).get("roi_head", {}).get("bbox_head", {})
        return int(bbox_head.get("num_classes", 1))

    def _temporal_window(self, frame_idx: int) -> Tuple[int, int]:
        half = (self.clip_len // 2) * self.frame_interval
        start = max(0, frame_idx - half)
        end = frame_idx + half
        return start, end

    @staticmethod
    def _get_video_info(video_path: str) -> Tuple[int, int, int, float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return width, height, total_frames, fps

    @staticmethod
    def _normalize_boxes(
        boxes: List[List[float]],
        width: int,
        height: int,
    ) -> np.ndarray:
        norm_boxes = []
        for x1, y1, x2, y2 in boxes:
            norm_boxes.append([
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ])
        return np.array(norm_boxes, dtype=np.float32)

    def _decode_actions(
        self,
        scores: np.ndarray,
        score_thr: float,
        topk: int,
    ) -> List[Tuple[str, float]]:
        if scores.size == 0:
            return []
        indices = np.argsort(scores)[::-1]
        results: List[Tuple[str, float]] = []
        for idx in indices[:topk]:
            score = float(scores[idx])
            if score < score_thr:
                continue
            if self.label_map_ids:
                label_id = self.label_map_ids[idx] if idx < len(self.label_map_ids) else int(idx + 1)
                label = self.label_map.get(int(label_id), str(idx))
            else:
                label = self.label_map.get(int(idx + 1), str(idx))
            results.append((label, score))
        return results

    def infer(
        self,
        video_path: str,
        boxes_by_frame: Dict[int, List[Tuple[str, List[float]]]],
        fps: Optional[float] = None,
        score_thr: float = 0.2,
        topk: int = 3,
    ) -> List[ActionResult]:
        video_id = Path(video_path).stem
        width, height, total_frames, video_fps = self._get_video_info(video_path)
        fps = fps or video_fps

        results: List[ActionResult] = []
        for frame_idx, items in sorted(boxes_by_frame.items()):
            if not items:
                continue
            timestamp = frame_idx / fps
            object_ids = [item[0] for item in items]
            boxes = [item[1] for item in items]
            proposals = self._normalize_boxes(boxes, width, height)
            gt_labels = np.zeros((len(boxes), self.num_classes), dtype=np.float32)
            sample_fps = 1
            sample_timestamp = frame_idx
            data = dict(
                filename=video_path,
                video_id=video_id,
                img_key=f"{video_id},{frame_idx:04d}",
                timestamp=sample_timestamp,
                timestamp_start=0,
                timestamp_end=total_frames,
                fps=sample_fps,
                start_index=0,
                total_frames=total_frames,
                proposals=proposals,
                gt_bboxes=proposals,
                gt_labels=gt_labels,
                modality="RGB",
            )
            data = self.pipeline(data)
            batch = pseudo_collate([data])
            with torch.no_grad():
                pred = self.model.test_step(batch)[0]
            pred_instances = pred.pred_instances
            scores = pred_instances.scores.detach().cpu().numpy()

            window = self._temporal_window(frame_idx)
            for object_id, box, score_vec in zip(object_ids, boxes, scores):
                actions = self._decode_actions(score_vec, score_thr, topk)
                results.append(
                    ActionResult(
                        object_id=object_id,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        window=window,
                        box=box,
                        actions=actions,
                    )
                )

        return results


def add_actions_to_graph(
    graph: Dict,
    video_path: str,
    detector: VideoMAEActionDetector,
    fps: Optional[float] = None,
    frame_interval: int = 15,
    score_thr: float = 0.2,
    topk: int = 3,
) -> Dict:
    boxes_by_frame: Dict[int, List[Tuple[str, List[float]]]] = {}
    for obj_node in graph.get("object_nodes", []):
        node_id = obj_node["node_id"]
        bboxes = obj_node.get("bboxes", {})
        for frame_key, box in bboxes.items():
            frame_idx = int(frame_key)
            if frame_interval > 1 and frame_idx % frame_interval != 0:
                continue
            boxes_by_frame.setdefault(frame_idx, []).append((node_id, box))

    results = detector.infer(
        video_path=video_path,
        boxes_by_frame=boxes_by_frame,
        fps=fps,
        score_thr=score_thr,
        topk=topk,
    )

    action_nodes = graph.get("action_nodes", [])
    action_index = len(action_nodes)

    for result in results:
        for label, score in result.actions:
            action_id = f"action_{action_index}"
            action_index += 1
            action_nodes.append({
                "node_id": action_id,
                "object_node_id": result.object_id,
                "action_label": label,
                "frame_idx": result.frame_idx,
                "confidence": score,
                "start_frame": result.window[0],
                "end_frame": result.window[1],
            })

    graph["action_nodes"] = action_nodes
    return graph


def _load_graph(jsonl_path: str, video_path: str) -> Dict:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("video_path") == video_path or Path(data.get("video_path", "")).stem == Path(video_path).stem:
                return data
    raise ValueError(f"No graph found for video {video_path}")


def main(
    config: str,
    checkpoint: str,
    jsonl: str,
    video: str,
    output: str,
    label_map: Optional[str] = None,
    fps: Optional[float] = None,
    frame_interval: int = 15,
    score_thr: float = 0.5,
    topk: int = 3,
) -> None:
    detector = VideoMAEActionDetector(
        config_path=config,
        checkpoint_path=checkpoint,
        label_map_path=label_map,
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
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(graph, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
