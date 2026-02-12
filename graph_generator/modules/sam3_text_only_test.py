import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from modules.scene_detector import SceneDetector
from sam3.sam3.model_builder import build_sam3_video_predictor
from sam3.sam3.model.sam3_tracker_utils import mask_to_box


@dataclass
class ObjectNode:
    node_id: str
    global_track_id: int
    object_class: str
    start_frame: int
    end_frame: int
    shot_ids: List[int]
    bboxes: Optional[Dict[int, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "node_id": self.node_id,
            "global_track_id": self.global_track_id,
            "object_class": self.object_class,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "shot_ids": self.shot_ids,
        }
        if self.bboxes:
            data["bboxes"] = self.bboxes
        return data


@dataclass
class SceneGraph:
    video: str
    video_path: str
    temporal_nodes: List[Dict] = field(default_factory=list)
    object_nodes: List[Dict] = field(default_factory=list)
    attribute_nodes: List[Dict] = field(default_factory=list)
    action_nodes: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video,
            "video_path": self.video_path,
            "temporal_nodes": self.temporal_nodes,
            "object_nodes": self.object_nodes,
            "attribute_nodes": self.attribute_nodes,
            "action_nodes": self.action_nodes,
            "edges": self.edges,
        }

    def save_to_jsonl(self, output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")


def _get_video_info(video_path: str) -> Dict[str, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": width, "height": height, "total_frames": total_frames}


def _track_text_prompt(
    predictor,
    video_path: str,
    clip: Dict[str, Any],
    text_prompt: str,
) -> Dict[int, Dict[str, Any]]:
    session_response = predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = session_response["session_id"]

    start_frame = clip["start_frame"]
    end_frame = clip["end_frame"]

    predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )

    tracked_frames: Dict[int, Dict[str, Any]] = {}
    propagate_request = dict(
        type="propagate_in_video",
        session_id=session_id,
        propagation_direction="forward",
        start_frame_index=start_frame,
    )

    for response in predictor.handle_stream_request(propagate_request):
        frame_idx = response.get("frame_index")
        if frame_idx is None or frame_idx < start_frame or frame_idx > end_frame:
            continue
        outputs = response.get("outputs", {})
        out_masks = outputs.get("out_binary_masks", None)
        if out_masks is None or len(out_masks) == 0:
            continue
        mask = np.array(out_masks[0], dtype=np.uint8)
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        bbox_tensor = mask_to_box(mask_tensor)
        bbox = bbox_tensor[0, 0].tolist()
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        tracked_frames[frame_idx] = {"box": bbox}

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return tracked_frames


def run(
    video: str,
    output: str = "output/scene_graphs_sam3_text_only.jsonl",
    sam3_model: str = "sam3.pt",
    text_prompt: str = "cat",
    scene_threshold: float = 3.0,
    min_scene_duration: float = 1.0,
) -> None:
    video_path = Path(video)
    video_name = video_path.stem
    graph = SceneGraph(video=video_name, video_path=str(video_path))

    scene_detector = SceneDetector(
        str(video_path),
        threshold=scene_threshold,
        min_scene_duration=min_scene_duration,
    )
    clips = scene_detector.detect()
    graph.temporal_nodes = [clip.to_dict() for clip in clips]

    predictor = build_sam3_video_predictor(checkpoint_path=sam3_model)
    video_info = _get_video_info(str(video_path))
    print(f"Video info: {video_info['width']}x{video_info['height']} frames={video_info['total_frames']}")

    global_track_id = 0
    for clip in graph.temporal_nodes:
        tracked_frames = _track_text_prompt(
            predictor=predictor,
            video_path=str(video_path),
            clip=clip,
            text_prompt=text_prompt,
        )
        if not tracked_frames:
            continue

        bboxes = {frame_idx: data["box"] for frame_idx, data in tracked_frames.items()}
        start_frame = min(bboxes.keys())
        end_frame = max(bboxes.keys())

        obj_node = ObjectNode(
            node_id=f"obj_{text_prompt}_{global_track_id}",
            global_track_id=global_track_id,
            object_class=text_prompt,
            start_frame=start_frame,
            end_frame=end_frame,
            shot_ids=[clip["clip_id"]],
            bboxes=bboxes,
        )
        graph.object_nodes.append(obj_node.to_dict())
        global_track_id += 1

    graph.save_to_jsonl(output)
    print(f"Saved graph to {output}")


if __name__ == "__main__":
    import fire

    fire.Fire(run)
