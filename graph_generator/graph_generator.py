import json
import os
import subprocess
import sys
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import fire
from modules.scene_detector import SceneDetector
from modules.yolo_detector import YOLOKeyframeDetector
from modules.graph_filter import GraphFilter


def _video_record_path(video_path: str) -> str:
    return Path(video_path).name


@dataclass
class ObjectNode:
    node_id: str
    global_track_id: int
    object_class: str
    start_frame: int
    end_frame: int
    shot_ids: List[int]
    bboxes: Optional[Dict[int, List[float]]] = None
    is_dynamic: Optional[bool] = None
    appearance: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'node_id': self.node_id,
            'global_track_id': self.global_track_id,
            'object_class': self.object_class,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'shot_ids': self.shot_ids,
        }
        if self.bboxes:
            data['bboxes'] = self.bboxes
        if self.is_dynamic is not None:
            data['is_dynamic'] = self.is_dynamic
        if self.appearance:
            data['appearance'] = self.appearance
        return data


@dataclass
class AttributeNode:
    node_id: str
    object_node_id: str
    attribute_type: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'object_node_id': self.object_node_id,
            'attribute_type': self.attribute_type,
            'description': self.description,
        }


@dataclass
class ActionNode:
    node_id: str
    object_node_id: str
    action_label: str
    frame_idx: int
    confidence: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'node_id': self.node_id,
            'object_node_id': self.object_node_id,
            'action_label': self.action_label,
            'frame_idx': self.frame_idx,
            'confidence': self.confidence,
        }
        if self.start_frame is not None:
            data['start_frame'] = self.start_frame
        if self.end_frame is not None:
            data['end_frame'] = self.end_frame
        return data


@dataclass
class Edge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    relation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_id': self.edge_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'relation': self.relation,
        }


@dataclass
class SceneGraph:
    video: str
    video_path: str
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    temporal_nodes: List[Dict] = field(default_factory=list)
    object_nodes: List[Dict] = field(default_factory=list)
    attribute_nodes: List[Dict] = field(default_factory=list)
    action_nodes: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'video': self.video,
            'video_path': self.video_path,
            'video_width': self.video_width,
            'video_height': self.video_height,
            'temporal_nodes': self.temporal_nodes,
            'object_nodes': self.object_nodes,
            'attribute_nodes': self.attribute_nodes,
            'action_nodes': self.action_nodes,
            'edges': self.edges,
        }
    
    def save_to_jsonl(self, output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + '\n')


class SceneGraphGenerator:
    def __init__(
        self,
        yolo_model: str,
        tracker_backend: str = "sam3",
        scene_threshold: float = 3.0,
        min_scene_duration: float = 1.0,
        conf: float = 0.25,
        iou: float = 0.5,
        sam3_model: str = "sam3.pt",
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_checkpoint: Optional[str] = None,
        sam3_redetection_interval: int = 15,
        sam3_iou_threshold: float = 0.4,
        sam3_overlap_threshold: float = 0.6,
        groundedsam2_mask_output_dir: Optional[str] = None,
        sam3_mask_output_dir: Optional[str] = None,
        sam3_match_output_dir: Optional[str] = None,
        sam3_match_log_path: Optional[str] = None,
        use_action_detection: bool = False,
        action_config: Optional[str] = None,
        action_checkpoint: Optional[str] = None,
        action_label_map: Optional[str] = None,
        action_score_thr: float = 0.2,
        action_topk: int = 3,
        action_frame_interval: int = 1,
        skip_filter: bool = False,
        filter_min_frames: int = 5,
    ):
        self.yolo_model = yolo_model
        self.tracker_backend = tracker_backend
        self.scene_threshold = scene_threshold
        self.min_scene_duration = min_scene_duration
        self.conf = conf
        self.iou = iou
        self.sam3_model = sam3_model
        self.sam2_model_cfg = sam2_model_cfg
        self.sam2_checkpoint = sam2_checkpoint
        self.sam3_redetection_interval = sam3_redetection_interval
        self.sam3_iou_threshold = sam3_iou_threshold
        self.sam3_overlap_threshold = sam3_overlap_threshold
        self.groundedsam2_mask_output_dir = groundedsam2_mask_output_dir
        self.sam3_mask_output_dir = sam3_mask_output_dir
        self.sam3_match_output_dir = sam3_match_output_dir
        self.sam3_match_log_path = sam3_match_log_path
        self.use_action_detection = use_action_detection
        self.action_config = action_config
        self.action_checkpoint = action_checkpoint
        self.action_label_map = action_label_map
        self.action_score_thr = action_score_thr
        self.action_topk = action_topk
        self.action_frame_interval = action_frame_interval
        self.skip_filter = skip_filter
        self.filter_min_frames = filter_min_frames
        self._graph_filter: Optional[GraphFilter] = None
        self._keyframe_detector: Optional[YOLOKeyframeDetector] = None
        self._tracker: Optional[Any] = None
        self._action_detector: Optional[Any] = None

    def _init_runtime(self) -> None:
        if self._graph_filter is None:
            self._graph_filter = GraphFilter(min_frames=self.filter_min_frames)

        if self._keyframe_detector is None:
            self._keyframe_detector = YOLOKeyframeDetector(
                model_path=self.yolo_model,
                conf=self.conf,
                iou=self.iou,
                keyframe_interval=self.sam3_redetection_interval,
            )

        if self._tracker is None:
            if self.tracker_backend == "groundedsam2":
                from modules.groundingedsam2_tracker import GroundedSAM2Tracker

                self._tracker = GroundedSAM2Tracker(
                    sam2_model_cfg=self.sam2_model_cfg,
                    sam2_checkpoint=self.sam2_checkpoint,
                    iou_threshold=self.sam3_iou_threshold,
                    overlap_threshold=self.sam3_overlap_threshold,
                    redetection_interval=self.sam3_redetection_interval,
                    mask_output_dir=self.groundedsam2_mask_output_dir,
                )
            elif self.tracker_backend == "sam3":
                from modules.sam3_tracker import SAM3Tracker

                self._tracker = SAM3Tracker(
                    model_path=self.sam3_model,
                    iou_threshold=self.sam3_iou_threshold,
                    overlap_threshold=self.sam3_overlap_threshold,
                    redetection_interval=self.sam3_redetection_interval,
                    mask_output_dir=self.sam3_mask_output_dir,
                    match_output_dir=self.sam3_match_output_dir,
                    match_log_path=self.sam3_match_log_path,
                )
            elif self.tracker_backend != "yolo":
                raise ValueError(
                    "Invalid tracker_backend. Expected one of: sam3, groundedsam2, yolo"
                )

        if self.use_action_detection and self._action_detector is None:
            from modules.action_detector import VideoMAEActionDetector

            if not self.action_config or not self.action_checkpoint:
                raise ValueError("Action detection requires action_config and action_checkpoint.")

            self._action_detector = VideoMAEActionDetector(
                config_path=self.action_config,
                checkpoint_path=self.action_checkpoint,
                label_map_path=self.action_label_map,
            )
    
    def process_video(self, video_path: str, output_path: str) -> SceneGraph:
        self._init_runtime()
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        graph = SceneGraph(
            video=video_name,
            video_path=_video_record_path(video_path),
            video_width=video_width if video_width > 0 else None,
            video_height=video_height if video_height > 0 else None,
        )

        print("[scene_graph] detect scenes")
        scene_detector = SceneDetector(
            video_path,
            threshold=self.scene_threshold,
            min_scene_duration=self.min_scene_duration,
        )
        clips = scene_detector.detect()

        graph.temporal_nodes = [clip.to_dict() for clip in clips]
        print(f"[scene_graph] scenes: {len(clips)}")

        print("[scene_graph] detect keyframes")
        keyframe_detector = self._keyframe_detector
        all_detections = keyframe_detector.detect_keyframes(video_path, clips)

        if self.tracker_backend == "groundedsam2":
            print("[scene_graph] track objects: Grounded-SAM2")
            tracker = self._tracker
            global_tracks = tracker.track_video(video_path, clips, all_detections)
            print(f"[scene_graph] tracks: {len(global_tracks)}")
        elif self.tracker_backend == "sam3":
            print("[scene_graph] track objects: SAM3")
            tracker = self._tracker
            global_tracks = tracker.track_video(video_path, clips, all_detections)
            print(f"[scene_graph] tracks: {len(global_tracks)}")
        elif self.tracker_backend == "yolo":
            print("[scene_graph] track objects: YOLO detections only")
            global_tracks = keyframe_detector.detections_to_tracks(all_detections)
            print(f"[scene_graph] detections: {len(global_tracks)}")
        else:
            raise ValueError(
                "Invalid tracker_backend. Expected one of: sam3, groundedsam2, yolo"
            )

        print("[scene_graph] build object nodes")
        for g_track in global_tracks:
            bboxes = {}
            for local_track in g_track.local_tracks:
                for frame_idx, frame_data in local_track.frames.items():
                    bboxes[frame_idx] = frame_data['box']

            obj_node = ObjectNode(
                node_id=f"{g_track.object_class}_{g_track.global_id}",
                global_track_id=g_track.global_id,
                object_class=g_track.object_class,
                start_frame=g_track.start_frame,
                end_frame=g_track.end_frame,
                shot_ids=list(set(t.clip_id for t in g_track.local_tracks)),
                bboxes=bboxes,
            )
            graph.object_nodes.append(obj_node.to_dict())

        print(f"[scene_graph] object nodes: {len(graph.object_nodes)}")
        graph_filter = self._graph_filter

        if self.skip_filter:
            print("[scene_graph] skip graph filtering")
            normalized_graph = graph_filter.normalize_graph(graph.to_dict(), filter_objects=False)
            graph.object_nodes = normalized_graph['object_nodes']
            graph.edges = normalized_graph['edges']
            print(f"[scene_graph] final: {len(graph.object_nodes)} objects, {len(graph.edges)} edges")
        else:
            print("[scene_graph] filter graph")

            graph_dict = graph.to_dict()
            filtered_graph_dict = graph_filter.filter_graph(graph_dict)

            filtered_objects = len(filtered_graph_dict['object_nodes'])
            removed_objects = len(graph.object_nodes) - filtered_objects
            print(f"[scene_graph] kept {filtered_objects}, removed {removed_objects}")

            graph.object_nodes = filtered_graph_dict['object_nodes']
            graph.edges = filtered_graph_dict['edges']

            print(f"[scene_graph] final: {len(graph.object_nodes)} objects, {len(graph.edges)} edges")

        if self.use_action_detection:
            from modules.action_detector import add_actions_to_graph

            graph_dict = add_actions_to_graph(
                graph.to_dict(),
                video_path,
                detector=self._action_detector,
                fps=scene_detector._fps,
                frame_interval=self.action_frame_interval,
                score_thr=self.action_score_thr,
                topk=self.action_topk,
            )
            graph_dict = graph_filter.normalize_graph(graph_dict, filter_objects=False)
            graph.action_nodes = graph_dict['action_nodes']
            graph.edges = graph_dict['edges']

            print(f"[scene_graph] action nodes: {len(graph.action_nodes)}")

        graph.save_to_jsonl(output_path)
        print(f"[scene_graph] saved: {output_path}")

        return graph
    
    def process_videos(self, video_dir: str, output_path: str) -> List[SceneGraph]:
        video_dir = Path(video_dir)
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        existing_videos = _load_existing_videos(output_path)
        
        graphs = []
        for i, video_path in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] Processing {video_path.name}")
            try:
                if str(video_path) in existing_videos or video_path.stem in existing_videos:
                    print(f"  Skip existing: {video_path.name}")
                    continue
                graph = self.process_video(str(video_path), output_path)
                graphs.append(graph)
                existing_videos.add(str(video_path))
                existing_videos.add(video_path.stem)
            except Exception as e:
                print(f"  Error: {e}")
        
        return graphs



def _format_fire_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _load_existing_videos(output_path: str) -> set[str]:
    output_file = Path(output_path)
    if not output_file.exists():
        return set()

    existing: set[str] = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            video_path = str(data.get("video_path", "")).strip()
            if video_path:
                existing.add(video_path)
                existing.add(Path(video_path).stem)

            video_name = str(data.get("video", "")).strip()
            if video_name:
                existing.add(video_name)

    return existing


def _run_fire_module(
    module: str,
    args: Dict[str, Any],
    *,
    python_exec: Optional[str] = None,
    cwd: Optional[Path] = None,
    env_updates: Optional[Dict[str, str]] = None,
) -> None:
    command = [python_exec or sys.executable, "-m", module]
    for key, value in args.items():
        if value is None:
            continue
        command.append(f"--{key}={_format_fire_value(value)}")

    env = os.environ.copy()
    if env_updates:
        env.update({k: str(v) for k, v in env_updates.items() if v is not None})

    print(f"[pipeline] -> {module.split('.')[-1]}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def _run_full_pipeline_for_video(
    *,
    video_path: str,
    output_jsonl: str,
    project_root: Path,
    pipeline_python: Optional[str],
    cuda_visible_devices: Optional[str],
    hf_endpoint: Optional[str],
    with_attribute: bool,
    attribute_model_name: str,
    attribute_masks_json: Optional[str],
    attribute_model_path: str,
    attribute_max_frames: int,
    with_action: bool,
    action_config: Optional[str],
    action_checkpoint: Optional[str],
    action_label_map: Optional[str],
    action_frame_interval: int,
    action_python: Optional[str],
    action_device: str,
    with_relation: bool,
    relation_model_name: str,
    relation_crop_output_dir: str,
    relation_min_shared_frames: int,
    relation_save_intermediate_frames: bool,
    relation_verbose: bool,
    with_reference: bool,
    reference_model_name: str,
    reference_crop_output_dir: str,
    reference_frames_per_shot: int,
    reference_save_intermediate_frames: bool,
) -> None:
    common_env: Dict[str, str] = {}
    if cuda_visible_devices is not None:
        common_env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if hf_endpoint is not None:
        common_env["HF_ENDPOINT"] = str(hf_endpoint)

    if with_attribute:
        masks_json = attribute_masks_json
        cleanup_attribute_masks = False
        if not masks_json:
            default_mask_dir = project_root / "output" / "sam2_masks"
            masks_json = str(default_mask_dir / f"{Path(video_path).stem}_sam2_masks_indexed.json")
            cleanup_attribute_masks = True
        _run_fire_module(
            "modules.attribute_generator",
            {
                "jsonl": output_jsonl,
                "video": video_path,
                "model_name": attribute_model_name,
                "masks_json": masks_json,
                "model_path": attribute_model_path,
                "max_frames": attribute_max_frames,
            },
            python_exec=pipeline_python,
            cwd=project_root,
            env_updates=common_env,
        )
        if cleanup_attribute_masks and masks_json:
            mask_path = Path(masks_json)
            if mask_path.exists():
                mask_path.unlink()
                print(f"[pipeline] cleaned temporary SAM2 masks: {mask_path}")

    if with_action:
        if not action_config or not action_checkpoint or not action_label_map:
            raise ValueError(
                "with_action=True requires action_config, action_checkpoint, and action_label_map."
            )
        action_env = dict(common_env)
        mmaction2_path = str(project_root / "dependence" / "mmaction2")
        action_env["PYTHONPATH"] = (
            f"{mmaction2_path}:{action_env.get('PYTHONPATH') or os.environ.get('PYTHONPATH', '')}"
        ).rstrip(":")
        _run_fire_module(
            "modules.action_detector",
            {
                "config": action_config,
                "checkpoint": action_checkpoint,
                "label_map": action_label_map,
                "frame_interval": action_frame_interval,
                "jsonl": output_jsonl,
                "video": video_path,
                "device": action_device,
            },
            python_exec=action_python or pipeline_python,
            cwd=project_root,
            env_updates=action_env,
        )

    if with_relation:
        _run_fire_module(
            "modules.relation_generator",
            {
                "jsonl": output_jsonl,
                "video": video_path,
                "model_name": relation_model_name,
                "crop_output_dir": relation_crop_output_dir,
                "min_shared_frames": relation_min_shared_frames,
                "save_intermediate_frames": relation_save_intermediate_frames,
                "verbose": relation_verbose,
            },
            python_exec=pipeline_python,
            cwd=project_root,
            env_updates=common_env,
        )

    if with_reference:
        graph_data: Optional[Dict[str, Any]] = None
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                graph_video = data.get("video_path", "")
                if graph_video == video_path or Path(graph_video).stem == Path(video_path).stem:
                    graph_data = data
                    break

        if graph_data is None:
            raise ValueError(f"Failed to locate graph for video in {output_jsonl}: {video_path}")

        shot_ids = sorted(
            {
                int(node["clip_id"])
                for node in graph_data.get("temporal_nodes", [])
                if "clip_id" in node
            }
        )
        if len(shot_ids) < 2:
            print(f"[pipeline][reference] skip single-shot video: {Path(video_path).name}")
        else:
            print(f"[pipeline][reference] {len(shot_ids)} shots -> {len(shot_ids) * (len(shot_ids) - 1) // 2} pairs")
            for shot_a, shot_b in combinations(shot_ids, 2):
                crop_output_dir = reference_crop_output_dir
                if reference_save_intermediate_frames:
                    crop_output_dir = str(Path(reference_crop_output_dir) / f"shot_{shot_a}_{shot_b}")
                print(f"[pipeline][reference] pair {shot_a} <-> {shot_b}")
                _run_fire_module(
                    "modules.reference_edge_generator",
                    {
                        "jsonl": output_jsonl,
                        "video": video_path,
                        "model_name": reference_model_name,
                        "crop_output_dir": crop_output_dir,
                        "save_intermediate_frames": reference_save_intermediate_frames,
                        "shot_a": shot_a,
                        "shot_b": shot_b,
                        "frames_per_shot": reference_frames_per_shot,
                    },
                    python_exec=pipeline_python,
                    cwd=project_root,
                    env_updates=common_env,
                )

def run(
    video: str = None,
    video_dir: str = None,
    max_videos: int = 0,
    output: str = "output/scene_graphs.jsonl",
    yolo_model: str = "yolo26x.pt",
    tracker_backend: str = "sam3",
    scene_threshold: float = 3.0,
    min_scene_duration: float = 1.0,
    conf: float = 0.25,
    iou: float = 0.5,
    sam3_model: str = "sam3.pt",
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_checkpoint: str = None,
    sam3_redetection_interval: int = 15,
    sam3_iou_threshold: float = 0.4,
    sam3_overlap_threshold: float = 0.6,
    groundedsam2_mask_output_dir: str = None,
    sam3_mask_output_dir: str = None,
    sam3_match_output_dir: str = None,
    sam3_match_log_path: str = None,
    use_action_detection: bool = False,
    action_config: str = None,
    action_checkpoint: str = None,
    action_label_map: str = None,
    action_score_thr: float = 0.2,
    action_topk: int = 3,
    action_frame_interval: int = 15,
    skip_filter: bool = False,
    filter_min_frames: int = 5,
    full_pipeline: bool = False,
    pipeline_python: str = None,
    cuda_visible_devices: str = None,
    hf_endpoint: str = None,
    with_attribute: bool = True,
    attribute_model_name: str = "gemini-3-flash-preview",
    attribute_masks_json: str = None,
    attribute_model_path: str = "nvidia/DAM-3B-Video",
    attribute_max_frames: int = 8,
    with_action: bool = True,
    action_python: str = None,
    action_device: str = "cuda:0",
    with_relation: bool = True,
    relation_model_name: str = "gemini-3-flash-preview",
    relation_crop_output_dir: str = "output/relation_crops",
    relation_min_shared_frames: int = 3,
    relation_save_intermediate_frames: bool = True,
    relation_verbose: bool = False,
    with_reference: bool = False,
    reference_model_name: str = "gemini-3-flash-preview",
    reference_crop_output_dir: str = "output/reference_id_match_test",
    reference_frames_per_shot: int = 3,
    reference_save_intermediate_frames: bool = False,
):
    project_root = Path(__file__).resolve().parent
    output_path = str((project_root / output).resolve()) if not Path(output).is_absolute() else output

    generator = SceneGraphGenerator(
        yolo_model=yolo_model,
        tracker_backend=tracker_backend,
        scene_threshold=scene_threshold,
        min_scene_duration=min_scene_duration,
        conf=conf,
        iou=iou,
        sam3_model=sam3_model,
        sam2_model_cfg=sam2_model_cfg,
        sam2_checkpoint=sam2_checkpoint,
        sam3_iou_threshold = sam3_iou_threshold,
        sam3_overlap_threshold=sam3_overlap_threshold,
        sam3_redetection_interval=sam3_redetection_interval,
        groundedsam2_mask_output_dir=groundedsam2_mask_output_dir,
        sam3_mask_output_dir=sam3_mask_output_dir,
        sam3_match_output_dir=sam3_match_output_dir,
        sam3_match_log_path=sam3_match_log_path,
        use_action_detection=use_action_detection,
        action_config=action_config,
        action_checkpoint=action_checkpoint,
        action_label_map=action_label_map,
        action_score_thr=action_score_thr,
        action_topk=action_topk,
        action_frame_interval=action_frame_interval,
        skip_filter=skip_filter,
        filter_min_frames=filter_min_frames,
    )

    if not full_pipeline:
        if video:
            generator.process_video(video, output_path)
        elif video_dir:
            generator.process_videos(video_dir, output_path)
        else:
            print("Please provide video or video_dir")
        return

    if not video and not video_dir:
        raise ValueError("full_pipeline=True requires video or video_dir.")

    if video:
        video_paths: List[Path] = [Path(video)]
    else:
        src_dir = Path(video_dir)
        video_paths = sorted(list(src_dir.glob("*.mp4")) + list(src_dir.glob("*.avi")))

    if max_videos and max_videos > 0:
        video_paths = video_paths[:max_videos]

    if not video_paths:
        raise ValueError("No videos found to process.")

    existing_videos = _load_existing_videos(output_path)
    print(f"[pipeline] start: {len(video_paths)} video(s)")
    for idx, video_path_obj in enumerate(video_paths, start=1):
        video_path = str(video_path_obj)
        print()
        print(f"[pipeline] [{idx}/{len(video_paths)}] {video_path_obj.name}")
        if video_path in existing_videos or video_path_obj.stem in existing_videos:
            print("[pipeline] skip: already exists in output jsonl")
            continue
        print("[pipeline] stage: scene_graph")
        generator.process_video(video_path, output_path)
        existing_videos.add(video_path)
        existing_videos.add(video_path_obj.stem)
        print("[pipeline] stage: enrich_graph")
        _run_full_pipeline_for_video(
            video_path=video_path,
            output_jsonl=output_path,
            project_root=project_root,
            pipeline_python=pipeline_python,
            cuda_visible_devices=cuda_visible_devices,
            hf_endpoint=hf_endpoint,
            with_attribute=with_attribute,
            attribute_model_name=attribute_model_name,
            attribute_masks_json=attribute_masks_json,
            attribute_model_path=attribute_model_path,
            attribute_max_frames=attribute_max_frames,
            with_action=with_action,
            action_config=action_config,
            action_checkpoint=action_checkpoint,
            action_label_map=action_label_map,
            action_frame_interval=action_frame_interval,
            action_python=action_python,
            action_device=action_device,
            with_relation=with_relation,
            relation_model_name=relation_model_name,
            relation_crop_output_dir=relation_crop_output_dir,
            relation_min_shared_frames=relation_min_shared_frames,
            relation_save_intermediate_frames=relation_save_intermediate_frames,
            relation_verbose=relation_verbose,
            with_reference=with_reference,
            reference_model_name=reference_model_name,
            reference_crop_output_dir=reference_crop_output_dir,
            reference_frames_per_shot=reference_frames_per_shot,
            reference_save_intermediate_frames=reference_save_intermediate_frames,
        )

    print()
    print("[pipeline] finished")


if __name__ == '__main__':
    fire.Fire(run)
