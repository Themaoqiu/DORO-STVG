import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
from modules.scene_detector import SceneDetector
from modules.yolo_tracker import YOLOTracker
from modules.graph_filter import GraphFilter, SAM3QualityFilter


@dataclass
class ObjectNode:
    node_id: str
    global_track_id: int
    object_class: str
    start_frame: int
    end_frame: int
    clip_ids: List[int]
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
            'clip_ids': self.clip_ids,
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'object_node_id': self.object_node_id,
            'action_label': self.action_label,
            'frame_idx': self.frame_idx,
            'confidence': self.confidence,
        }


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
    temporal_nodes: List[Dict] = field(default_factory=list)
    object_nodes: List[Dict] = field(default_factory=list)
    attribute_nodes: List[Dict] = field(default_factory=list)
    action_nodes: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'video': self.video,
            'video_path': self.video_path,
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
        yolo_model: str = "yolo11n.pt",
        tracker_config: str = "botsort.yaml",
        scene_threshold: float = 3.0,
        min_scene_duration: float = 1.0,
        conf: float = 0.25,
        iou: float = 0.5,
        gap_threshold: int = 5,
        min_track_length: int = 10,
        use_sam3: bool = False,
        sam3_model: str = "sam3.pt",
        sam3_redetection_interval: int = 15,
        filter_min_frames: int = 30,
        filter_max_gap_ratio: float = 0.5,
        filter_min_temporal_coverage: float = 0.1,
        filter_max_flicker_segments: int = 3,
        filter_min_stable_segment_length: int = 15,
    ):
        self.yolo_model = yolo_model
        self.tracker_config = tracker_config
        self.scene_threshold = scene_threshold
        self.min_scene_duration = min_scene_duration
        self.conf = conf
        self.iou = iou
        self.gap_threshold = gap_threshold
        self.min_track_length = min_track_length
        self.use_sam3 = use_sam3
        self.sam3_model = sam3_model
        self.sam3_redetection_interval = sam3_redetection_interval
        self.filter_min_frames = filter_min_frames
        self.filter_max_gap_ratio = filter_max_gap_ratio
        self.filter_min_temporal_coverage = filter_min_temporal_coverage
        self.filter_max_flicker_segments = filter_max_flicker_segments
        self.filter_min_stable_segment_length = filter_min_stable_segment_length
    
    def process_video(self, video_path: str, output_path: str) -> SceneGraph:
        video_name = Path(video_path).stem
        graph = SceneGraph(video=video_name, video_path=video_path)

        print(f"[1/5] Detecting scenes...")
        scene_detector = SceneDetector(
            video_path,
            threshold=self.scene_threshold,
            min_scene_duration=self.min_scene_duration,
        )
        clips = scene_detector.detect()
        fps = scene_detector._fps

        graph.temporal_nodes = [clip.to_dict() for clip in clips]
        print(f"  Found {len(clips)} scenes")

        print(f"[2/5] Tracking objects with YOLO...")
        tracker = YOLOTracker(
            model_path=self.yolo_model,
            tracker_config=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            gap_threshold=self.gap_threshold,
            min_track_length=self.min_track_length,
        )
        all_shot_tracks = tracker.track_video(video_path, clips)
        global_tracks = tracker.merge_tracks(all_shot_tracks, fps=fps)

        total_local = sum(len(t) for t in all_shot_tracks.values())
        print(f"  Tracked {total_local} local tracks -> {len(global_tracks)} global tracks")

        if self.use_sam3:
            print(f"[3/5] Enhancing tracks with SAM3...")
            from modules.sam3_tracker import SAM3Tracker

            sam3_tracker = SAM3Tracker(
                model_path=self.sam3_model,
                redetection_interval=self.sam3_redetection_interval,
            )
            global_tracks = sam3_tracker.enhance_tracks(video_path, global_tracks, clips)
            print(f"  Enhanced {len(global_tracks)} tracks with SAM3")
        else:
            print(f"[3/5] Skipping SAM3 enhancement")

        print(f"[4/5] Building object nodes...")
        for g_track in global_tracks:
            bboxes = {}
            for local_track in g_track.local_tracks:
                for frame_idx, frame_data in local_track.frames.items():
                    bboxes[frame_idx] = frame_data['box']

            obj_node = ObjectNode(
                node_id=f"obj_{g_track.object_class}_{g_track.global_id}",
                global_track_id=g_track.global_id,
                object_class=g_track.object_class,
                start_frame=g_track.start_frame,
                end_frame=g_track.end_frame,
                clip_ids=list(set(t.clip_id for t in g_track.local_tracks)),
                bboxes=bboxes,
            )
            graph.object_nodes.append(obj_node.to_dict())

            for clip_id in obj_node.clip_ids:
                edge = Edge(
                    edge_id=f"edge_in_shot_{obj_node.node_id}_{clip_id}",
                    source_id=obj_node.node_id,
                    target_id=f"shot_{clip_id}",
                    edge_type="appears_in",
                )
                graph.edges.append(edge.to_dict())

        print(f"  Created {len(graph.object_nodes)} object nodes, {len(graph.edges)} edges")

        print(f"[5/5] Filtering graph...")
        if self.use_sam3:
            graph_filter = SAM3QualityFilter(
                min_frames=self.filter_min_frames,
                max_gap_ratio=self.filter_max_gap_ratio,
                min_temporal_coverage=self.filter_min_temporal_coverage,
                max_flicker_segments=self.filter_max_flicker_segments,
                min_stable_segment_length=self.filter_min_stable_segment_length,
            )
        else:
            graph_filter = GraphFilter(
                min_frames=self.filter_min_frames,
                max_gap_ratio=self.filter_max_gap_ratio,
                min_temporal_coverage=self.filter_min_temporal_coverage,
                max_flicker_segments=self.filter_max_flicker_segments,
                min_stable_segment_length=self.filter_min_stable_segment_length,
            )

        graph_dict = graph.to_dict()
        filtered_graph_dict = graph_filter.filter_graph(graph_dict)

        graph.object_nodes = filtered_graph_dict['object_nodes']
        graph.edges = filtered_graph_dict['edges']

        print(f"  Filtered to {len(graph.object_nodes)} object nodes, {len(graph.edges)} edges")

        graph.save_to_jsonl(output_path)
        print(f"Saved scene graph to {output_path}")

        return graph
    
    def process_videos(self, video_dir: str, output_path: str) -> List[SceneGraph]:
        video_dir = Path(video_dir)
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        
        graphs = []
        for i, video_path in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] Processing {video_path.name}")
            try:
                graph = self.process_video(str(video_path), output_path)
                graphs.append(graph)
            except Exception as e:
                print(f"  Error: {e}")
        
        return graphs


def run(
    video: str = None,
    video_dir: str = None,
    output: str = "output/scene_graphs.jsonl",
    yolo_model: str = "yolo11n.pt",
    tracker_config: str = "botsort.yaml",
    scene_threshold: float = 3.0,
    min_scene_duration: float = 1.0,
    conf: float = 0.25,
    iou: float = 0.5,
    gap_threshold: int = 5,
    min_track_length: int = 10,
):
    generator = SceneGraphGenerator(
        yolo_model=yolo_model,
        tracker_config=tracker_config,
        scene_threshold=scene_threshold,
        min_scene_duration=min_scene_duration,
        conf=conf,
        iou=iou,
        gap_threshold=gap_threshold,
        min_track_length=min_track_length,
    )
    
    if video:
        generator.process_video(video, output)
    elif video_dir:
        generator.process_videos(video_dir, output)
    else:
        print("Please provide video or video_dir")


if __name__ == '__main__':
    fire.Fire(run)
