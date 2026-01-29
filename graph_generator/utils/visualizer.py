import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import random


class SceneGraphVisualizer:
    def __init__(self, scene_graph_jsonl: str, video_path: str):
        self.video_path = video_path
        self.graph_data = self._load_graph(scene_graph_jsonl)
        self.colors = {}
        self._generate_colors()
    
    def _load_graph(self, jsonl_path: str) -> Dict:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['video_path'] == self.video_path or Path(data['video_path']).stem == Path(self.video_path).stem:
                    return data
        raise ValueError(f"No graph found for video {self.video_path}")
    
    def _generate_colors(self):
        for obj_node in self.graph_data['object_nodes']:
            obj_id = obj_node['global_track_id']
            self.colors[obj_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )
    
    def draw_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        show_labels: bool = True,
        show_ids: bool = True,
    ) -> np.ndarray:
        frame = frame.copy()
        
        for obj_node in self.graph_data['object_nodes']:
            if 'bboxes' not in obj_node:
                continue
            
            bboxes = obj_node['bboxes']
            if str(frame_idx) not in bboxes:
                continue
            
            box = bboxes[str(frame_idx)]
            x1, y1, x2, y2 = map(int, box)
            
            obj_id = obj_node['global_track_id']
            obj_class = obj_node['object_class']
            color = self.colors[obj_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_parts = []
            if show_ids:
                label_parts.append(f"ID:{obj_id}")
            if show_labels:
                label_parts.append(obj_class)
            
            if label_parts:
                label = " ".join(label_parts)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def visualize_video(
        self,
        output_path: str,
        fps: Optional[float] = None,
        show_labels: bool = True,
        show_ids: bool = True,
    ):
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps:.2f} fps")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = self.draw_frame(frame, frame_idx, show_labels, show_ids)
            out.write(annotated)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames")
        
        cap.release()
        out.release()
        print(f"Saved annotated video to {output_path}")
    
    def visualize_scene(
        self,
        clip_id: int,
        output_path: str,
        fps: Optional[float] = None,
    ):
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temporal_nodes = self.graph_data['temporal_nodes']
        clip = next((c for c in temporal_nodes if c['clip_id'] == clip_id), None)
        
        if not clip:
            raise ValueError(f"Clip {clip_id} not found")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip['start_frame'])
        
        for frame_idx in range(clip['start_frame'], clip['end_frame'] + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = self.draw_frame(frame, frame_idx)
            out.write(annotated)
        
        cap.release()
        out.release()
        print(f"Saved clip {clip_id} to {output_path}")
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Scene Graph Summary: {Path(self.video_path).name}")
        print(f"{'='*60}")
        
        print(f"\nTemporal Nodes (Scenes): {len(self.graph_data['temporal_nodes'])}")
        for clip in self.graph_data['temporal_nodes']:
            print(f"  Clip {clip['clip_id']}: frames {clip['start_frame']}-{clip['end_frame']} ({clip['duration']:.2f}s)")
        
        print(f"\nObject Nodes: {len(self.graph_data['object_nodes'])}")
        for obj in self.graph_data['object_nodes']:
            num_frames = len(obj.get('bboxes', {}))
            print(f"  {obj['node_id']}: {num_frames} frames, clips {obj['clip_ids']}")
        
        print(f"\nEdges: {len(self.graph_data['edges'])}")
        edge_types = {}
        for edge in self.graph_data['edges']:
            edge_type = edge['edge_type']
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        for edge_type, count in edge_types.items():
            print(f"  {edge_type}: {count}")
        
        print(f"{'='*60}\n")


def visualize_from_cli(
    jsonl_path: str,
    video_path: str,
    output_path: str,
    clip_id: Optional[int] = None,
):
    viz = SceneGraphVisualizer(jsonl_path, video_path)
    viz.print_summary()
    
    if clip_id is not None:
        viz.visualize_scene(clip_id, output_path)
    else:
        viz.visualize_video(output_path)


if __name__ == '__main__':
    import fire
    fire.Fire(visualize_from_cli)
