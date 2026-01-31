import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


class GraphFilter:
    def __init__(
        self,
        min_frames: int = 30,
        max_gap_ratio: float = 0.5,
        min_temporal_coverage: float = 0.1,
    ):
        self.min_frames = min_frames
        self.max_gap_ratio = max_gap_ratio
        self.min_temporal_coverage = min_temporal_coverage

    def find_continuous_segments(self, frame_indices: List[int], max_gap: int = 5) -> List[Tuple[int, int]]:
        if not frame_indices:
            return []

        segments = []
        start = frame_indices[0]
        prev = frame_indices[0]

        for curr in frame_indices[1:]:
            if curr - prev > max_gap:
                segments.append((start, prev))
                start = curr
            prev = curr

        segments.append((start, prev))
        return segments

    def compute_metrics(self, obj_node: Dict[str, Any]) -> Dict[str, float]:
        bboxes = obj_node.get('bboxes', {})
        if not bboxes:
            return {'total_frames': 0, 'gap_ratio': 1.0, 'temporal_coverage': 0.0}

        frame_indices = sorted([int(f) for f in bboxes.keys()])
        total_frames = len(frame_indices)
        start_frame = obj_node['start_frame']
        end_frame = obj_node['end_frame']
        span = end_frame - start_frame + 1

        gaps = []
        for i in range(len(frame_indices) - 1):
            gap = frame_indices[i + 1] - frame_indices[i] - 1
            if gap > 0:
                gaps.append(gap)

        gap_ratio = sum(gaps) / span if span > 0 and gaps else 0.0
        temporal_coverage = total_frames / span if span > 0 else 0.0

        return {
            'total_frames': total_frames,
            'gap_ratio': gap_ratio,
            'temporal_coverage': temporal_coverage,
            'num_gaps': len(gaps),
        }

    def should_keep(self, obj_node: Dict[str, Any]) -> bool:
        metrics = self.compute_metrics(obj_node)

        if metrics['total_frames'] < self.min_frames:
            print(f"    Filtered {obj_node['node_id']}: too few frames ({metrics['total_frames']} < {self.min_frames})")
            return False

        if metrics['gap_ratio'] > self.max_gap_ratio:
            print(f"    Filtered {obj_node['node_id']}: gap ratio too high ({metrics['gap_ratio']:.2f} > {self.max_gap_ratio})")
            return False

        if metrics['temporal_coverage'] < self.min_temporal_coverage:
            print(f"    Filtered {obj_node['node_id']}: temporal coverage too low ({metrics['temporal_coverage']:.2f} < {self.min_temporal_coverage})")
            return False

        return True

    def filter_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        filtered_objects = []
        kept_node_ids = set()

        for obj_node in graph['object_nodes']:
            if self.should_keep(obj_node):
                filtered_objects.append(obj_node)
                kept_node_ids.add(obj_node['node_id'])

        filtered_edges = [
            edge for edge in graph['edges']
            if edge['source_id'] in kept_node_ids
        ]

        graph['object_nodes'] = filtered_objects
        graph['edges'] = filtered_edges

        return graph

    def filter_jsonl(self, input_path: str, output_path: str):
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_objects_before = 0
        total_objects_after = 0

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                graph = json.loads(line.strip())
                total_objects_before += len(graph['object_nodes'])

                filtered_graph = self.filter_graph(graph)
                total_objects_after += len(filtered_graph['object_nodes'])

                f_out.write(json.dumps(filtered_graph, ensure_ascii=False) + '\n')

        print(f"Filtered: {total_objects_before} -> {total_objects_after} objects "
              f"({total_objects_before - total_objects_after} removed)")

    def filter_video_folder(self, input_jsonl: str, video_folder: str, output_path: str):
        input_file = Path(input_jsonl)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        video_folder = Path(video_folder)
        video_paths = set()

        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']:
            video_paths.update(str(p) for p in video_folder.glob(ext))

        if not video_paths:
            print(f"No videos found in {video_folder}")
            return

        print(f"Found {len(video_paths)} videos in folder")

        total_objects_before = 0
        total_objects_after = 0
        graphs_processed = 0
        graphs_kept = 0

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                graph = json.loads(line.strip())
                graph_video_path = graph.get('video_path', '')
                graphs_processed += 1

                if graph_video_path not in video_paths:
                    if Path(graph_video_path).name not in [Path(p).name for p in video_paths]:
                        continue

                graphs_kept += 1
                total_objects_before += len(graph['object_nodes'])

                filtered_graph = self.filter_graph(graph)
                total_objects_after += len(filtered_graph['object_nodes'])

                f_out.write(json.dumps(filtered_graph, ensure_ascii=False) + '\n')

        print(f"Processed: {graphs_processed} graphs, kept {graphs_kept} graphs from video folder")
        print(f"Filtered: {total_objects_before} -> {total_objects_after} objects "
              f"({total_objects_before - total_objects_after} removed)")


class SAM3QualityFilter(GraphFilter):
    def __init__(
        self,
        min_frames: int = 30,
        max_gap_ratio: float = 0.5,
        min_temporal_coverage: float = 0.1,
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 10.0,
    ):
        super().__init__(
            min_frames=min_frames,
            max_gap_ratio=max_gap_ratio,
            min_temporal_coverage=min_temporal_coverage,
        )
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def check_bbox_quality(self, obj_node: Dict[str, Any]) -> bool:
        bboxes = obj_node.get('bboxes', {})
        if not bboxes:
            return False

        for frame_idx in bboxes.keys():
            bbox = bboxes.get(frame_idx) or bboxes.get(str(frame_idx))
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                return False

            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                return False

        return True

    def should_keep(self, obj_node: Dict[str, Any]) -> bool:
        if not super().should_keep(obj_node):
            return False

        return self.check_bbox_quality(obj_node)


def filter_scene_graphs(
    input_path: str,
    output_path: str,
    video_folder: str = None,
    min_frames: int = 30,
    max_gap_ratio: float = 0.5,
    min_temporal_coverage: float = 0.1,
    use_sam3_quality: bool = False,
):
    if use_sam3_quality:
        graph_filter = SAM3QualityFilter(
            min_frames=min_frames,
            max_gap_ratio=max_gap_ratio,
            min_temporal_coverage=min_temporal_coverage,
        )
    else:
        graph_filter = GraphFilter(
            min_frames=min_frames,
            max_gap_ratio=max_gap_ratio,
            min_temporal_coverage=min_temporal_coverage,
        )

    if video_folder:
        graph_filter.filter_video_folder(input_path, video_folder, output_path)
    else:
        graph_filter.filter_jsonl(input_path, output_path)


if __name__ == '__main__':
    import fire
    fire.Fire(filter_scene_graphs)
