import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


class GraphFilter:
    def __init__(
        self,
        min_frames: int = 30,
        max_gap_ratio: float = 0.5,
        min_temporal_coverage: float = 0.1,
        max_flicker_segments: int = 3,
        min_stable_segment_length: int = 15,
    ):
        self.min_frames = min_frames
        self.max_gap_ratio = max_gap_ratio
        self.min_temporal_coverage = min_temporal_coverage
        self.max_flicker_segments = max_flicker_segments
        self.min_stable_segment_length = min_stable_segment_length

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

    def is_flickering(self, segments: List[Tuple[int, int]]) -> bool:
        if len(segments) <= self.max_flicker_segments:
            return False

        short_segments = sum(1 for s, e in segments if (e - s + 1) < self.min_stable_segment_length)

        if short_segments > len(segments) * 0.6:
            return True

        return False

    def keep_stable_segments(self, obj_node: Dict[str, Any]) -> Dict[str, Any]:
        bboxes = obj_node.get('bboxes', {})
        if not bboxes:
            return obj_node

        frame_indices = sorted([int(f) for f in bboxes.keys()])
        segments = self.find_continuous_segments(frame_indices, max_gap=5)

        if not self.is_flickering(segments):
            return obj_node

        stable_segments = [(s, e) for s, e in segments
                          if (e - s + 1) >= self.min_stable_segment_length]

        if not stable_segments:
            return None

        new_bboxes = {}
        for start, end in stable_segments:
            for frame_idx in frame_indices:
                if start <= frame_idx <= end:
                    new_bboxes[str(frame_idx)] = bboxes[str(frame_idx)]

        if len(new_bboxes) < self.min_frames:
            return None

        obj_node['bboxes'] = new_bboxes
        obj_node['start_frame'] = min(int(f) for f in new_bboxes.keys())
        obj_node['end_frame'] = max(int(f) for f in new_bboxes.keys())

        return obj_node

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
            return False

        if metrics['gap_ratio'] > self.max_gap_ratio:
            return False

        if metrics['temporal_coverage'] < self.min_temporal_coverage:
            return False

        return True

    def filter_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        filtered_objects = []
        kept_node_ids = set()

        for obj_node in graph['object_nodes']:
            processed_node = self.keep_stable_segments(obj_node)

            if processed_node is None:
                continue

            if self.should_keep(processed_node):
                filtered_objects.append(processed_node)
                kept_node_ids.add(processed_node['node_id'])

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
        max_flicker_segments: int = 3,
        min_stable_segment_length: int = 15,
        max_area_change_ratio: float = 0.5,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
    ):
        super().__init__(
            min_frames=min_frames,
            max_gap_ratio=max_gap_ratio,
            min_temporal_coverage=min_temporal_coverage,
            max_flicker_segments=max_flicker_segments,
            min_stable_segment_length=min_stable_segment_length,
        )
        self.max_area_change_ratio = max_area_change_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def check_bbox_quality(self, obj_node: Dict[str, Any]) -> bool:
        bboxes = obj_node.get('bboxes', {})
        if not bboxes:
            return False

        frame_indices = sorted([int(f) for f in bboxes.keys()])

        for i, frame_idx in enumerate(frame_indices):
            bbox = bboxes[str(frame_idx)]
            x1, y1, x2, y2 = bbox

            width = x2 - x1
            height = y2 - y1
            area = width * height

            if width <= 0 or height <= 0 or area <= 0:
                return False

            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                return False

            if i > 0:
                prev_bbox = bboxes[str(frame_indices[i - 1])]
                prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])

                if prev_area > 0:
                    area_change_ratio = abs(area - prev_area) / prev_area
                    if area_change_ratio > self.max_area_change_ratio:
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
    max_flicker_segments: int = 3,
    min_stable_segment_length: int = 15,
    use_sam3_quality: bool = False,
    max_area_change_ratio: float = 0.5,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 5.0,
):
    if use_sam3_quality:
        filter = SAM3QualityFilter(
            min_frames=min_frames,
            max_gap_ratio=max_gap_ratio,
            min_temporal_coverage=min_temporal_coverage,
            max_flicker_segments=max_flicker_segments,
            min_stable_segment_length=min_stable_segment_length,
            max_area_change_ratio=max_area_change_ratio,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
    else:
        filter = GraphFilter(
            min_frames=min_frames,
            max_gap_ratio=max_gap_ratio,
            min_temporal_coverage=min_temporal_coverage,
            max_flicker_segments=max_flicker_segments,
            min_stable_segment_length=min_stable_segment_length,
        )

    if video_folder:
        filter.filter_video_folder(input_path, video_folder, output_path)
    else:
        filter.filter_jsonl(input_path, output_path)


if __name__ == '__main__':
    import fire
    fire.Fire(filter_scene_graphs)
