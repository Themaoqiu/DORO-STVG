import json
from pathlib import Path
from typing import Dict, Any


class GraphFilter:
    def __init__(
        self,
        min_frames: int = 5,
    ):
        self.min_frames = min_frames

    def should_keep(self, obj_node: Dict[str, Any]) -> bool:
        total_frames = len(obj_node.get('bboxes', {}))

        if total_frames < self.min_frames:
            print(f"    Filtered {obj_node['node_id']}: too few frames ({total_frames} < {self.min_frames})")
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

def filter_scene_graphs(
    input_path: str,
    output_path: str,
    video_folder: str = None,
    min_frames: int = 5,
):
    graph_filter = GraphFilter(min_frames=min_frames)

    if video_folder:
        graph_filter.filter_video_folder(input_path, video_folder, output_path)
    else:
        graph_filter.filter_jsonl(input_path, output_path)


if __name__ == '__main__':
    import fire
    fire.Fire(filter_scene_graphs)
