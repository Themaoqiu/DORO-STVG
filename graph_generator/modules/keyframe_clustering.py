from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans


class KeyframeClustering:
    def __init__(
        self,
        max_interval: int = 15,
        window_size: int = 30,
        min_keyframes: int = 2,
    ):
        self.max_interval = max_interval
        self.window_size = window_size
        self.min_keyframes = min_keyframes

    def select_keyframes(
        self,
        frames_dict: Dict[int, Dict],
        video_width: int,
        video_height: int,
    ) -> List[int]:
        if not frames_dict:
            return []

        frame_indices = sorted(frames_dict.keys())

        if len(frame_indices) <= self.min_keyframes:
            return frame_indices

        features = self._extract_features(frames_dict, frame_indices, video_width, video_height)

        keyframes = []
        start_idx = 0

        while start_idx < len(frame_indices):
            end_idx = min(start_idx + self.window_size, len(frame_indices))
            window_frames = frame_indices[start_idx:end_idx]
            window_features = features[start_idx:end_idx]

            if len(window_frames) <= 2:
                keyframes.extend(window_frames)
                break

            n_clusters = max(1, len(window_frames) // self.max_interval)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(window_features)

            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    continue

                cluster_features = window_features[cluster_mask]
                centroid = kmeans.cluster_centers_[cluster_id]

                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]

                keyframe = window_frames[closest_idx]
                if not keyframes or keyframe > keyframes[-1]:
                    keyframes.append(keyframe)

            start_idx = end_idx

        keyframes = sorted(set(keyframes))

        keyframes = self._enforce_interval(keyframes, frame_indices)

        if frame_indices[0] not in keyframes:
            keyframes.insert(0, frame_indices[0])
        if frame_indices[-1] not in keyframes:
            keyframes.append(frame_indices[-1])

        return sorted(set(keyframes))

    def _extract_features(
        self,
        frames_dict: Dict[int, Dict],
        frame_indices: List[int],
        video_width: int,
        video_height: int,
    ) -> np.ndarray:
        features = []

        for frame_idx in frame_indices:
            frame_data = frames_dict[frame_idx]
            box = frame_data['box']
            conf = frame_data.get('conf', 1.0)

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / (2 * video_width)
            cy = (y1 + y2) / (2 * video_height)
            area = ((x2 - x1) * (y2 - y1)) / (video_width * video_height)

            features.append([cx, cy, area, conf])

        return np.array(features)

    def _enforce_interval(self, keyframes: List[int], all_frames: List[int]) -> List[int]:
        if len(keyframes) <= 1:
            return keyframes

        result = [keyframes[0]]

        for i in range(1, len(keyframes)):
            gap = keyframes[i] - result[-1]

            if gap <= self.max_interval:
                result.append(keyframes[i])
            else:
                num_inserts = gap // self.max_interval
                for j in range(1, num_inserts + 1):
                    target_frame = result[-1] + j * self.max_interval
                    closest = min(
                        [f for f in all_frames if result[-1] < f < keyframes[i]],
                        key=lambda f: abs(f - target_frame),
                        default=None
                    )
                    if closest and closest not in result:
                        result.append(closest)

                result.append(keyframes[i])

        return sorted(set(result))
