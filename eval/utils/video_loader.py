import os
from typing import List, Tuple

import numpy as np
from decord import VideoReader, cpu


os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def sample_video_uniform(
    video_path: str,
    fps: float = 2.0,
) -> Tuple[np.ndarray, List[int], float]:
    """Uniformly sample frames at the target fps.

    Returns (frames_uint8[T,H,W,3], sampled_indices_in_original, native_fps).
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")
    native_fps = float(vr.get_avg_fps()) or 30.0

    duration = total / native_fps
    target = max(1, min(total, int(round(duration * fps))))

    if target == 1:
        indices = [0]
    else:
        positions = np.linspace(0, total - 1, num=target)
        indices = sorted({int(round(v)) for v in positions})

    frames = vr.get_batch(indices).asnumpy()
    return frames, [int(i) for i in indices], native_fps
