import os
import re
from typing import List, Optional, Tuple

import numpy as np
from decord import VideoReader, cpu


os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def sample_video_uniform(
    video_path: str,
    fps: float = 2.0,
    max_frames: Optional[int] = None,
    gt_fps: Optional[float] = None,
) -> Tuple[np.ndarray, List[int], float]:
    """Uniformly sample frames at the target fps, capped at ``max_frames``.

    Returned ``sampled_indices`` are in the **GT reference space**: each
    sampled frame is reported as the closest integer index along a virtual
    timeline running at ``gt_fps``. With ``gt_fps == fps`` (the typical
    setup where the model is told "video sampled at X fps" and GT is
    annotated at the same X fps), this is exactly the GT key space.

    If ``gt_fps`` is ``None`` it defaults to ``fps`` (caller-supplied
    sampling fps), which is what you want when the GT was constructed with
    the same fps you are feeding the model.

    Returns (frames_uint8[T,H,W,3], gt_space_indices, native_fps).
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")
    native_fps = float(vr.get_avg_fps()) or 30.0
    reference_fps = float(gt_fps) if gt_fps is not None else float(fps)

    duration = total / native_fps
    target = max(1, min(total, int(round(duration * fps))))
    if max_frames is not None and max_frames > 0:
        target = min(target, max_frames)

    if target == 1:
        native_indices = [0]
    else:
        positions = np.linspace(0, total - 1, num=target)
        native_indices = sorted({int(round(v)) for v in positions})

    frames = vr.get_batch(native_indices).asnumpy()

    # Convert each native frame index to GT-fps space:
    #   gt_idx = round((native_idx / native_fps) * gt_fps)
    # Time-based, so it works for any (native_fps, sampling fps, gt_fps)
    # combination, including videos whose native fps already equals gt_fps.
    gt_space_indices = [
        int(round((n / native_fps) * reference_fps)) for n in native_indices
    ]

    return frames, gt_space_indices, native_fps


def remap_frame_indices(response_text: str, sampled_indices: List[int]) -> str:
    """Rewrite ``"{idx}": [...]`` frame-index keys in a model response from
    sampled-frame space (1..K, with K = len(sampled_indices)) back to the
    original-video-frame space.

    Caller should only invoke this when downsampling actually occurred. A
    1-based index is assumed to match the prompt's ``Frame-1, Frame-2, ...``
    convention; 0-based hits are also accepted as a fallback.
    """
    if not response_text or not sampled_indices:
        return response_text

    K = len(sampled_indices)

    def _replace(match: re.Match) -> str:
        k = int(match.group(1))
        if 1 <= k <= K:
            mapped = sampled_indices[k - 1]
        elif 0 <= k < K:
            mapped = sampled_indices[k]
        else:
            return match.group(0)
        return f'"{mapped}":'

    return re.sub(r'"(\d+)"\s*:', _replace, response_text)
