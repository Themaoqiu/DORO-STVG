__all__ = [
    "Qwen2_5VL",
    "Qwen3VL",
    "LlavaSTQwen2",
    "VideoMolmoModel",
    "CGSTVGModel",
    "TASTVGModel",
]


def __getattr__(name):
    if name in {"Qwen2_5VL", "Qwen3VL"}:
        from .qwen_family import Qwen2_5VL, Qwen3VL

        return {"Qwen2_5VL": Qwen2_5VL, "Qwen3VL": Qwen3VL}[name]

    if name == "LlavaSTQwen2":
        from .llava_st import LlavaSTQwen2

        return LlavaSTQwen2

    if name == "VideoMolmoModel":
        from .videomolmo import VideoMolmoModel

        return VideoMolmoModel

    if name == "CGSTVGModel":
        from .cgstvg import CGSTVGModel

        return CGSTVGModel

    if name == "TASTVGModel":
        from .tastvg import TASTVGModel

        return TASTVGModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
