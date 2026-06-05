#!/usr/bin/env python3
"""Merge mose+perception query jsonl and split 5:2 into sft / rl.

Usage:
    python split_compstvg_sft_rl.py \
        --inputs /home/wangxingjian/data/compstvg/mose.queries.jsonl \
                 /home/wangxingjian/data/compstvg/preception.queries.jsonl \
        --out-dir /home/wangxingjian/data/compstvg/splits \
        --seed 42

Records are merged, shuffled with the given seed, then split 5/(5+2) to sft
and 2/(5+2) to rl. A `source` field is added so each record traces back to
its origin file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft-ratio", type=int, default=5)
    p.add_argument("--rl-ratio", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for path in args.inputs:
        src = Path(path).stem
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["source"] = src
                records.append(obj)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    total = len(records)
    denom = args.sft_ratio + args.rl_ratio
    n_sft = total * args.sft_ratio // denom
    sft_records = records[:n_sft]
    rl_records = records[n_sft:]

    sft_path = out_dir / "sft.jsonl"
    rl_path = out_dir / "rl.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(rl_path, "w", encoding="utf-8") as f:
        for r in rl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total: {total}")
    print(f"SFT  : {len(sft_records)} -> {sft_path}")
    print(f"RL   : {len(rl_records)} -> {rl_path}")
    by_src_sft: dict[str, int] = {}
    by_src_rl: dict[str, int] = {}
    for r in sft_records:
        by_src_sft[r["source"]] = by_src_sft.get(r["source"], 0) + 1
    for r in rl_records:
        by_src_rl[r["source"]] = by_src_rl.get(r["source"], 0) + 1
    print(f"SFT by source: {by_src_sft}")
    print(f"RL  by source: {by_src_rl}")


if __name__ == "__main__":
    main()
