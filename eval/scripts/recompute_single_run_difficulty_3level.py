#!/usr/bin/env python3
import json
from pathlib import Path


THRESHOLDS_TEXT = "easy < 1/3, medium < 2/3, hard >= 2/3"
BUCKETS = ("easy", "medium", "hard")
METRIC_KEYS = ("m_tIoU", "m_vIoU", "vIoU@0.3", "vIoU@0.5")


def to_bucket(difficulty_score: float) -> str:
    if difficulty_score < 1 / 3:
        return "easy"
    if difficulty_score < 2 / 3:
        return "medium"
    return "hard"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def main() -> None:
    run_dir = Path("/home/wangxingjian/DORO-STVG/eval/res/dorostvg_compstvg_20260525_025350")
    results_path = run_dir / "results.jsonl"
    status_path = run_dir / "status.json"
    summary_path = run_dir / "difficulty_3level_summary.md"
    detail_path = run_dir / "difficulty_3level_status.json"

    status = json.loads(status_path.read_text())
    accumulators = {
        bucket: {"num_samples": 0, **{metric: 0.0 for metric in METRIC_KEYS}}
        for bucket in BUCKETS
    }

    with results_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            bucket = to_bucket(row["metadata"]["difficulty_score"])
            bucket_stats = accumulators[bucket]
            bucket_stats["num_samples"] += 1
            for metric in METRIC_KEYS:
                bucket_stats[metric] += row["metrics"][metric]

    averages = {}
    for bucket, bucket_stats in accumulators.items():
        count = bucket_stats["num_samples"]
        averages[bucket] = {
            "num_samples": count,
            **{
                metric: (bucket_stats[metric] / count if count else 0.0)
                for metric in METRIC_KEYS
            },
        }

    detail_payload = {
        "dataset": status.get("dataset"),
        "model": status.get("model"),
        "timestamp": status.get("timestamp"),
        "num_samples": status.get("num_samples"),
        "difficulty_3level_thresholds": THRESHOLDS_TEXT,
        "average_metrics": averages,
    }
    detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2) + "\n")

    table_row = (
        f"| {status.get('model', 'unknown')} | "
        f"{averages['easy']['num_samples']} | {format_percent(averages['easy']['m_tIoU'])} | "
        f"{format_percent(averages['easy']['m_vIoU'])} | {format_percent(averages['easy']['vIoU@0.3'])} | "
        f"{format_percent(averages['easy']['vIoU@0.5'])} | {averages['medium']['num_samples']} | "
        f"{format_percent(averages['medium']['m_tIoU'])} | {format_percent(averages['medium']['m_vIoU'])} | "
        f"{format_percent(averages['medium']['vIoU@0.3'])} | {format_percent(averages['medium']['vIoU@0.5'])} | "
        f"{averages['hard']['num_samples']} | {format_percent(averages['hard']['m_tIoU'])} | "
        f"{format_percent(averages['hard']['m_vIoU'])} | {format_percent(averages['hard']['vIoU@0.3'])} | "
        f"{format_percent(averages['hard']['vIoU@0.5'])} |"
    )
    summary = "\n".join(
        [
            "# DORO-STVG 3-Level Difficulty Summary",
            "",
            f"Source: `{run_dir}`",
            "",
            "Thresholds: `easy < 1/3`, `medium < 2/3`, `hard >= 2/3`",
            "",
            "| model | easy count | easy m_tIoU | easy m_vIoU | easy vIoU@0.3 | easy vIoU@0.5 | medium count | medium m_tIoU | medium m_vIoU | medium vIoU@0.3 | medium vIoU@0.5 | hard count | hard m_tIoU | hard m_vIoU | hard vIoU@0.3 | hard vIoU@0.5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            table_row,
            "",
        ]
    )
    summary_path.write_text(summary)


if __name__ == "__main__":
    main()
