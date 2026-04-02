import re
from typing import Any


_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = _ANSWER_PATTERN.search(text)
    if match is None:
        return ""
    return match.group(1).strip()


def _normalize_mc_option(text: str) -> str:
    answer = _extract_answer(text) or (text.strip() if isinstance(text, str) else "")
    answer = answer.strip().upper()
    return answer[:1] if answer else ""


def _accuracy_reward(response: str, ground_truth: str, problem_type: str) -> float:
    if (problem_type or "").strip().lower() != "multiple choice":
        return 0.0
    pred = _normalize_mc_option(response)
    gt = _normalize_mc_option(ground_truth)
    return 1.0 if pred and gt and pred == gt else 0.0


def _format_reward(response: str) -> float:
    if not isinstance(response, str):
        return 0.0
    blocks = re.findall(
        r"<time>.*?</time>\s*<caption>.*?</caption>\s*<think>.*?</think>",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    times = re.findall(r"<time>.*?</time>", response, flags=re.DOTALL | re.IGNORECASE)
    captions = re.findall(r"<caption>.*?</caption>", response, flags=re.DOTALL | re.IGNORECASE)
    thinks = re.findall(r"<think>.*?</think>", response, flags=re.DOTALL | re.IGNORECASE)
    answers = re.findall(r"<answer>.*?</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    is_valid = (
        len(blocks) > 0
        and len(times) == len(captions) == len(thinks) == len(blocks)
        and len(answers) == 1
    )
    return 1.0 if is_valid else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 1.0,
    accuracy_weight: float = 1.0,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for item in reward_inputs:
        response = item.get("response", "")
        ground_truth = item.get("ground_truth", "")
        problem_type = item.get("problem_type", "")
        format_score = _format_reward(response)
        accuracy_score = _accuracy_reward(response, ground_truth, problem_type)
        overall = accuracy_weight * accuracy_score + format_weight * format_score
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
