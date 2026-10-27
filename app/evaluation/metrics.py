"""Classification and aggregation metrics for routing decisions.

Rows are dicts with at least ``expected`` and the chosen action (``got`` by default), as action
values (``reuse`` / ``request_evidence`` / ``synthesize``). Standard deviations are sample standard
deviations (ddof = 1) over orderings.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

LABELS = ("reuse", "request_evidence", "synthesize")


def classification(rows: Sequence[dict], got: str = "got", expected: str = "expected") -> dict:
    """Accuracy, the confusion matrix (rows: expected, columns: chosen), per-class P/R/F1 and macro-F1."""
    confusion = Counter((r[expected], r[got]) for r in rows)
    n = len(rows)
    correct = sum(confusion[(label, label)] for label in LABELS)
    per_class = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        support = sum(confusion[(label, other)] for other in LABELS)
        predicted = sum(confusion[(other, label)] for other in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": predicted}
    matrix = [[confusion[(row, col)] for col in LABELS] for row in LABELS]
    normalized = [[v / sum(line) if sum(line) else 0.0 for v in line] for line in matrix]
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0, "per_class": per_class,
            "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
            "confusion": matrix, "confusion_row_normalized": normalized, "labels": list(LABELS)}  # fmt: skip


def accuracy(rows: Iterable[dict], got: str = "got", expected: str = "expected") -> float:
    rows = list(rows)
    return sum(r[got] == r[expected] for r in rows) / len(rows) if rows else float("nan")


def grouped(rows: Iterable[dict], key: Callable[[dict], str], got: str = "got") -> dict[str, dict]:
    """Accuracy per group (e.g. per Table-2 cell)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: {"n": len(rs), "accuracy": accuracy(rs, got)} for name, rs in sorted(groups.items())}


def mean_std(values: Sequence[float]) -> dict[str, float]:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(values)}


def relative_gain(learned: float, baseline: float) -> float:
    return (learned - baseline) / abs(baseline) if baseline else float("nan")
