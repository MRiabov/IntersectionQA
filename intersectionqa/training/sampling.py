"""Sampling helpers for capped training/evaluation runs."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Callable, Literal, TypeVar

RowT = TypeVar("RowT")
SamplingStrategy = Literal["random", "stratified_task"]


def select_rows(
    rows: list[RowT],
    *,
    limit: int | None,
    seed: int,
    strategy: SamplingStrategy = "stratified_task",
    key: Callable[[RowT], str],
) -> list[RowT]:
    """Return a deterministic capped sample.

    `random` keeps the previous behavior. `stratified_task` round-robins across
    task groups after shuffling within each group, which prevents tiny GRPO
    canaries from starving low-frequency IntersectionEdit task families.
    """

    rng = random.Random(seed)
    if limit is None or limit <= 0:
        selected = list(rows)
        rng.shuffle(selected)
        return selected
    if limit >= len(rows):
        selected = list(rows)
        rng.shuffle(selected)
        return selected
    if strategy == "random":
        selected = list(rows)
        rng.shuffle(selected)
        return selected[:limit]
    if strategy != "stratified_task":
        raise ValueError(f"Unknown row sampling strategy: {strategy}")

    groups: dict[str, list[RowT]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    group_names = sorted(groups)
    rng.shuffle(group_names)
    selected: list[RowT] = []
    while len(selected) < limit and group_names:
        next_group_names: list[str] = []
        for group_name in group_names:
            group_rows = groups[group_name]
            if group_rows:
                selected.append(group_rows.pop())
                if len(selected) >= limit:
                    break
            if group_rows:
                next_group_names.append(group_name)
        group_names = next_group_names
    return selected


def select_diverse_low_reward_samples(samples: list[dict], sample_count: int) -> list[dict]:
    """Select debug samples, prioritizing low reward and task diversity."""

    if sample_count <= 0:
        return []
    sorted_samples = sorted(samples, key=lambda sample: (float(sample["reward"]) >= 1.0, float(sample["reward"])))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    seen_tasks: set[str] = set()
    for sample in sorted_samples:
        task_type = str(sample["task_type"])
        if task_type in seen_tasks:
            continue
        selected.append(sample)
        selected_ids.add(str(sample["row_id"]))
        seen_tasks.add(task_type)
        if len(selected) >= sample_count:
            return selected
    for sample in sorted_samples:
        row_id = str(sample["row_id"])
        if row_id in selected_ids:
            continue
        selected.append(sample)
        if len(selected) >= sample_count:
            break
    return selected
