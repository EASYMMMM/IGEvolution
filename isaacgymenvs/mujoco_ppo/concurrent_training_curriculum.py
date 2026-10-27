"""Balanced task allocation and training-only checkpoint ranking."""

from collections import deque

import numpy as np


TASKS = ("supported_startup", "stage3", "unsupported_inplace")


def allocate_tasks(count, fractions):
    weights = np.asarray(fractions, dtype=float)
    if (count < 3 or weights.shape != (3,) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("Three tasks require >=3 environments and three positive fractions summing to one")
    remaining = (int(count) - 3) * weights
    counts = np.floor(remaining).astype(int) + 1
    for index in np.argsort(-(remaining - np.floor(remaining)), kind="stable")[:int(count - counts.sum())]:
        counts[index] += 1
    return tuple(task for task, size in zip(TASKS, counts) for _ in range(size))


class TrainingBestTracker:
    def __init__(self, window=30, minimum=10, max_age=100):
        if minimum < 1 or window < minimum or max_age < 1:
            raise ValueError("Invalid training-best window, minimum or maximum age")
        self.window, self.minimum, self.max_age = int(window), int(minimum), int(max_age)
        self.reset()

    def reset(self):
        self.episodes = {task: deque(maxlen=self.window) for task in TASKS}
        self.generation = 0
        self.last_checked = 0

    def add(self, task, update, length, total_return, success, estimated=True):
        if not estimated:
            return
        if length < 1 or not np.isfinite(total_return):
            raise ValueError("Invalid completed training episode")
        self.episodes[task].append({
            "update": int(update), "length": int(length),
            "return": float(total_return), "success": bool(success),
        })
        self.generation += 1

    def candidate(self, update, horizon):
        if self.generation == self.last_checked:
            return None
        self.last_checked = self.generation
        stats = {}
        for task, episodes in self.episodes.items():
            fresh = [ep for ep in episodes if ep["update"] >= update - self.max_age]
            if len(fresh) < self.minimum:
                return None
            stats[task] = {
                "episodes": len(fresh),
                "success": float(np.mean([ep["success"] for ep in fresh])),
                "normalized_length": float(np.mean([ep["length"] / horizon for ep in fresh])),
                "reward_per_step": float(np.mean([ep["return"] / ep["length"] for ep in fresh])),
                "first_completed_update": min(ep["update"] for ep in fresh),
                "last_completed_update": max(ep["update"] for ep in fresh),
            }
        score = (
            min(s["success"] for s in stats.values()),
            float(np.mean([s["success"] for s in stats.values()])),
            min(s["normalized_length"] for s in stats.values()),
            float(np.mean([s["reward_per_step"] for s in stats.values()])),
        )
        return score, stats
