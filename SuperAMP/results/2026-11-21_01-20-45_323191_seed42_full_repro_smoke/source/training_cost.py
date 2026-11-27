"""Shared wall-clock accounting boundary for PPO and WMAC checkpoints."""

WALL_TIME_SCOPE = (
    "runner initialization, collection, warmup, updates, prior saves/logging/evaluations; "
    "excludes simulator/environment construction and external evaluation"
)
