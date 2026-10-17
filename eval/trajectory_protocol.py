"""Ordering and safe resume rules for trajectory-persistent evaluation state."""
from __future__ import annotations

import hashlib
import json

PROTOCOL = "androidcontrol_trajectory_v3"


def episode_id(sample):
    value = sample.get("episode_id")
    if value is None or not str(value).strip():
        raise ValueError("Trajectory evaluation requires an episode_id for every sample")
    return str(value).strip()


def step_index(sample):
    raw = sample.get("step_index")
    if raw is None:
        raw = sample.get("step")
    if isinstance(raw, bool) or raw is None or not str(raw).isdigit():
        raise ValueError(f"Invalid trajectory step: {raw!r}")
    return int(raw)


def trajectory_samples(samples, chunk_id=0, num_chunks=1):
    """Keep episode encounter order, sort steps, and never split an episode."""
    if num_chunks < 1 or not 0 <= chunk_id < num_chunks:
        raise ValueError("Invalid trajectory chunk")
    groups = {}
    seen = set()
    for sample in samples:
        key = episode_id(sample), step_index(sample)
        if key in seen:
            raise ValueError(f"Duplicate episode/step: {key}")
        seen.add(key)
        groups.setdefault(key[0], []).append(sample)
    episodes = list(groups.values())
    start = len(episodes) * chunk_id // num_chunks
    end = len(episodes) * (chunk_id + 1) // num_chunks
    return [s for group in episodes[start:end] for s in sorted(group, key=step_index)]


def protocol_fingerprint(samples, config):
    """Reject checkpoints from different data, order, or execution settings."""
    data = json.dumps([PROTOCOL, samples, config], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def resume_boundary(samples, completed):
    """Replay a partially completed episode; no MI state is saved in JSON."""
    if not 0 <= completed <= len(samples):
        raise ValueError("Checkpoint sample count is outside this evaluation")
    if completed == len(samples):
        return completed
    start = completed
    while start > 0 and episode_id(samples[start - 1]) == episode_id(samples[completed]):
        start -= 1
    return start
