"""Runtime wall-clock / latency / sync metrics for STaR-KV claims.

FLOPs-only "near-zero overhead" numbers omit GPU stalls from ``.item()`` /
``.cpu()`` syncs, CPU histogram work, and PCIe traffic. This module records:

- end-to-end wall-clock
- prefill vs decode split (via generate logits-processor probe)
- TTFT and inter-token latency (ITL)
- optional CUDA synchronize cost around compression MI
- peak / current GPU memory (allocated + reserved)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

try:
    from transformers import LogitsProcessor, LogitsProcessorList
except Exception:  # pragma: no cover
    LogitsProcessor = object  # type: ignore
    LogitsProcessorList = list  # type: ignore


class _LatencyProbe(LogitsProcessor):
    """Records a timestamp after each generate step (post-forward)."""

    def __init__(self, synchronize_cuda: bool = True):
        self.synchronize_cuda = bool(synchronize_cuda)
        self.timestamps: List[float] = []
        self._t0 = time.perf_counter()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timestamps.append(time.perf_counter())
        return scores

    def summary(self) -> Dict[str, Any]:
        ts = self.timestamps
        if not ts:
            return {
                "num_decode_steps": 0,
                "ttft_s": None,
                "itl_mean_s": None,
                "itl_p50_s": None,
                "decode_total_s": None,
                "tokens_per_s": None,
            }
        ttft = ts[0] - self._t0
        itls = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        itl_mean = sum(itls) / len(itls) if itls else None
        itl_p50 = sorted(itls)[len(itls) // 2] if itls else None
        decode_total = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        n = len(ts)
        return {
            "num_decode_steps": n,
            "ttft_s": round(ttft, 6),
            "prefill_plus_first_token_s": round(ttft, 6),
            "itl_mean_s": round(itl_mean, 6) if itl_mean is not None else None,
            "itl_p50_s": round(itl_p50, 6) if itl_p50 is not None else None,
            "decode_total_s": round(decode_total, 6),
            "tokens_per_s": round((n - 1) / decode_total, 3) if decode_total > 0 and n > 1 else None,
            "note": (
                "TTFT is wall time from generate() entry to first logits-processor "
                "callback (prefill + first token). ITL is subsequent step gaps. "
                "CUDA synchronize is enabled on the probe when requested."
            ),
        }


class RuntimeMetricsTracker:
    """Accumulates per-sample latency and compression-stage sync timings."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._enabled = True
        self._samples: List[Dict[str, Any]] = []
        self._mi_cuda_ms: List[float] = []
        self._mi_wall_ms: List[float] = []
        self._compress_wall_ms: List[float] = []
        self._grd_stats: List[Dict[str, Any]] = []
        self._grd_total_ms: List[float] = []
        self._grd_reduce_ms: List[float] = []
        self._grd_corr_ms: List[float] = []
        self._run_t0: Optional[float] = None
        self._run_t1: Optional[float] = None
        self._warmup_samples = 0
        self._warmup_seen = 0

    def enable(self, enabled: bool = True):
        self._enabled = bool(enabled)

    def set_warmup_samples(self, count: int = 0):
        self._warmup_samples = max(0, int(count))
        self._warmup_seen = 0

    def mark_run_start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._run_t0 = time.perf_counter()

    def mark_run_end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._run_t1 = time.perf_counter()

    def record_mi_timing(self, wall_ms: float, cuda_ms: Optional[float] = None):
        if not self._enabled:
            return
        self._mi_wall_ms.append(float(wall_ms))
        if cuda_ms is not None:
            self._mi_cuda_ms.append(float(cuda_ms))

    def record_compress_timing(self, wall_ms: float):
        if not self._enabled:
            return
        self._compress_wall_ms.append(float(wall_ms))

    def record_grd_stats(self, stats: Dict[str, Any]):
        """Record GRD-only timings / recoverability (kept separate from MI/TEM)."""
        if not self._enabled:
            return
        row = dict(stats) if stats else {}
        self._grd_stats.append(row)
        if row.get("total_ms") is not None:
            self._grd_total_ms.append(float(row["total_ms"]))
        if row.get("feature_reduce_ms") is not None:
            self._grd_reduce_ms.append(float(row["feature_reduce_ms"]))
        if row.get("correspondence_sim_ms") is not None:
            self._grd_corr_ms.append(float(row["correspondence_sim_ms"]))

    def record_sample(self, latency: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        if not self._enabled:
            return
        if self._warmup_seen < self._warmup_samples:
            self._warmup_seen += 1
            if self._warmup_seen == self._warmup_samples:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                self._run_t0 = time.perf_counter()
            return
        row = dict(latency)
        if extra:
            row.update(extra)
        self._samples.append(row)

    def get_summary(self, device: str = "cuda") -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        wall = None
        if self._run_t0 is not None and self._run_t1 is not None:
            wall = round(self._run_t1 - self._run_t0, 6)

        def _mean(xs: List[float]) -> Optional[float]:
            return round(sum(xs) / len(xs), 6) if xs else None

        def _sum(xs: List[float]) -> float:
            return round(sum(xs), 6) if xs else 0.0

        def _percentile(xs: List[float], q: float) -> Optional[float]:
            """Linearly interpolated percentile without a NumPy dependency."""
            if not xs:
                return None
            ordered = sorted(float(x) for x in xs)
            if len(ordered) == 1:
                return round(ordered[0], 6)
            pos = (len(ordered) - 1) * float(q)
            lo = int(pos)
            hi = min(lo + 1, len(ordered) - 1)
            value = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
            return round(value, 6)

        ttfts = [s["ttft_s"] for s in self._samples if s.get("ttft_s") is not None]
        itls = [s["itl_mean_s"] for s in self._samples if s.get("itl_mean_s") is not None]
        e2e = [s.get("e2e_generate_s") for s in self._samples if s.get("e2e_generate_s") is not None]
        step_e2e = [s.get("step_e2e_s") for s in self._samples if s.get("step_e2e_s") is not None]
        generated_tokens = sum(
            int(s.get("num_decode_steps", 0) or 0) for s in self._samples
        )
        generate_wall = sum(float(x) for x in e2e)

        mem = None
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_resv = torch.cuda.max_memory_reserved(device)
            curr = torch.cuda.memory_allocated(device)
            mem = {
                "peak_allocated_gb": round(peak_alloc / (1024 ** 3), 3),
                "peak_reserved_gb": round(peak_resv / (1024 ** 3), 3),
                "current_allocated_gb": round(curr / (1024 ** 3), 3),
                "note": "torch.cuda peak counters; prefer same cache layout when comparing methods.",
            }

        return {
            "wall_clock_s": wall,
            "num_samples_timed": len(self._samples),
            "num_warmup_samples_discarded": self._warmup_seen,
            "e2e_generate_mean_s": _mean(e2e),
            "e2e_generate_p50_s": _percentile(e2e, 0.50),
            "e2e_generate_p95_s": _percentile(e2e, 0.95),
            "step_e2e_mean_s": _mean(step_e2e),
            "step_e2e_p50_s": _percentile(step_e2e, 0.50),
            "step_e2e_p95_s": _percentile(step_e2e, 0.95),
            "steps_per_s_e2e_timed": (
                round(len(step_e2e) / sum(float(x) for x in step_e2e), 6)
                if step_e2e and sum(float(x) for x in step_e2e) > 0 else None
            ),
            "ttft_mean_s": _mean(ttfts),
            "ttft_p50_s": _percentile(ttfts, 0.50),
            "ttft_p95_s": _percentile(ttfts, 0.95),
            "itl_mean_s": _mean(itls),
            "itl_p50_s": _percentile(itls, 0.50),
            "itl_p95_s": _percentile(itls, 0.95),
            "generated_tokens_timed": generated_tokens,
            "generate_wall_s_sum": round(generate_wall, 6),
            "samples_per_s_generate": (
                round(len(e2e) / generate_wall, 6) if generate_wall > 0 else None
            ),
            "tokens_per_s_generate_e2e": (
                round(generated_tokens / generate_wall, 6)
                if generate_wall > 0 else None
            ),
            "mi_wall_ms_sum": _sum(self._mi_wall_ms),
            "mi_wall_ms_mean": _mean(self._mi_wall_ms),
            "mi_cuda_ms_sum": _sum(self._mi_cuda_ms) if self._mi_cuda_ms else None,
            "mi_cuda_ms_mean": _mean(self._mi_cuda_ms),
            "compress_wall_ms_sum": _sum(self._compress_wall_ms),
            "compress_wall_ms_mean": _mean(self._compress_wall_ms),
            "grd_total_ms_sum": _sum(self._grd_total_ms),
            "grd_total_ms_mean": _mean(self._grd_total_ms),
            "grd_feature_reduce_ms_mean": _mean(self._grd_reduce_ms),
            "grd_correspondence_sim_ms_mean": _mean(self._grd_corr_ms),
            "grd_last_stats": self._grd_stats[-1] if self._grd_stats else None,
            "grd_num_records": len(self._grd_stats),
            "gpu_memory": mem,
            "per_sample": self._samples,
            "metrics_disclaimer": (
                "These are wall-clock / CUDA-event measurements, not FLOPs. "
                "compress_wall_ms wraps the full update_kv() compressor when "
                "--runtime_metrics is on. Enabling metrics with CUDA synchronize "
                "on the generate probe perturbs absolute latency — compare methods "
                "under identical metrics settings, warm-up, and repeats."
            ),
        }


RUNTIME_METRICS = RuntimeMetricsTracker()


def reset_runtime_metrics():
    RUNTIME_METRICS.reset()


def collect_runtime_metrics(device: str = "cuda") -> Optional[Dict[str, Any]]:
    return RUNTIME_METRICS.get_summary(device=device)


def timed_generate(model, *, synchronize_cuda: bool = True, record: bool = True, **generate_kwargs):
    """Wrap ``model.generate`` with TTFT / ITL probing.

    Injects a logits processor that timestamps each step after an optional
    ``torch.cuda.synchronize()``. Returns ``(outputs, latency_dict)``.
    """
    probe = _LatencyProbe(synchronize_cuda=synchronize_cuda)
    existing = generate_kwargs.pop("logits_processor", None)
    if existing is None:
        lps = LogitsProcessorList([probe])
    elif isinstance(existing, LogitsProcessorList):
        lps = existing
        lps.append(probe)
    else:
        lps = LogitsProcessorList(list(existing) + [probe])
    generate_kwargs["logits_processor"] = lps

    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    probe._t0 = t0
    outputs = model.generate(**generate_kwargs)
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency = probe.summary()
    latency["e2e_generate_s"] = round(t1 - t0, 6)
    latency["synchronize_cuda"] = bool(synchronize_cuda)
    if record:
        RUNTIME_METRICS.record_sample(latency)
    return outputs, latency


class CudaTimer:
    """Context manager: wall ms + optional CUDA event ms for a code region."""

    def __init__(self, use_cuda_events: bool = True):
        self.use_cuda_events = bool(use_cuda_events) and torch.cuda.is_available()
        self.wall_ms = 0.0
        self.cuda_ms: Optional[float] = None
        self._t0 = 0.0
        self._start_evt = None
        self._end_evt = None

    def __enter__(self):
        if self.use_cuda_events:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wall_ms = (time.perf_counter() - self._t0) * 1000.0
        if self.use_cuda_events and self._start_evt is not None and self._end_evt is not None:
            self._end_evt.record()
            self._end_evt.synchronize()
            self.cuda_ms = float(self._start_evt.elapsed_time(self._end_evt))
        return False


__all__ = [
    "RuntimeMetricsTracker",
    "RUNTIME_METRICS",
    "reset_runtime_metrics",
    "collect_runtime_metrics",
    "timed_generate",
    "CudaTimer",
]
