"""Turn raw timings into the numbers the report shows, and compare loss curves.

Kept free of torch so it is cheap to test and to reuse in the report step.
"""
from __future__ import annotations

GIB = 2**30


def summarise(elapsed_s: float, timed_steps: int, samples_per_step: int, peak_bytes: int, last_loss: float) -> dict:
    return {
        "samples_per_s": samples_per_step * timed_steps / elapsed_s,
        "ms_per_step": 1000 * elapsed_s / timed_steps,
        "peak_gb": peak_bytes / GIB,
        "loss": last_loss,
    }


def scaling_efficiency(single_samples_per_s: float, ddp_samples_per_s: float, world: int = 2) -> float:
    return ddp_samples_per_s / (world * single_samples_per_s)


def max_abs_diff(a: list[float], b: list[float]) -> tuple[float, int]:
    """Largest per-step absolute difference and the step (0-based) where it happens."""
    if len(a) != len(b):
        raise ValueError(f"loss curves differ in length: {len(a)} vs {len(b)}")
    diffs = [abs(x - y) for x, y in zip(a, b)]
    k = max(range(len(diffs)), key=diffs.__getitem__)
    return diffs[k], k


def pairwise_diffs(runs: dict[str, list[float]]) -> dict[str, dict]:
    """Max per-step loss difference for every pair of runs, keyed 'a_vs_b'."""
    names = list(runs)
    out = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d, step = max_abs_diff(runs[x], runs[y])
            out[f"{x}_vs_{y}"] = {"max_abs_diff": d, "at_step": step + 1}
    return out


def largest_fitting(rows: list[dict]) -> int | None:
    ok = [r["batch"] for r in rows if r.get("status") == "ok"]
    return max(ok) if ok else None
