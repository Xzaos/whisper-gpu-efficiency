"""Merge the per-stage json files into efficiency.json and render report.md.

A missing stage shows as `_` in the tables instead of being filled in.
"""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import pairwise_diffs, scaling_efficiency


def _read(raw: Path, name: str):
    p = raw / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build(raw: str | Path) -> dict:
    raw = Path(raw)
    single = _read(raw, "single.json") or {}
    base, ddp = _read(raw, "ddp_single.json"), _read(raw, "ddp.json")
    eff = {"env": single.get("env"), "configs": single.get("configs", []), "max_batch": single.get("sweep", {}),
           "sweep_steps": single.get("sweep_steps")}

    scaling = {"single_gpu": base, "ddp": ddp, "efficiency": None}
    if base and ddp and base.get("status") == "ok" and ddp.get("status") == "ok":
        scaling["efficiency"] = scaling_efficiency(base["samples_per_s"], ddp["samples_per_s"], ddp["world"])
    eff["ddp_scaling"] = scaling

    runs = {k: _read(raw, f"equiv_{k}.json") for k in ("a", "b", "b_repeat", "c")}
    ok = {k: v["losses"] for k, v in runs.items() if v and v.get("status") == "ok"}
    eff["equivalence"] = {"runs": runs, "pairwise": pairwise_diffs(ok) if len(ok) > 1 else {}}
    return eff


def _f(x, spec):
    return "_" if x is None else format(x, spec)


def _row(r: dict) -> list[str]:
    if r.get("status") == "OOM":
        return [r.get("name", "_"), str(r.get("batch", "_")), "OOM", "OOM", "OOM", "OOM"]
    return [r["name"], str(r["batch"]), _f(r["samples_per_s"], ".1f"), _f(r["ms_per_step"], ".0f"), _f(r["peak_gb"], ".2f"), _f(r["loss"], ".3f")]


def _table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render(eff: dict) -> str:
    out = ["# Efficiency report", ""]
    env = eff.get("env")
    if env:
        out += [f"GPUs: {', '.join(env['gpus'])}. torch {env['torch']} (CUDA {env['cuda']}), transformers {env['transformers']}, peft {env.get('peft', '_')}.", ""]
    else:
        out += ["Environment: _ (the single-GPU stage has not run).", ""]

    head = ["config", "batch", "samples/s", "ms/step", "peak GB (GiB)", "loss at last step"]
    out += ["## Configs (10 warmup + 40 timed steps)", ""]
    out += [_table(head, [_row(r) for r in eff["configs"]]) if eff["configs"] else "_", ""]

    out += ["## Max batch size that fits", ""]
    steps = eff.get("sweep_steps")
    if steps:
        out += [f"Each batch size ran {steps['warmup']} warmup + {steps['timed']} timed steps, stopping at the first OOM.", ""]
    rows = []
    for name, s in eff["max_batch"].items():
        tried = ", ".join(f"{r['batch']}{'' if r['status'] == 'ok' else ' (OOM)'}" for r in s["rows"])
        top = next((r for r in s["rows"] if r["batch"] == s["max_batch"]), None)
        rows.append([name, str(s["max_batch"] or "none"), tried, _f(top and top["samples_per_s"], ".1f"), _f(top and top["peak_gb"], ".2f")])
    out += [_table(["config", "max batch", "tried", "samples/s at max", "peak GB at max"], rows) if rows else "_", ""]

    sc = eff["ddp_scaling"]
    out += ["## DDP scaling (LoRA fp16)", ""]
    rows = []
    for label, r in [("1 GPU", sc["single_gpu"]), ("2 GPU DDP", sc["ddp"])]:
        if r is None:
            rows.append([label, "_", "_", "_", "_"])
        elif r.get("status") == "OOM":
            rows.append([label, "OOM", "OOM", "OOM", "OOM"])
        else:
            layout = f"{r['batch']} x {r['accum']} x {r['world']}"
            rows.append([label, f"{layout} = {r['global_batch']}", _f(r["samples_per_s"], ".1f"), _f(r["ms_per_step"], ".0f"), _f(r["peak_gb"], ".2f")])
    out += [_table(["run", "batch x accum x gpus = global", "samples/s", "ms/step", "peak GB"], rows), ""]
    if sc["single_gpu"] and sc["single_gpu"].get("fallback"):
        out += [f"Single-GPU baseline fell back to 8 x accum 2 ({sc['single_gpu']['fallback']}).", ""]
    out += [f"Scaling efficiency = throughput_2gpu / (2 x throughput_1gpu): {_f(sc['efficiency'], '.3f')}", ""]

    out += ["## Equivalence check (LoRA fp16, dropout off, same data order, 30 steps)", ""]
    eq = eff["equivalence"]
    rows = []
    for k, label in [("a", "a: 1 GPU, batch 16"), ("b", "b: 1 GPU, batch 8 x accum 2"), ("b_repeat", "b rerun (noise floor)"), ("c", "c: DDP 2 x 8")]:
        r = eq["runs"].get(k)
        if r is None:
            rows.append([label, "_", "_", "_"])
        elif r.get("status") == "OOM":
            rows.append([label, "OOM", "_", "_"])
        else:
            rows.append([label, f"{r['losses'][0]:.4f}", f"{r['losses'][-1]:.4f}", str(r["skipped_steps"])])
    out += [_table(["run", "loss step 1", "loss step 30", "fp16 skipped steps"], rows), ""]
    if eq["pairwise"]:
        rows = [[k, f"{v['max_abs_diff']:.2e}", str(v["at_step"])] for k, v in eq["pairwise"].items()]
        out += [_table(["pair", "max per-step abs loss diff", "at step"], rows), ""]
    else:
        out += ["Max per-step loss difference: _", ""]
    return "\n".join(out)
