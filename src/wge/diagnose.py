"""Per-step, per-rank logging for the fp16 skipped-step investigation.

For every optimizer step a rank records the GradScaler scale, whether and
where gradients went non-finite (on this rank before DDP's all-reduce, via a
tensor hook, and after backward returns), the micro-batch sample ids, and a
checksum of the LoRA weights so replica divergence is visible.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .metrics import max_abs_diff


def _finite_max(g: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(g), g.detach().abs(), torch.zeros((), device=g.device, dtype=g.dtype)).max()


def scaler_found_inf(scaler, opt):
    """The scaler's own found-inf flag after unscale_, or None if this torch hides it."""
    try:
        state = scaler._per_optimizer_states[id(opt)]["found_inf_per_device"]
        return any(float(v.item()) > 0 for v in state.values())
    except (AttributeError, KeyError):
        return None


class Diag:
    def __init__(self, named_params: dict[str, torch.nn.Parameter]):
        self.params = named_params
        self.steps: list[dict] = []
        self._pre_bad: dict[str, torch.Tensor] = {}
        self._pre_max: torch.Tensor | None = None
        self._post: dict = {}
        self._found_inf = None
        for name, p in named_params.items():
            p.register_hook(lambda g, n=name: self._on_grad(n, g))

    def _on_grad(self, name, g):
        bad = ~torch.isfinite(g).all()
        self._pre_bad[name] = bad if name not in self._pre_bad else self._pre_bad[name] | bad
        m = _finite_max(g)
        self._pre_max = m if self._pre_max is None else torch.maximum(self._pre_max, m)

    def begin_step(self):
        self._pre_bad, self._pre_max, self._post, self._found_inf = {}, None, {}, None

    def after_backward(self):
        bad, mx = [], 0.0
        for name, p in self.params.items():
            if p.grad is None:
                continue
            if not bool(torch.isfinite(p.grad).all()):
                bad.append(name)
            mx = max(mx, float(_finite_max(p.grad)))
        self._post = {"nonfinite": bad, "max_grad": mx}

    def after_unscale(self, scaler, opt):
        self._found_inf = scaler_found_inf(scaler, opt)

    def end_step(self, step: int, scale_before, share: torch.Tensor, sample_ids: list[list[str]]):
        w = [p.detach().double() for p in self.params.values()]
        self.steps.append({
            "step": step,
            "scale_before": scale_before,
            "loss_finite": bool(torch.isfinite(share)),
            "found_inf_scaler": self._found_inf,
            "found_inf_derived": bool(self._post["nonfinite"]),
            "max_grad_pre_reduce": 0.0 if self._pre_max is None else float(self._pre_max),
            "nonfinite_pre_reduce": sorted(n for n, b in self._pre_bad.items() if bool(b)),
            "max_grad_post_reduce": self._post["max_grad"],
            "nonfinite_post_reduce": self._post["nonfinite"],
            "sample_ids": sample_ids,
            "weight_sum": float(sum(x.sum() for x in w)),
            "weight_abs_sum": float(sum(x.abs().sum() for x in w)),
        })


def gather_ranks(diag: Diag, world: int) -> list[list[dict]]:
    if world == 1:
        return [diag.steps]
    import torch.distributed as dist

    out = [None] * world
    dist.all_gather_object(out, diag.steps)
    return out


# ---- analysis of the saved runs -> results/diagnose.md ----

RUNS = [("c_fp16", "c: DDP 2 x 8, fp16"), ("b_fp16", "b: 1 GPU, 8 x accum 2, fp16"),
        ("b_fp32", "b: 1 GPU, 8 x accum 2, fp32"), ("c_fp32", "c: DDP 2 x 8, fp32")]


def _load(raw: Path, key: str):
    p = raw / f"diag_{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def inf_events(run: dict) -> list[dict]:
    """Every (step, rank) where gradients were non-finite, and whether that showed up before the all-reduce."""
    ev = []
    for rank, steps in enumerate(run["diag"]["ranks"]):
        for r in steps:
            if r["nonfinite_pre_reduce"] or r["nonfinite_post_reduce"] or r["found_inf_scaler"] or r["found_inf_derived"]:
                ev.append({"step": r["step"], "rank": rank, "scale": r["scale_before"], "pre": r["nonfinite_pre_reduce"],
                           "post": r["nonfinite_post_reduce"], "scaler_found_inf": r["found_inf_scaler"], "loss_finite": r["loss_finite"]})
    return sorted(ev, key=lambda e: (e["step"], e["rank"]))


def replica_gap(run: dict) -> tuple[float, int | None]:
    """Largest |weight checksum| difference between ranks, and the first step it exceeds 1e-9."""
    ranks = run["diag"]["ranks"]
    if len(ranks) < 2:
        return 0.0, None
    worst, first = 0.0, None
    for a, b in zip(ranks[0], ranks[1]):
        d = max(abs(a["weight_sum"] - b["weight_sum"]), abs(a["weight_abs_sum"] - b["weight_abs_sum"]))
        worst = max(worst, d)
        if first is None and d > 1e-9:
            first = a["step"]
    return worst, first


def same_batches(c: dict, b: dict) -> bool:
    """Both DDP ranks together see exactly the samples b sees at each step."""
    for step_b, *step_c in zip(b["diag"]["ranks"][0], *c["diag"]["ranks"]):
        want = sorted(i for m in step_b["sample_ids"] for i in m)
        got = sorted(i for sc in step_c for m in sc["sample_ids"] for i in m)
        if want != got:
            return False
    return True


def analyse(raw: str | Path) -> dict:
    raw = Path(raw)
    runs = {k: _load(raw, k) for k, _ in RUNS}
    ok = {k: r for k, r in runs.items() if r and r.get("status") == "ok"}
    out = {"runs": {k: ("missing" if runs[k] is None else runs[k].get("status")) for k in runs}, "inf_events": {}, "replica_gap": {}, "gaps": {}}
    for k, r in ok.items():
        out["inf_events"][k] = inf_events(r)
        out["replica_gap"][k] = replica_gap(r)
    if "b_fp16" in ok and "c_fp16" in ok:
        out["same_batches_fp16"] = same_batches(ok["c_fp16"], ok["b_fp16"])
    for tag in ("fp16", "fp32"):
        if f"b_{tag}" in ok and f"c_{tag}" in ok:
            d, step = max_abs_diff(ok[f"b_{tag}"]["losses"], ok[f"c_{tag}"]["losses"])
            out["gaps"][f"b_vs_c_{tag}"] = {"max_abs_diff": d, "at_step": step + 1}
    base, rep = raw / "equiv_b.json", raw / "equiv_b_repeat.json"
    if base.exists() and rep.exists():
        d, step = max_abs_diff(json.loads(base.read_text())["losses"], json.loads(rep.read_text())["losses"])
        out["gaps"]["fp16_noise_floor_b_vs_b_rerun"] = {"max_abs_diff": d, "at_step": step + 1}
    return out


def render(an: dict) -> str:
    out = ["# Diagnosis of the DDP skipped steps", "", "Run status: " + ", ".join(f"{k} {v}" for k, v in an["runs"].items()), ""]
    out += ["## Which steps and ranks found inf", ""]
    for k, ev in an["inf_events"].items():
        out.append(f"{k}: " + ("no non-finite gradients" if not ev else f"{len(ev)} (step, rank) events"))
        for e in ev:
            where = []
            if e["pre"]:
                where.append(f"before all-reduce on this rank: {len(e['pre'])} params")
            if e["post"]:
                where.append(f"after all-reduce: {len(e['post'])} params")
            if not where:
                where.append("flag only, no non-finite grad found")
            names = ", ".join((e["pre"] or e["post"])[:4])
            out.append(f"- step {e['step']}, rank {e['rank']}, scale {e['scale']}, loss finite {e['loss_finite']}, {'; '.join(where)}"
                       + (f" (e.g. {names})" if names else ""))
        out.append("")
    if not an["inf_events"]:
        out += ["_", ""]
    out += ["## Do the DDP replicas diverge", ""]
    for k, (gap, first) in an["replica_gap"].items():
        if k.startswith("c_"):
            out.append(f"- {k}: largest weight checksum difference between ranks {gap:.3e}" + (f", first above 1e-9 at step {first}" if first else ", never above 1e-9"))
    if "same_batches_fp16" in an:
        out += ["", f"Both DDP ranks together saw the same samples as b at every step (fp16): {an['same_batches_fp16']}"]
    out += ["", "## Loss gaps (max per-step absolute difference)", ""]
    out += ["| comparison | max abs diff | at step |", "|---|---|---|"]
    for k, v in an["gaps"].items():
        out.append(f"| {k} | {v['max_abs_diff']:.2e} | {v['at_step']} |")
    if not an["gaps"]:
        out.append("| _ | _ | _ |")
    return "\n".join(out) + "\n"
