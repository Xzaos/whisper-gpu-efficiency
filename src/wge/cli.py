"""Command line: prepare features, run each benchmark stage, build the report.

Stages write small json files to results/raw, and `report` merges them. The
ddp-* commands are meant to run under torchrun, one process per GPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
import json
from pathlib import Path

import torch

from .metrics import largest_fitting, scaling_efficiency
from .oom import guarded
from .train import Cfg, bench_config, equiv_run, init_cuda

RAW = Path("results/raw")
SWEEP = [4, 8, 16, 32, 64]
LORA_FP16 = dict(mode="lora", amp=True)

MAIN = [
    Cfg("1 full fp32", batch=8),
    Cfg("2 full fp16", amp=True, batch=8),
    Cfg("3 full fp16 + ckpt", amp=True, ckpt=True, batch=8),
    Cfg("4 lora fp16", batch=8, **LORA_FP16),
]
SWEEP_BASES = [
    Cfg("2 full fp16", amp=True),
    Cfg("3 full fp16 + ckpt", amp=True, ckpt=True),
    Cfg("4 lora fp16", **LORA_FP16),
]


def dump(name: str, obj) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("wrote", RAW / name, flush=True)


def env_info() -> dict:
    import transformers

    info = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        pass
    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def cuda_device(local_rank: int = 0):
    return init_cuda(torch.device("cuda", local_rank))


def with_batch(cfg: Cfg, batch: int, accum: int = 1, name: str | None = None) -> Cfg:
    return Cfg(name or cfg.name, cfg.mode, cfg.amp, cfg.ckpt, batch, accum, cfg.lr)


def cmd_prepare(a):
    from asrbench.data import prepare_hf

    from .data import build

    manifest = prepare_hf("google/fleurs", "hi_in", "train", "hi", "raw_transcription", "data", a.n, seed=a.seed, tag="train_hi_bench")
    print(build(str(manifest), a.out))


def cmd_bench(a):
    from .data import load

    data, dev = load(a.data), cuda_device()
    rows = []
    for cfg in MAIN:
        print("run", cfg.name, flush=True)
        rows.append({**guarded(bench_config, cfg, data, dev, warmup=a.warmup, timed=a.timed), "name": cfg.name, "batch": cfg.batch})
    sweeps = {}
    for base in SWEEP_BASES:
        out = []
        for b in SWEEP:
            print("sweep", base.name, b, flush=True)
            r = guarded(bench_config, with_batch(base, b), data, dev, warmup=2, timed=4)
            out.append({**r, "name": base.name, "batch": b})
            if r["status"] == "OOM":
                break
        sweeps[base.name] = {"rows": out, "max_batch": largest_fitting(out)}
    dump("single.json", {"env": env_info(), "configs": rows, "sweep": sweeps, "sweep_steps": {"warmup": 2, "timed": 4}})


def cmd_baseline(a):
    from .data import load

    data, dev = load(a.data), cuda_device()
    cfg = Cfg("1 gpu lora fp16", batch=16, **LORA_FP16)
    r = guarded(bench_config, cfg, data, dev, warmup=a.warmup, timed=a.timed)
    if r["status"] == "OOM":
        cfg = Cfg("1 gpu lora fp16 8x2", batch=8, accum=2, **LORA_FP16)
        r = {**guarded(bench_config, cfg, data, dev, warmup=a.warmup, timed=a.timed), "fallback": "batch 16 OOM"}
    dump("ddp_single.json", {**r, "name": cfg.name, "batch": cfg.batch, "accum": cfg.accum})


def init_dist():
    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), cuda_device(local)


def cmd_ddp_bench(a):
    import torch.distributed as dist

    from .data import load

    rank, world, dev = init_dist()
    data = load(a.data)
    cfg = Cfg("ddp lora fp16", batch=8, **LORA_FP16)
    r = guarded(bench_config, cfg, data, dev, warmup=a.warmup, timed=a.timed, rank=rank, world=world)
    peaks = [None] * world
    dist.all_gather_object(peaks, r.get("peak_gb"))
    if rank == 0:
        dump("ddp.json", {**r, "peak_gb_per_rank": peaks})
    dist.destroy_process_group()


EQUIV = {
    "a": Cfg("a 1gpu b16", batch=16, **LORA_FP16),
    "b": Cfg("b 1gpu b8 x2", batch=8, accum=2, **LORA_FP16),
    "b_repeat": Cfg("b_repeat 1gpu b8 x2 (rerun)", batch=8, accum=2, **LORA_FP16),
}


def cmd_equiv(a):
    from .data import load

    r = guarded(equiv_run, EQUIV[a.run], load(a.data), cuda_device(), steps=a.steps)
    dump(f"equiv_{a.run}.json", r)


def cmd_ddp_equiv(a):
    import torch.distributed as dist

    from .data import load

    rank, world, dev = init_dist()
    r = guarded(equiv_run, Cfg("c ddp 2x8", batch=8, **LORA_FP16), load(a.data), dev, steps=a.steps, rank=rank, world=world)
    if rank == 0:
        dump("equiv_c.json", r)
    dist.destroy_process_group()


def cmd_report(a):
    from .report import build, render

    eff = build(RAW)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "efficiency.json").write_text(json.dumps(eff, indent=2), encoding="utf-8")
    (Path(a.out) / "report.md").write_text(render(eff), encoding="utf-8")
    print("wrote", Path(a.out) / "report.md")


def main(argv=None):
    p = argparse.ArgumentParser(prog="wge")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **flags):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        for k, v in flags.items():
            s.add_argument("--" + k.replace("_", "-"), default=v, type=type(v))
        return s

    add("prepare", cmd_prepare, n=400, seed=0, out="data/features.pt")
    for name, fn in [("bench", cmd_bench), ("baseline", cmd_baseline), ("ddp-bench", cmd_ddp_bench)]:
        add(name, fn, data="data/features.pt", warmup=10, timed=40)
    add("equiv", cmd_equiv, data="data/features.pt", steps=30).add_argument("--run", choices=list(EQUIV), required=True)
    add("ddp-equiv", cmd_ddp_equiv, data="data/features.pt", steps=30)
    add("report", cmd_report, out="results")
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
