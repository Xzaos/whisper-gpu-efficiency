import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from wge.diagnose import Diag, analyse, inf_events, render, replica_gap
from wge.train import Cfg, equiv_run

CPU = torch.device("cpu")


def test_diag_records_fields_and_weights_change(tiny_config, tiny_data):
    r = equiv_run(Cfg("d", mode="lora", batch=4, accum=2), tiny_data, CPU, steps=3, whisper_config=tiny_config, diagnose=True)
    (steps,) = r["diag"]["ranks"]
    assert [s["step"] for s in steps] == [1, 2, 3]
    s = steps[0]
    assert s["loss_finite"] and s["nonfinite_pre_reduce"] == [] and s["nonfinite_post_reduce"] == []
    assert s["max_grad_pre_reduce"] > 0 and s["max_grad_post_reduce"] > 0
    assert len(s["sample_ids"]) == 2 and all(len(m) == 4 for m in s["sample_ids"])
    assert steps[0]["weight_sum"] != steps[1]["weight_sum"]
    json.dumps(r)


def test_nonfinite_grad_is_named_by_hook_and_post_check():
    lin = torch.nn.Linear(2, 1)
    diag = Diag(dict(lin.named_parameters()))
    diag.begin_step()
    lin(torch.tensor([[float("inf"), 1.0]])).sum().backward()
    diag.after_backward()
    diag.end_step(1, 65536.0, torch.tensor(1.0), [["x"]])
    s = diag.steps[0]
    assert s["nonfinite_pre_reduce"] == ["weight"] and s["nonfinite_post_reduce"] == ["weight"]
    assert s["found_inf_derived"] and s["found_inf_scaler"] is None
    assert s["max_grad_pre_reduce"] == 1.0


def _inf_worker(rank, world, init_file, out):
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    lin = torch.nn.Linear(2, 1)
    diag = Diag(dict(lin.named_parameters()))
    model = torch.nn.parallel.DistributedDataParallel(lin)
    x = torch.tensor([[float("inf"), 1.0]]) if rank == 1 else torch.tensor([[1.0, 1.0]])
    diag.begin_step()
    model(x).sum().backward()
    diag.after_backward()
    diag.end_step(1, 1.0, torch.tensor(1.0), [[str(rank)]])
    torch.save(diag.steps[0], f"{out}.{rank}")
    dist.destroy_process_group()


def test_inf_on_one_rank_shows_before_reduce_there_and_after_reduce_on_both():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r")
        mp.spawn(_inf_worker, args=(2, os.path.join(d, "init").replace("\\", "/"), out), nprocs=2, join=True)
        r0, r1 = torch.load(f"{out}.0"), torch.load(f"{out}.1")
    assert r0["nonfinite_pre_reduce"] == [] and r1["nonfinite_pre_reduce"] == ["weight"]
    assert r0["nonfinite_post_reduce"] == ["weight"] and r1["nonfinite_post_reduce"] == ["weight"]


def _ddp_worker(rank, world, init_file, tiny_config, tiny_data, out):
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)
    r = equiv_run(Cfg("c", mode="lora", batch=4), tiny_data, CPU, steps=4, rank=rank, world=world, whisper_config=tiny_config, diagnose=True)
    if rank == 0:
        torch.save(r, out)
    dist.destroy_process_group()


def test_ddp_ranks_see_same_samples_as_single_and_replicas_stay_in_sync(tiny_config, tiny_data):
    single = equiv_run(Cfg("b", mode="lora", batch=4, accum=2), tiny_data, CPU, steps=4, whisper_config=tiny_config, diagnose=True)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "c.pt")
        mp.spawn(_ddp_worker, args=(2, os.path.join(d, "init").replace("\\", "/"), tiny_config, tiny_data, out), nprocs=2, join=True)
        c = torch.load(out)
    assert len(c["diag"]["ranks"]) == 2
    for sb, s0, s1 in zip(single["diag"]["ranks"][0], *c["diag"]["ranks"]):
        want = sorted(i for m in sb["sample_ids"] for i in m)
        assert want == sorted(s0["sample_ids"][0] + s1["sample_ids"][0])
        assert not set(s0["sample_ids"][0]) & set(s1["sample_ids"][0])
    gap, first = replica_gap(c)
    assert gap < 1e-9 and first is None


def test_analyse_and_render(tmp_path):
    def step(i, pre=(), post=(), w=1.0, ids=(("a",), ("b",))):
        return {"step": i, "scale_before": 65536.0, "loss_finite": True, "found_inf_scaler": bool(post), "found_inf_derived": bool(post),
                "max_grad_pre_reduce": 1.0, "nonfinite_pre_reduce": list(pre), "max_grad_post_reduce": 1.0,
                "nonfinite_post_reduce": list(post), "sample_ids": [list(m) for m in ids], "weight_sum": w, "weight_abs_sum": w}

    def write(key, ranks, losses):
        (tmp_path / f"diag_{key}.json").write_text(json.dumps({"status": "ok", "losses": losses, "diag": {"ranks": ranks}}))

    write("c_fp16", [[step(1, ids=[("a",)]), step(2, pre=["p"], post=["p"], ids=[("a",)])], [step(1, ids=[("b",)]), step(2, post=["p"], w=1.5, ids=[("b",)])]], [1.0, 0.9])
    write("b_fp16", [[step(1), step(2)]], [1.0, 0.8])
    ev = inf_events(json.loads((tmp_path / "diag_c_fp16.json").read_text()))
    assert [(e["step"], e["rank"]) for e in ev] == [(2, 0), (2, 1)]
    an = analyse(tmp_path)
    assert an["same_batches_fp16"] is True and an["replica_gap"]["c_fp16"][1] == 2
    assert an["gaps"]["b_vs_c_fp16"]["at_step"] == 2
    text = render(an)
    assert "step 2, rank 0" in text and "before all-reduce" in text and "b_fp32 missing" in text
