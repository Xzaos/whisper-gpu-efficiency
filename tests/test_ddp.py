import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from wge.metrics import max_abs_diff
from wge.train import Cfg, equiv_run

CPU = torch.device("cpu")


def _worker(rank, world, init_file, tiny_config, tiny_data, out):
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)
    r = equiv_run(Cfg("ddp", mode="lora", batch=4), tiny_data, CPU, steps=5, rank=rank, world=world, whisper_config=tiny_config)
    if rank == 0:
        torch.save(r["losses"], out)
    dist.destroy_process_group()


def test_ddp_gloo_matches_single_process(tiny_config, tiny_data):
    single = equiv_run(Cfg("s", mode="lora", batch=8), tiny_data, CPU, steps=5, whisper_config=tiny_config)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ddp.pt")
        init = os.path.join(d, "init").replace("\\", "/")
        mp.spawn(_worker, args=(2, init, tiny_config, tiny_data, out), nprocs=2, join=True)
        ddp = torch.load(out)
    assert max_abs_diff(single["losses"], ddp)[0] < 1e-4
