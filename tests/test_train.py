import pytest
import torch

from wge.data import global_slice, make_order, micro_batches
from wge.metrics import max_abs_diff
from wge.train import Cfg, bench_config, equiv_run, loss_sum, make_model, train_loop

CPU = torch.device("cpu")


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_training_step_changes_trainable_params_only(mode, tiny_config, tiny_data):
    cfg = Cfg("t", mode=mode, batch=4)
    model = make_model(cfg, whisper_config=tiny_config)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train_loop(model, cfg, tiny_data, steps=2, device=CPU)
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert changed
    if mode == "lora":
        assert all("lora" in n for n in changed)


def test_loss_decreases_on_repeated_data(tiny_config, tiny_data):
    cfg = Cfg("t", batch=8, lr=1e-3)
    r = equiv_run(cfg, tiny_data, CPU, steps=15, whisper_config=tiny_config)
    assert r["losses"][-1] < r["losses"][0]


def test_gradient_checkpointing_gives_same_loss(tiny_config, tiny_data):
    plain = equiv_run(Cfg("p", batch=4), tiny_data, CPU, steps=3, whisper_config=tiny_config)
    ckpt = equiv_run(Cfg("c", batch=4, ckpt=True), tiny_data, CPU, steps=3, whisper_config=tiny_config)
    assert max_abs_diff(plain["losses"], ckpt["losses"])[0] < 1e-4


def test_bench_config_row(tiny_config, tiny_data):
    row = bench_config(Cfg("t", mode="lora", batch=4), tiny_data, CPU, warmup=1, timed=2, whisper_config=tiny_config)
    assert row["samples_per_s"] > 0 and row["global_batch"] == 4 and row["timed"] == 2


def test_accumulation_matches_big_batch(tiny_config, tiny_data):
    a = equiv_run(Cfg("a", mode="lora", batch=8), tiny_data, CPU, steps=6, whisper_config=tiny_config)
    b = equiv_run(Cfg("b", mode="lora", batch=4, accum=2), tiny_data, CPU, steps=6, whisper_config=tiny_config)
    assert max_abs_diff(a["losses"], b["losses"])[0] < 1e-4


def test_naive_microbatch_mean_would_not_match(tiny_config, tiny_data):
    """Why the loss is divided by the global token count: per-micro-batch means differ."""
    model = make_model(Cfg("a"), whisper_config=tiny_config)
    idx = global_slice(make_order(len(tiny_data), 8, 0), 0, 8)
    start = model.config.decoder_start_token_id
    parts = [(tiny_data.features[m], tiny_data.labels[m]) for m in micro_batches(idx, 4, 2, 0, 1)]
    with torch.no_grad():
        sums = [loss_sum(model, f, l, start) for f, l in parts]
    toks = [int((l != -100).sum()) for _, l in parts]
    token_norm = sum(sums) / sum(toks)
    naive = sum(s / t for s, t in zip(sums, toks)) / 2
    assert toks[0] != toks[1]
    assert abs(float(token_norm - naive)) > 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memory_stats_on_fresh_device_path():
    from wge.train import peak_bytes, reset_peak

    dev = reset_peak(torch.device("cuda"))
    assert dev.index is not None
    x = torch.zeros(1024, 1024, device=dev)
    assert peak_bytes(torch.device("cuda")) >= x.numel() * 4


class FakeScaler:
    """Stands in for GradScaler and halves its scale on chosen update calls."""

    def __init__(self, skip_on):
        self.scale_value, self.calls, self.skip_on = 64.0, 0, skip_on

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale_value

    def scale(self, x):
        return x

    def unscale_(self, opt):
        pass

    def step(self, opt):
        opt.step()

    def update(self):
        self.calls += 1
        if self.calls in self.skip_on:
            self.scale_value /= 2


def test_skipped_step_indices_are_recorded(monkeypatch, tiny_config, tiny_data):
    monkeypatch.setattr("wge.train.make_scaler", lambda enabled, init_scale=65536.0: FakeScaler({2, 5}))
    r = equiv_run(Cfg("t", mode="lora", batch=4), tiny_data, CPU, steps=6, whisper_config=tiny_config)
    assert r["skipped_indices"] == [2, 5] and r["skipped_steps"] == 2


def test_init_scale_reaches_scaler(monkeypatch, tiny_config, tiny_data):
    seen = []
    monkeypatch.setattr("wge.train.make_scaler", lambda enabled, init_scale=65536.0: seen.append(init_scale) or FakeScaler(set()))
    r = equiv_run(Cfg("t", mode="lora", batch=4, init_scale=32768.0), tiny_data, CPU, steps=1, whisper_config=tiny_config)
    assert seen == [32768.0] and r["init_scale"] == 32768.0
