"""One training loop for every config: full fine-tune or LoRA, fp32 or fp16
autocast, optional gradient checkpointing, gradient accumulation and DDP.

The loss is a sum over tokens divided by the token count of the whole global
batch. That makes batch 16, 8 x accum 2 and DDP 2 x 8 the same gradient up to
floating-point noise; averaging per-micro-batch means would not be, because
micro-batches hold different numbers of tokens.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .diagnose import Diag, gather_ranks
from .data import Data, global_slice, make_order, micro_batches
from .metrics import summarise

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


@dataclass
class Cfg:
    name: str
    mode: str = "full"      # "full" or "lora"
    amp: bool = False       # fp16 autocast + GradScaler
    ckpt: bool = False      # gradient checkpointing
    batch: int = 8          # per GPU, per micro-batch
    accum: int = 1
    lr: float | None = None
    init_scale: float = 65536.0  # GradScaler starting loss scale

    @property
    def learning_rate(self):
        return self.lr if self.lr is not None else (5e-4 if self.mode == "lora" else 1e-5)


def make_model(cfg: Cfg, seed: int = 0, whisper_config=None, model_id: str = "openai/whisper-small", lora_r: int = 32):
    from transformers import WhisperForConditionalGeneration

    torch.manual_seed(seed)
    if whisper_config is None:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0)
    else:
        model = WhisperForConditionalGeneration(whisper_config)
    model.config.use_cache = False
    if cfg.mode == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, target_modules=LORA_TARGETS))
    if cfg.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def start_token(model) -> int:
    m = model.module if hasattr(model, "module") else model
    return m.config.decoder_start_token_id


def loss_sum(model, feats: torch.Tensor, labels: torch.Tensor, start_id: int) -> torch.Tensor:
    dec_in = torch.cat([torch.full_like(labels[:, :1], start_id), labels[:, :-1]], dim=1)
    dec_in = dec_in.masked_fill(dec_in == -100, 0)
    logits = model(input_features=feats, decoder_input_ids=dec_in).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")


def make_scaler(enabled: bool, init_scale: float = 65536.0):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", init_scale=init_scale, enabled=enabled)
    return torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)


def train_step(model, opt, scaler, micro, total_tokens: int, world: int, amp: bool, device, params, diag=None) -> torch.Tensor:
    """One optimizer step over this rank's micro-batches. Returns this rank's
    share of the global loss (local token-loss sum / global token count)."""
    start = start_token(model)
    share = torch.zeros((), device=device)
    for j, (feats, labels) in enumerate(micro):
        last = j == len(micro) - 1
        # skip the gradient all-reduce until the last micro-batch under DDP
        sync = contextlib.nullcontext() if (last or not hasattr(model, "no_sync")) else model.no_sync()
        with sync:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                s = loss_sum(model, feats, labels, start)
            # DDP averages gradients over ranks, so scale by world to get the sum
            scaler.scale(s * world / total_tokens).backward()
        share += s.detach() / total_tokens
    if diag:
        diag.after_backward()
    scaler.unscale_(opt)
    if diag:
        diag.after_unscale(scaler, opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return share


def init_cuda(device) -> torch.device:
    """Resolve a cuda device to an explicit index and initialise its context.

    Memory-stat calls raise "Invalid device argument" on a device that has not
    been touched yet, so select it and allocate a tiny tensor first.
    """
    if device.type != "cuda":
        return device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(idx)
    torch.zeros(1, device=f"cuda:{idx}")
    return torch.device("cuda", idx)


def reset_peak(device) -> torch.device:
    device = init_cuda(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return device


def peak_bytes(device) -> int:
    return torch.cuda.max_memory_allocated(init_cuda(device)) if device.type == "cuda" else 0


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_loop(model, cfg: Cfg, data: Data, *, steps: int, warmup: int = 0, seed: int = 0, rank: int = 0, world: int = 1,
               device=torch.device("cpu"), record: bool = False, diag=None) -> dict:
    """Run warmup untimed steps then `steps` timed ones.

    With record=True the global loss of every step is read back (one sync per
    step), which is for the equivalence check and not for timing.
    """
    import torch.distributed as dist

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    amp = cfg.amp and device.type == "cuda"
    scaler = make_scaler(amp, cfg.init_scale)
    gb = cfg.batch * cfg.accum * world
    order = make_order(len(data), (warmup + steps) * gb, seed)
    model.train()

    losses, skipped, share, t0 = [], [], None, None
    for s in range(warmup + steps):
        if s == warmup:
            _sync(device)
            t0 = time.perf_counter()
        idx = global_slice(order, s, gb)
        total_tokens = int((data.labels[idx] != -100).sum())
        mbs = micro_batches(idx, cfg.batch, cfg.accum, rank, world)
        micro = [(data.features[m].to(device), data.labels[m].to(device)) for m in mbs]
        scale_before = scaler.get_scale() if (record and scaler.is_enabled()) else None
        if diag:
            diag.begin_step()
        share = train_step(model, opt, scaler, micro, total_tokens, world, amp, device, params, diag)
        if diag:
            diag.end_step(s + 1, scale_before, share, [[data.ids[i] for i in m.tolist()] for m in mbs])
        if record:
            g = share.clone()
            if world > 1:
                dist.all_reduce(g)
            losses.append(g.item())
            if scale_before is not None and scaler.get_scale() < scale_before:
                skipped.append(s + 1)
    _sync(device)
    elapsed = time.perf_counter() - t0
    if world > 1:
        dist.all_reduce(share)
    return {"elapsed": elapsed, "last_loss": share.item(), "losses": losses, "skipped_steps": len(skipped), "skipped_indices": skipped, "samples_per_step": gb}


def _wrap(model, device, world):
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    return model


def bench_config(cfg: Cfg, data: Data, device, *, warmup: int = 10, timed: int = 40, seed: int = 0, rank: int = 0, world: int = 1,
                 whisper_config=None) -> dict:
    """Build a fresh model, train warmup + timed steps, return the metric row."""
    device = reset_peak(device)
    model = _wrap(make_model(cfg, seed, whisper_config).to(device), device, world)
    out = train_loop(model, cfg, data, steps=timed, warmup=warmup, seed=seed, rank=rank, world=world, device=device)
    peak = peak_bytes(device)
    row = summarise(out["elapsed"], timed, out["samples_per_step"], peak, out["last_loss"])
    return {"name": cfg.name, "mode": cfg.mode, "amp": cfg.amp, "ckpt": cfg.ckpt, "batch": cfg.batch, "accum": cfg.accum,
            "world": world, "global_batch": out["samples_per_step"], "warmup": warmup, "timed": timed, **row}


def equiv_run(cfg: Cfg, data: Data, device, *, steps: int = 30, seed: int = 0, rank: int = 0, world: int = 1, whisper_config=None,
              diagnose: bool = False) -> dict:
    device = init_cuda(device)
    model = make_model(cfg, seed, whisper_config).to(device)
    diag = Diag({n: p for n, p in model.named_parameters() if p.requires_grad}) if diagnose else None
    model = _wrap(model, device, world)
    out = train_loop(model, cfg, data, steps=steps, warmup=0, seed=seed, rank=rank, world=world, device=device, record=True, diag=diag)
    extra = {"diag": {"ranks": gather_ranks(diag, world)}} if diag else {}
    return {**extra, "name": cfg.name, "global_batch": out["samples_per_step"], "losses": out["losses"], "skipped_steps": out["skipped_steps"],
            "skipped_indices": out["skipped_indices"], "init_scale": cfg.init_scale}
