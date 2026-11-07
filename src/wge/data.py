"""Precomputed Whisper features and the fixed sample order.

Log-mel features and label ids are computed once and saved to one tensor
file, so every benchmark step measures the GPU and not audio decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

MODEL_ID = "openai/whisper-small"


@dataclass
class Data:
    features: torch.Tensor  # (N, n_mels, frames) float32
    labels: torch.Tensor    # (N, L) long, -100 for padding
    ids: list[str]
    meta: dict

    def __len__(self):
        return self.features.shape[0]


def make_order(n: int, count: int, seed: int) -> torch.Tensor:
    """First `count` indices of back-to-back seeded permutations of range(n)."""
    g = torch.Generator().manual_seed(seed)
    reps = -(-count // n)
    return torch.cat([torch.randperm(n, generator=g) for _ in range(reps)])[:count]


def global_slice(order: torch.Tensor, step: int, global_batch: int) -> torch.Tensor:
    return order[step * global_batch:(step + 1) * global_batch]


def micro_batches(idx: torch.Tensor, batch: int, accum: int, rank: int, world: int) -> list[torch.Tensor]:
    """Split one global step into this rank's micro-batches.

    Micro-batch j of rank r is block (j * world + r) of the global slice, so
    every layout (single, accumulation, DDP) consumes the same samples.
    """
    assert len(idx) == batch * accum * world, (len(idx), batch, accum, world)
    return [idx[(j * world + rank) * batch:(j * world + rank + 1) * batch] for j in range(accum)]


def pad_labels(seqs: list[list[int]], max_len: int) -> tuple[torch.Tensor, int]:
    """Right-pad with -100, truncating to max_len. Returns the tensor and the truncated count."""
    cut = sum(len(s) > max_len for s in seqs)
    width = min(max_len, max(len(s) for s in seqs))
    out = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:width]
        out[i, :len(s)] = torch.tensor(s)
    return out, cut


def build(manifest: str, out: str, model_id: str = MODEL_ID, max_label_len: int = 224) -> dict:
    import soundfile as sf
    from asrbench.data import read_manifest
    from transformers import WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model_id, language="hindi", task="transcribe")
    rows = read_manifest(manifest)
    feats, seqs = [], []
    sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    for r in rows:
        audio, _ = sf.read(r["audio"], dtype="float32")
        feats.append(proc.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt").input_features[0])
        ids = proc.tokenizer(r["text"]).input_ids
        # the model prepends <|startoftranscript|> itself when it shifts labels right
        seqs.append(ids[1:] if ids[0] == sot else ids)
    labels, cut = pad_labels(seqs, max_label_len)
    meta = {"n": len(rows), "labels_truncated": cut, "model_id": model_id, "manifest": manifest}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": torch.stack(feats), "labels": labels, "ids": [r["id"] for r in rows], "meta": meta}, out)
    return meta


def load(path: str) -> Data:
    d = torch.load(path)
    return Data(d["features"], d["labels"], d["ids"], d["meta"])
