"""Run a benchmark function and turn a CUDA OOM into a result instead of a crash.

The cleanup happens after the except block so the traceback no longer holds
the model and optimizer when the cache is emptied.
"""
from __future__ import annotations

import gc

import torch


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def guarded(fn, *args, **kwargs) -> dict:
    oom = False
    result = None
    try:
        result = fn(*args, **kwargs)
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        oom = True
    free_gpu()
    return {"status": "OOM"} if oom else {"status": "ok", **result}
