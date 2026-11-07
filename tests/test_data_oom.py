import pytest
import torch

from wge.data import global_slice, make_order, micro_batches, pad_labels
from wge.oom import guarded


def test_order_is_seeded_and_covers_each_epoch():
    a, b = make_order(10, 25, seed=3), make_order(10, 25, seed=3)
    assert torch.equal(a, b)
    assert sorted(a[:10].tolist()) == list(range(10))
    assert not torch.equal(a, make_order(10, 25, seed=4))


def test_layouts_consume_same_samples():
    order = make_order(64, 32, seed=0)
    idx = global_slice(order, 1, 16)
    single = micro_batches(idx, 16, 1, 0, 1)
    accum = micro_batches(idx, 8, 2, 0, 1)
    ddp = micro_batches(idx, 8, 1, 0, 2) + micro_batches(idx, 8, 1, 1, 2)
    for layout in (single, accum, ddp):
        assert sorted(torch.cat(layout).tolist()) == sorted(idx.tolist())


def test_pad_labels_truncates_and_counts():
    t, cut = pad_labels([[1, 2, 3], [4, 5, 6, 7, 8]], max_len=4)
    assert cut == 1
    assert t.tolist() == [[1, 2, 3, -100], [4, 5, 6, 7]]


def test_guarded_turns_oom_into_result():
    def boom():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    assert guarded(boom) == {"status": "OOM"}


def test_guarded_passes_result_and_other_errors():
    assert guarded(lambda: {"x": 1}) == {"status": "ok", "x": 1}

    def bad():
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        guarded(bad)
