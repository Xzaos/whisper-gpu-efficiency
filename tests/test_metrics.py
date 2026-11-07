import pytest

from wge.metrics import GIB, largest_fitting, max_abs_diff, pairwise_diffs, scaling_efficiency, summarise


def test_summarise():
    r = summarise(elapsed_s=20.0, timed_steps=40, samples_per_step=8, peak_bytes=3 * GIB, last_loss=1.5)
    assert r["samples_per_s"] == pytest.approx(16.0)
    assert r["ms_per_step"] == pytest.approx(500.0)
    assert r["peak_gb"] == pytest.approx(3.0)
    assert r["loss"] == 1.5


def test_scaling_efficiency():
    assert scaling_efficiency(10.0, 18.0) == pytest.approx(0.9)


def test_max_abs_diff_reports_step():
    d, k = max_abs_diff([1.0, 2.0, 3.0], [1.0, 2.5, 2.9])
    assert d == pytest.approx(0.5) and k == 1


def test_max_abs_diff_length_mismatch():
    with pytest.raises(ValueError):
        max_abs_diff([1.0], [1.0, 2.0])


def test_pairwise_diffs_keys_and_step_is_one_based():
    out = pairwise_diffs({"a": [1, 2], "b": [1, 3], "c": [1, 2]})
    assert set(out) == {"a_vs_b", "a_vs_c", "b_vs_c"}
    assert out["a_vs_b"] == {"max_abs_diff": 1, "at_step": 2}
    assert out["a_vs_c"]["max_abs_diff"] == 0


def test_largest_fitting():
    rows = [{"batch": 4, "status": "ok"}, {"batch": 8, "status": "ok"}, {"batch": 16, "status": "OOM"}]
    assert largest_fitting(rows) == 8
    assert largest_fitting([{"batch": 4, "status": "OOM"}]) is None
