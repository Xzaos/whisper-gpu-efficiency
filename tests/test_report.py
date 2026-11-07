import json

from wge.report import build, render


def _write(raw, name, obj):
    (raw / name).write_text(json.dumps(obj), encoding="utf-8")


def test_empty_raw_renders_placeholders(tmp_path):
    text = render(build(tmp_path))
    assert "Scaling efficiency" in text and ": _" in text


def test_report_with_oom_and_scaling(tmp_path):
    ok = {"status": "ok", "name": "x", "batch": 8, "samples_per_s": 10.0, "ms_per_step": 800.0, "peak_gb": 5.0, "loss": 2.0,
          "accum": 1, "world": 1, "global_batch": 8}
    _write(tmp_path, "single.json", {
        "env": {"gpus": ["Tesla T4", "Tesla T4"], "torch": "2.x", "transformers": "4.x", "cuda": "12"},
        "configs": [ok, {"status": "OOM", "name": "y", "batch": 8}],
        "sweep": {"y": {"rows": [{**ok, "batch": 4}, {"status": "OOM", "batch": 8}], "max_batch": 4}},
        "sweep_steps": {"warmup": 2, "timed": 4},
    })
    _write(tmp_path, "ddp_single.json", {**ok, "batch": 16, "global_batch": 16, "samples_per_s": 20.0})
    _write(tmp_path, "ddp.json", {**ok, "world": 2, "global_batch": 16, "samples_per_s": 36.0})
    _write(tmp_path, "equiv_a.json", {"status": "ok", "losses": [1.0, 0.9], "skipped_steps": 0})
    _write(tmp_path, "equiv_b.json", {"status": "ok", "losses": [1.0, 0.91], "skipped_steps": 0})
    eff = build(tmp_path)
    assert eff["ddp_scaling"]["efficiency"] == 0.9
    assert eff["equivalence"]["pairwise"]["a_vs_b"]["max_abs_diff"] > 0.0099
    text = render(eff)
    assert "OOM" in text and "0.900" in text and "Tesla T4" in text


def test_equiv_name_and_extra_run_in_report(tmp_path):
    from wge.cli import equiv_name

    assert equiv_name("c", 65536.0) == "equiv_c.json"
    assert equiv_name("c", 32768.0) == "equiv_c_init32768.json"
    for name, last, skips in [("b", 0.5, []), ("c_init32768", 0.51, [3])]:
        _write(tmp_path, f"equiv_{name}.json", {"status": "ok", "losses": [1.0, last], "skipped_steps": len(skips), "skipped_indices": skips})
    eff = build(tmp_path)
    assert "b_vs_c_init32768" in eff["equivalence"]["pairwise"]
    assert "init scale 32768" in render(eff) and "(steps 3)" in render(eff)
