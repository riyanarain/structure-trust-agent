import json

import numpy as np

from strustagent.agent import extract_json, parse_final, run_one
from strustagent.baselines import BASELINES
from strustagent.config import Config
from strustagent.evaluate import auroc, evaluate_method, paired_bootstrap
from strustagent.llm import CachedClient, MockClient
from strustagent.synthetic import make_synthetic
from strustagent.tools import ToolContext, parse_foldseek_m8, parse_regions

FINAL = json.dumps(
    {
        "action": "final",
        "segments": [{"start": 1, "end": 90, "trust": 0.9}, {"start": 91, "end": 104, "trust": 0.1}, {"start": 105, "end": 194, "trust": 0.9}],
        "domain_pairs": [{"a": "D1", "b": "D2", "placement_reliable": False, "confidence": 0.8}],
    }
)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('text ```json\n{"a": 2}\n``` more') == {"a": 2}
    assert extract_json("no json here") is None


def test_parse_final_accepts_words_reversed_pairs_and_ignores_bad_ranges():
    obj = {"segments": [{"start": 1, "end": 10, "trust": "high"}, {"start": 5, "end": 999, "trust": 0.5}], "domain_pairs": [{"a": "D2", "b": "D1", "placement_reliable": "no"}]}
    trust, j = parse_final(obj, 20, [("D1", "D2")])
    assert trust[0] == 0.9 and np.isnan(trust[15])
    assert j["D1-D2"]["reliable"] is False


def test_notools_run_scores_correctly_and_sends_images_only_in_image_condition():
    cfg = Config()
    p = make_synthetic(seed=0, misplace_deg=60.0)
    for cond, n_img in (("image", 2), ("text", 0), ("both", 2)):
        c = MockClient([FINAL])
        r = run_one(c, p, "notools", cond, cfg)
        assert r.error is None and r.n_model_calls == 1
        assert len(c.requests[0][1][0].images) == n_img
        assert "No tools are available" in c.requests[0][0]
        assert r.pair_judgments["D1-D2"]["reliable"] is False


def test_agent_calls_tool_then_answers_and_tool_errors_are_reported_back():
    cfg = Config()
    p = make_synthetic(seed=0)
    call = json.dumps({"action": "call_tool", "tool": "no_such_tool", "args": {}})
    c = MockClient([call, FINAL])
    r = run_one(c, p, "agent", "text", cfg, ToolContext(p, cfg))
    assert r.error is None and r.n_model_calls == 2
    assert r.tool_calls[0]["ok"] is False
    assert "Tool error" in c.requests[1][1][-1].text


def test_agent_is_forced_to_answer_after_max_steps():
    cfg = Config(max_steps=2)
    p = make_synthetic(seed=0)
    call = json.dumps({"action": "call_tool", "tool": "no_such_tool", "args": {}})
    c = MockClient([call, call, call])
    r = run_one(c, p, "agent", "text", cfg, ToolContext(p, cfg))
    assert r.error is not None  # mock never answers, but the loop must terminate
    assert r.n_model_calls <= cfg.max_steps + 1


def test_unparseable_reply_gets_one_retry_then_errors():
    cfg = Config()
    p = make_synthetic(seed=0)
    r = run_one(MockClient(["not json", "still not json"]), p, "notools", "text", cfg)
    assert r.error == "could not parse a JSON reply"


def test_baselines_and_evaluation_separate_good_from_bad_methods():
    cfg = Config()
    prots = {f"S{i}": make_synthetic(seed=i, misplace_deg=60.0 if i % 2 else 5.0, acc=f"S{i}") for i in range(8)}
    for p in prots.values():
        assert len(p.domains) >= 1
    pae = {k: BASELINES["pae_threshold"](p, cfg) for k, p in prots.items()}
    plddt = {k: BASELINES["plddt_threshold"](p, cfg) for k, p in prots.items()}
    e_pae, e_pl = evaluate_method(prots, pae, cfg), evaluate_method(prots, plddt, cfg)
    assert e_pae["pairs"]["accuracy"] >= 0.99  # synthetic PAE encodes the truth
    # pLDDT is high everywhere here, so it calls every misplaced pair reliable: the failure the PAE baseline avoids
    assert e_pl["pairs"]["accuracy"] == 0.0
    # (per-residue Spearman is not asserted: the synthetic truth has no local-quality signal to correlate with)


def test_auroc_and_bootstrap():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    a, b = np.array([0.8, 0.7, 0.9, 0.85]), np.array([0.5, 0.4, 0.6, 0.55])
    r = paired_bootstrap(a, b)
    assert r["ci_low"] > 0 and abs(r["mean_diff"] - 0.3) < 1e-9


def test_foldseek_parser_excludes_own_pdb_entries_and_late_releases():
    m8 = "q\t1abc_A\t0.9\t0.9\t0.9\t1.0\t1.0\t0.5\t1e-5\t100\nq\t2xyz_B\t0.8\t0.8\t0.8\t1.0\t1.0\t0.3\t1e-4\t90\nq\t3new_A\t0.7\t0.7\t0.7\t1.0\t1.0\t0.3\t1e-4\t80\n"
    hits = parse_foldseek_m8(m8, {"1abc"}, 5, {"2xyz": "2015-01-01", "3new": "2023-01-01"}, "2018-04-30")
    assert [h["target"] for h in hits] == ["2xyz_B"]


def test_parse_regions_validates():
    assert parse_regions({"regions": [[1, 10]]}, 50) == [(1, 10)]
    for bad in ({}, {"regions": [[0, 5]]}, {"regions": [[40, 60]]}):
        try:
            parse_regions(bad, 50)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_cache_makes_second_call_free(tmp_path):
    inner = MockClient(["first", "second"])
    c = CachedClient(inner, tmp_path)
    from strustagent.llm import Message

    m = [Message("user", "hi")]
    assert c.generate("s", m) == "first" and c.generate("s", m) == "first"
    assert c.calls == 1 and c.hits == 1


def test_allowed_methods_cover_rcsb_spellings():
    from strustagent.data import ALLOWED_METHODS

    assert {"X-RAY", "EM"} <= ALLOWED_METHODS and "NMR" not in ALLOWED_METHODS


def test_baseline_results_are_json_serializable():
    cfg = Config()
    p = make_synthetic(seed=0, misplace_deg=60.0)
    for name in BASELINES:
        json.dumps(BASELINES[name](p, cfg).to_json())


def test_disabled_tool_is_hidden_from_prompt_and_rejected():
    from strustagent.agent import system_prompt
    from strustagent.tools import run_tool

    cfg = Config(tools=["esm2_plausibility"])
    prompt = system_prompt(True, cfg.tools)
    assert "esm2_plausibility:" in prompt and "foldseek_support:" not in prompt
    p = make_synthetic(seed=0)
    try:
        run_tool("foldseek_support", ToolContext(p, cfg), {"regions": [[1, 10]]})
    except KeyError:
        return
    raise AssertionError("disabled tool was run")
