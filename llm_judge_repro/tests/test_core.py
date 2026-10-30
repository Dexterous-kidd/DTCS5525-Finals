import numpy as np
import pandas as pd
import pytest

from judgestudy.checks import lcs_len, overlap
from judgestudy.metrics import cohen_kappa, holm, krippendorff_alpha_nominal, mcnemar_exact
from judgestudy.probit import category_probs, fit_probit, fit_probit_re, predict_cell
from judgestudy.prompts import parse_shown, render, to_canonical
from judgestudy.records import PairData
from judgestudy.sim import PROTOS, battle_vals, decide, gen_sets, k_battles, n_calls, simulate_z, zscore


# ---------- metrics: hand-worked example (2x2 table [[20,5],[10,15]]) ----------
R1 = ["A"] * 25 + ["B"] * 25
R2 = ["A"] * 20 + ["B"] * 5 + ["A"] * 10 + ["B"] * 15


def test_kappa_hand_worked():
    assert cohen_kappa(R1, R2) == pytest.approx(0.4)


def test_alpha_hand_worked():
    assert krippendorff_alpha_nominal(np.stack([R1, R2], 1)) == pytest.approx(0.4)


def test_holm():
    adj, rej = holm([0.01, 0.04, 0.03, 0.2])
    assert adj.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2]) and rej.tolist() == [True, False, False, False]


def test_mcnemar():
    r = mcnemar_exact([1] * 10 + [0] * 2, [0] * 10 + [1] * 2)
    assert r["only_a"] == 10 and r["only_b"] == 2 and 0 < r["p"] < 0.1


# ---------- contamination overlap ----------
def test_lcs():
    assert lcs_len("a b c d".split(), "a x c d".split()) == 3
    assert overlap("the quick brown fox", "the quick brown fox") == 1.0
    assert overlap("", "the quick") == 0.0


# ---------- prompts / parsing ----------
def test_parse_and_canonical():
    assert parse_shown("blah [[A]] more [[B]]") == "B"            # last tag wins
    assert parse_shown("no verdict here") is None
    assert to_canonical("A", "AB") == "A" and to_canonical("A", "BA") == "B" and to_canonical("C", "BA") == "T"
    assert to_canonical(None, "AB") == "X"


def test_render_swaps_answers():
    cfg = {"prompts": {"single": "prompts/pairwise_single.txt", "multi": "prompts/pairwise_multi.txt"}}
    pay = {"kind": "single", "question": "Q?", "resp_A": "AAA", "resp_B": "BBB"}
    ab = render(pay, "AB", cfg)[0]["content"]; ba = render(pay, "BA", cfg)[0]["content"]
    assert ab.index("AAA") < ab.index("BBB") and ba.index("BBB") < ba.index("AAA")
    multi = {"kind": "multi", "q1": "q1", "q2": "q2", "a1_A": "xA1", "a2_A": "xA2", "a1_B": "xB1", "a2_B": "xB2"}
    m = render(multi, "BA", cfg)[0]["content"]
    assert m.index("xB2") < m.index("xA2")


# ---------- simulation ----------
def _S(m=128, p=(0.3, 0.2, 0.5), seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([0, 0.5, 1], size=(m, 2, 5), p=list(p)).astype(float)


def test_budgets():
    assert k_battles("P1", 64) == 32 and k_battles("P0r", 64) == 64 and n_calls("P1x", 64) == 128


def test_zscore_and_decide():
    z = zscore(np.array([[1, 1, 1, 0.0], [0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]]))
    assert z[0] > 0 and z[1] == 0 and abs(z[2]) < 1e-9
    d = decide(np.array([-3.0, -0.5, 0.0, 0.5, 3.0]), 1.0)
    assert d.tolist() == [-1, 0, 0, 0, 1]
    assert decide(np.array([0.0, 0.1]), 0.0).tolist() == [0, 1]       # z*=0 removes the tie band, but exact 0.5 stays undecided


def test_replicates_use_disjoint_generations():
    g1, g2 = gen_sets(np.random.default_rng(0), 500)
    assert all(len(set(a) | set(b)) == 4 for a, b in zip(g1, g2))


def test_p1_agreement_rule():
    S = np.zeros((6, 2, 5)); S[:, 0, :] = 1.0; S[:, 1, :] = 1.0; S[3:, 1, :] = 0.0          # first 3 battles agree on A, last 3 disagree
    idx = np.tile(np.arange(6), (4, 1)); gs = gen_sets(np.random.default_rng(1), 4)
    v = battle_vals("P1", {"j": S}, "j", ["j"], idx, np.random.default_rng(1), np.zeros(6, int), "sampled", gs[0])
    assert (v[:, :3] == 1).all() and (v[:, 3:] == 0.5).all()


def test_p3_alternates_judges():
    S1 = np.ones((8, 2, 5)); S2 = np.zeros((8, 2, 5)); idx = np.tile(np.arange(8), (50, 1)); gs = gen_sets(np.random.default_rng(1), 50)
    v = battle_vals("P3", {"a": S1, "b": S2}, "rot", ["a", "b"], idx, np.random.default_rng(1), np.zeros(8, int), "sampled", gs[0])
    assert (v.mean(1) == 0.5).all() and ((v[:, 0::2] != v[:, 1::2]).all())


def test_simulate_returns_shared_subset_flips_zero_for_deterministic_record():
    S = np.ones((32, 2, 5))                                            # judge always says A in both orders
    r = simulate_z("P1", {"j": S}, "j", ["j"], 16, np.random.default_rng(0), 50, np.zeros(32, int))
    assert np.all(np.isinf(r[0])) and np.all(r[0] > 0) and np.all(decide(r[0], 1.0) == 1) and np.all(decide(r[1], 1.0) == 1)


# ---------- probit ----------
def test_probit_recovers_parameters():
    rng = np.random.default_rng(0); m = rng.uniform(-.3, .3, 8000); s = rng.choice([-1, 1], 8000)
    th = [4.0, 0.3, -0.4, np.log(0.7)]; pB, pT, pA = category_probs(th, m, s); u = rng.random(8000)
    y = np.where(u < pB, 0, np.where(u < pB + pT, 1, 2)); f = fit_probit(m, s, y)
    assert f["beta"] == pytest.approx(4.0, abs=0.5) and f["gamma"] == pytest.approx(0.3, abs=0.08) and f["converged"]


def test_probit_re_recovers_sigma():
    rng = np.random.default_rng(1); nu, nc = 1500, 6
    unit = np.repeat(np.arange(nu), nc); m = np.repeat(rng.uniform(-.3, .3, nu), nc); s = np.tile([1, -1, 1, -1, 1, -1], nu)
    ui = np.repeat(rng.normal(0, 0.8, nu), nc); th = [4.0, 0.3, -0.4, np.log(0.7)]
    pB, pT, pA = category_probs(th, m, s, shift=ui); u = rng.random(len(m)); y = np.where(u < pB, 0, np.where(u < pB + pT, 1, 2))
    f = fit_probit_re(m, s, y, unit)
    assert f["sigma_u"] == pytest.approx(0.8, abs=0.15)


def test_prediction_matches_simulation_without_battle_effects():
    """Independent-call judge: analytic flip-rate prediction should match the vectorised simulation."""
    rng = np.random.default_rng(3); th = [4.0, 0.25, -0.4, np.log(0.7)]; m_pair = 0.03; pool = 128
    s_ab = category_probs(th, m_pair, +1); s_ba = category_probs(th, m_pair, -1)
    S = np.zeros((pool, 2, 5))
    for o, p in enumerate((s_ab, s_ba)):
        S[:, o, :] = rng.choice([0, 0.5, 1], size=(pool, 5), p=[p[0], p[1], p[2]])
    zs = 0.8; D = 4000
    r = simulate_z("P0r", {"j": S}, "j", ["j"], 64, rng, D, np.zeros(pool, int))
    d1, d2 = decide(r[0], zs), decide(r[1], zs); both = (d1 != 0) & (d2 != 0)
    obs = ((d1 != d2) & both).sum() / both.sum()
    pred, cov = predict_cell("P0r", {"j": th}, m_pair, 64, zs, 64, 0.5, pool)
    assert abs(obs - pred) < 0.08                                      # sampling noise from the finite 128 x 4 record


# ---------- Ollama backend against a fake local server ----------
def test_ollama_backend_with_fake_server(monkeypatch):
    import json as _json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from judgestudy.judges import OllamaJudge
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, d):
            b = _json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send({"models": [{"name": "qwen3:8b", "digest": "abc123def456", "details": {"quantization_level": "Q4_K_M"}}]} if self.path == "/api/tags" else {"version": "0.0-test"})
        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                return self._send({"details": {"quantization_level": "Q4_K_M", "parameter_size": "8.2B", "format": "gguf"}})
            seen.append(body)
            self._send({"message": {"content": "reasoning... [[B]]"}, "done_reason": "stop", "prompt_eval_count": 321, "eval_count": 12})

    srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    spec = dict(name="qwen3-8b", ollama_model="qwen3:8b", think=False, max_model_len=8192, workers=3, host=f"http://127.0.0.1:{srv.server_port}",
                sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0})
    j = OllamaJudge(spec, {"max_tokens": 1024}); j.load()
    reqs = [dict(messages=[{"role": "user", "content": "x"}], temperature=t, seed=s, meta={}) for t, s in [(0, 0), (1, 11), (1, 44)]]
    out = j.generate(reqs); srv.shutdown()
    assert [parse_shown(o["text"]) for o in out] == ["B", "B", "B"] and out[0]["n_in"] == 321 and out[0]["n_out"] == 12
    assert j.header()["quantization"] == "Q4_K_M" and j.header()["revision"] == "abc123def456"
    by_seed = {b["options"]["seed"]: b for b in seen}
    assert all(b["options"]["num_ctx"] == 8192 and b["options"]["num_predict"] == 1024 and b["think"] is False and b["stream"] is False for b in seen)
    assert by_seed[0]["options"]["temperature"] == 0.0 and "top_k" not in by_seed[0]["options"]
    assert by_seed[11]["options"]["top_k"] == 20 and by_seed[11]["options"]["top_p"] == 0.8


def test_ollama_missing_model_message():
    from judgestudy.judges import OllamaJudge
    j = OllamaJudge(dict(name="x", ollama_model="nope:1b", max_model_len=8192, host="http://127.0.0.1:9", sampling={"temperature": 1, "top_p": 1}), {"max_tokens": 10})
    with pytest.raises(RuntimeError, match="cannot reach Ollama"):
        j.load()


def test_all_judges_skips_sensitivity(tmp_path):
    from judgestudy.tuning import all_judges
    for j in ("a", "b", "s"):
        (tmp_path / j).mkdir(); (tmp_path / j / "arena.shard0of1.jsonl").write_text("{}\n")
    cfg = {"paths": {"verdicts": str(tmp_path)}, "judges_cfg": {"judges": {"a": {"core": True}, "b": {"core": False}, "s": {"sensitivity": True}}}}
    assert all_judges(cfg, True, "arena") == ["a", "b"]
