"""Validity checks: determinism of temperature-zero calls, and a descriptive contamination probe."""
import json, logging
from pathlib import Path

import numpy as np
import pandas as pd

from .judges import make_judge
from .prompts import parse_shown, render, to_canonical

log = logging.getLogger("judgestudy")


def lcs_len(a, b):
    """Longest common subsequence length of two token lists (dynamic programming, O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def overlap(generated, truth):
    """Share of the true remainder's words recovered in order by the model's continuation."""
    g, t = generated.lower().split()[:300], truth.lower().split()[:300]
    return lcs_len(g, t) / len(t) if t else 0.0


def run_checks(ctx, judge_name, kinds=("determinism", "contamination"), mock=False):
    cfg = ctx.cfg; c = cfg["checks"]; rng = np.random.default_rng(cfg["seed"] + 11)
    proc = Path(cfg["paths"]["processed"])
    judge = make_judge(cfg, judge_name, mock=mock); judge.load()
    if "determinism" in kinds:
        u = pd.read_parquet(proc / "arena_units.parquet"); u = u[u.split == "dev"]
        pick = u.iloc[np.sort(rng.permutation(len(u))[:c["determinism_n"]])]
        reqs = [dict(messages=render(r.payload, "AB", cfg), temperature=0, seed=0, meta=dict(unit_id=r.unit_id, order="AB", m=0.0)) for r in pick.itertuples()]
        o1 = judge.generate(reqs)
        perm = rng.permutation(len(reqs))                                # different batch composition on the second pass
        o2 = [None] * len(reqs)
        for i, o in zip(perm, judge.generate([reqs[i] for i in perm])):
            o2[i] = o
        same_text = np.mean([a["text"] == b["text"] for a, b in zip(o1, o2)])
        same_verd = np.mean([parse_shown(a["text"]) == parse_shown(b["text"]) for a, b in zip(o1, o2)])
        ctx.record(f"check/determinism/{judge_name}", n=len(reqs), identical_text=float(same_text), identical_verdict=float(same_verd))
        if same_verd < 1:
            log.warning("[%s] temperature-zero verdicts differ in %.1f%% of repeats (batch-dependent GPU numerics); report this in Methods",
                        judge_name, 100 * (1 - same_verd))
    if "contamination" in kinds:
        rows = []
        specs = []
        ua = pd.read_parquet(proc / "arena_units.parquet")
        for r in ua.iloc[rng.permutation(len(ua))[:c["contamination_arena_n"]]].itertuples():
            specs.append(("arena", json.loads(r.payload)["question"]))
        if (proc / "mt_units.parquet").exists():
            um = pd.read_parquet(proc / "mt_units.parquet"); qs = sorted({json.loads(p).get("question") or json.loads(p).get("q1") for p in um.payload})
            for q in [qs[i] for i in rng.permutation(len(qs))[:c["contamination_mt_n"]]]:
                specs.append(("mt", q))
        reqs, truths, srcs = [], [], []
        for src, text in specs:
            w = text.split()[:400]
            if len(w) < 20:
                continue
            h = len(w) // 2
            prompt = "Continue the following text exactly as it would appear. Output only the continuation.\n\n" + " ".join(w[:h])
            reqs.append(dict(messages=[{"role": "user", "content": prompt}], temperature=0, seed=0, meta=dict(unit_id=f"c{len(reqs)}", order="AB", m=0.0)))
            truths.append(" ".join(w[h:])); srcs.append(src)
        outs = judge.generate(reqs)
        ov = np.array([overlap(o["text"], t) for o, t in zip(outs, truths)])
        df = pd.DataFrame(dict(source=srcs, overlap=ov))
        for s_, g in df.groupby("source"):
            ctx.record(f"check/contamination/{judge_name}/{s_}", n=len(g), share_above_threshold=float((g.overlap > c["contamination_threshold"]).mean()),
                       mean_overlap=float(g.overlap.mean()))
        ctx.table(df.groupby("source").overlap.describe().reset_index(), f"contamination_{judge_name}")
