"""Agreement metrics, hand-implemented from standard definitions (unit-tested in tests/test_metrics.py)."""
import itertools

import numpy as np
from scipy.stats import binomtest


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def krippendorff_alpha_nominal(ratings):
    """ratings: array (n_items, n_raters) of labels; no missing values."""
    r = np.asarray(ratings); n_items, m = r.shape
    cats = sorted(set(r.ravel().tolist()))
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for row in r:
        for x, y in itertools.permutations(range(m), 2):
            o[idx[row[x]], idx[row[y]]] += 1.0 / (m - 1)
    n = o.sum(); nc = o.sum(1)
    do = (o.sum() - np.trace(o)) / n
    de = (np.outer(nc, nc).sum() - (nc ** 2).sum()) / (n * (n - 1))
    return float(1 - do / de) if de > 0 else float("nan")


def mcnemar_exact(correct_a, correct_b):
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5).pvalue)
    return dict(only_a=n01, only_b=n10, p=p)


def holm(pvals, alpha=0.05):
    """Holm step-down. Returns adjusted p-values and reject flags."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj, adj <= alpha


def verdict_table_metrics(v, human):
    """v: DataFrame of verdict rows for one judge/split with columns unit_id, order, gen, verdict; human: Series unit_id -> label.
    Returns agreement (with ties and without), kappa, alpha, position consistency, test-retest."""
    import pandas as pd
    d = v[v.valid | (v.verdict != "X")].copy()
    d["h"] = d.unit_id.map(human)
    d = d[d.h.notna() & (d.verdict != "X")]
    out = {"n_calls": int(len(d))}
    out["agree_with_ties"] = float((d.verdict == d.h).mean())
    nt = d[(d.verdict != "T") & (d.h != "T")]
    out["agree_no_ties"] = float((nt.verdict == nt.h).mean()) if len(nt) else float("nan")
    out["kappa"] = cohen_kappa(d.verdict.to_numpy(), d.h.to_numpy())
    out["alpha"] = krippendorff_alpha_nominal(np.stack([d.verdict.to_numpy(), d.h.to_numpy()], 1))
    # position consistency: same canonical verdict in AB and BA (temperature-zero calls)
    t0 = v[(v.gen == 0) & (v.verdict != "X")].pivot_table(index="unit_id", columns="order", values="verdict", aggfunc="first").dropna()
    out["position_consistency"] = float((t0["AB"] == t0["BA"]).mean()) if len(t0) else float("nan")
    # test-retest: agreement of sampled generations (1..4) for the same unit and order
    s = v[(v.gen >= 1) & (v.verdict != "X")]
    agree = []
    for _, g in s.groupby(["unit_id", "order"]):
        lab = g.verdict.tolist()
        agree += [x == y for x, y in itertools.combinations(lab, 2)]
    out["test_retest"] = float(np.mean(agree)) if agree else float("nan")
    return out
