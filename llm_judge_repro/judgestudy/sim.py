"""Vectorised protocol simulation on the cached verdict record.

S[judge] has shape (m, 2, 5): battle x order(0=AB,1=BA) x gen(0=T0, 1..4 sampled), values in {0, .5, 1} (canonical).
All protocols draw from this record, so no new model calls are needed after generation.
"""
import numpy as np

PROTOS = ["P0", "P0r", "P1", "P2", "P3", "P1x"]
KINDS = ["order", "sampling", "battle", "all"]
CELL_COLS = ["flips", "both", "ndec", "nrep", "hc_correct", "hc_ndec", "cons_correct"]


def k_battles(proto, n):
    return n // 2 if proto in ("P1", "P2") else n


def n_calls(proto, n):
    return 2 * n if proto == "P1x" else n


def subset_idx(rng, m, k, D):
    if k >= m:
        return np.tile(np.arange(m), (D, 1))
    return np.argsort(rng.random((D, m)), axis=1)[:, :k]


def gen_sets(rng, D):
    """Per draw, split the four sampled generations into two disjoint pairs, one per replicate.
    Calls used by replicate 1 and replicate 2 therefore never coincide (they are independent samples)."""
    perm = np.argsort(rng.random((D, 4)), axis=1) + 1
    return perm[:, :2], perm[:, 2:]


def _gen(rng, shape, gen_mode, gset):
    if gen_mode == "t0":
        return np.zeros(shape, dtype=int)
    pick = rng.integers(0, 2, shape)
    return np.take_along_axis(gset, pick.reshape(shape[0], -1), axis=1).reshape(shape)


def zscore(vals, ties="half"):
    """z = (w - 0.5) / SE with SE = sd/sqrt(k). ties='drop' removes tie-valued battles first."""
    if ties == "drop":
        mask = vals != 0.5
        k = mask.sum(1); kk = np.maximum(k, 1)
        w = (vals * mask).sum(1) / kk
        var = (mask * (vals - w[:, None]) ** 2).sum(1) / np.maximum(k - 1, 1)
        se = np.sqrt(var / kk); ok = k >= 2
    else:
        k = vals.shape[1]; w = vals.mean(1); se = vals.std(1, ddof=1) / np.sqrt(k); ok = np.ones(len(w), bool)
    d = w - 0.5
    safe = np.where(se > 1e-12, se, 1.0)
    with np.errstate(invalid="ignore"):
        z = np.where(se > 1e-12, d / safe, np.sign(d) * np.inf)
    z = np.where(np.abs(d) < 1e-12, 0.0, z)
    return np.where(ok, z, 0.0)


def decide(z, zstar):
    d = np.zeros(z.shape, dtype=int)
    d[(z > 0) & (z >= zstar)] = 1
    d[(z < 0) & (z <= -zstar)] = -1
    return d


def battle_vals(proto, Sd, judge, rot, idx, rng, ds_order, gen_mode="sampled", gset=None):
    D, k = idx.shape
    S = Sd[judge] if judge in Sd else None
    if proto == "P0":
        o = ds_order[idx]
        return S[idx, o, _gen(rng, idx.shape, gen_mode, gset)]
    if proto == "P0r":
        o = rng.integers(0, 2, idx.shape)
        return S[idx, o, _gen(rng, idx.shape, gen_mode, gset)]
    if proto in ("P1", "P1x"):
        s1 = S[idx, 0, _gen(rng, idx.shape, gen_mode, gset)]
        s2 = S[idx, 1, _gen(rng, idx.shape, gen_mode, gset)]
        return np.where(s1 == s2, s1, 0.5)
    if proto == "P2":
        o = rng.integers(0, 2, idx.shape)
        g1 = np.broadcast_to(gset[:, [0]], idx.shape); g2 = np.broadcast_to(gset[:, [1]], idx.shape)   # two distinct sampled calls
        s1, s2 = S[idx, o, g1], S[idx, o, g2]
        return np.where(s1 == s2, s1, 0.5)
    if proto == "P3":
        o = rng.integers(0, 2, idx.shape); g = _gen(rng, idx.shape, gen_mode, gset)
        vals = np.stack([Sd[j][idx, o, g] for j in rot])             # (R, D, k)
        off = rng.integers(0, len(rot), (D, 1)); sel = (np.arange(k)[None, :] + off) % len(rot)
        return np.take_along_axis(vals, sel[None], axis=0)[0]
    raise ValueError(proto)


def simulate_z(proto, Sd, judge, rot, n, rng, D, ds_order, gen_mode="sampled", ties="half"):
    m = len(ds_order); k = k_battles(proto, n)
    if k > m or k < 2:
        return None
    idx = subset_idx(rng, m, k, D)                                   # both replicates share the battle subset
    gs = gen_sets(rng, D)
    return tuple(zscore(battle_vals(proto, Sd, judge, rot, idx, rng, ds_order, gen_mode, gs[r]), ties) for r in range(2))


def _sgn(w):
    return 0 if abs(w - 0.5) < 1e-12 else int(np.sign(w - 0.5))


def counts(z1, z2, zs, human_dec, human_clear, cons_dec):
    d1, d2 = decide(z1, zs), decide(z2, zs)
    both = (d1 != 0) & (d2 != 0); flips = both & (d1 != d2)
    ndec = int((d1 != 0).sum() + (d2 != 0).sum())
    hcc = int((d1 == human_dec).sum() + (d2 == human_dec).sum()) if human_clear else 0
    cc = int((d1 == cons_dec).sum() + (d2 == cons_dec).sum()) if cons_dec != 0 else 0
    return [int(flips.sum()), int(both.sum()), ndec, 2 * len(z1), hcc, ndec if human_clear else 0, cc]


def simulate_pair(P, judges, rot, protos, budgets, zs, rng, D, pool=None, gen_mode="sampled", ties="half"):
    Sd = P.S if pool is None else {j: s[pool] for j, s in P.S.items()}
    ds = P.ds_order if pool is None else P.ds_order[pool]
    slots = list(judges) + ["rot"]
    out = np.zeros((len(slots), len(protos), len(budgets), 7), dtype=np.int64)
    cons = {j: _sgn(Sd[j].mean()) for j in Sd}
    cons["rot"] = _sgn(np.mean([Sd[j].mean() for j in rot]))
    for si, slot in enumerate(slots):
        for pi, proto in enumerate(protos):
            if (slot == "rot") != (proto == "P3"):
                continue
            for bi, n in enumerate(budgets):
                r = simulate_z(proto, Sd, slot, rot, n, rng, D, ds, gen_mode, ties)
                if r is None:
                    continue
                out[si, pi, bi] = counts(r[0], r[1], zs[(slot, proto)], P.human_dec, P.human_clear, cons[slot])
    return out


def decomp_z(kind, S, n, rng, D, ties="half"):
    m = S.shape[0]
    if n > m:
        return None
    if kind == "order":                                  # canonical order reversed; deterministic decoding
        idx = subset_idx(rng, m, n, D); z = (zscore(S[idx, 0, 0], ties), zscore(S[idx, 1, 0], ties))
    elif kind == "sampling":                             # fixed order and battles; independent sampled generations
        idx = subset_idx(rng, m, n, D); gs = gen_sets(rng, D)
        z = tuple(zscore(S[idx, 0, _gen(rng, idx.shape, "sampled", gs[r])], ties) for r in range(2))
    elif kind == "battle":                               # fixed order, T0; different battle subsets
        z = tuple(zscore(S[subset_idx(rng, m, n, D), 0, 0], ties) for _ in range(2))
    else:                                                # all sources at once
        z = []; gs = gen_sets(rng, D)
        for r in range(2):
            idx = subset_idx(rng, m, n, D)
            z.append(zscore(S[idx, rng.integers(0, 2, idx.shape), _gen(rng, idx.shape, "sampled", gs[r])], ties))
        z = tuple(z)
    return z


def decomp_pair(P, judges, budgets, zs, rng, D, pool=None, ties="half"):
    out = np.zeros((len(judges), len(KINDS), len(budgets), 7), dtype=np.int64)
    for ji, j in enumerate(judges):
        S = P.S[j] if pool is None else P.S[j][pool]
        for ki, kind in enumerate(KINDS):
            for bi, n in enumerate(budgets):
                r = decomp_z(kind, S, n, rng, D, ties)
                if r is None:
                    continue
                out[ji, ki, bi] = counts(r[0], r[1], zs[(j, "P0")], 0, False, 0)
    return out


def tune_zstar(pairs, judges, rot, coverage, n, rng, D, ties="half", gen_mode="sampled"):
    """z* per (slot, protocol): the |z| value that leaves `coverage` of decisions decisive on development battles."""
    zs, diag = {}, []
    slots = list(judges) + ["rot"]
    for slot in slots:
        for proto in PROTOS:
            if (slot == "rot") != (proto == "P3"):
                continue
            allz = []
            for P in pairs:
                r = simulate_z(proto, P.S, slot, rot, n, rng, D, P.ds_order, gen_mode, ties)
                if r is not None:
                    allz += [np.abs(r[0]), np.abs(r[1])]
            a = np.sort(np.concatenate(allz))
            val = float(a[int(np.floor((1 - coverage) * len(a)))]) if len(a) else 0.0
            reach = float(np.mean(a > 0)) if len(a) else 0.0
            zs[(slot, proto)] = val
            diag.append(dict(slot=slot, protocol=proto, zstar=val, max_reachable_coverage=reach, n_draws=len(a),
                             coverage_ok=bool(val > 0 or reach >= coverage)))
    return zs, diag
