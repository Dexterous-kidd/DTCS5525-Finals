"""Stage `analyze`: every number, table and hypothesis test for one source (arena or mt)."""
import json, logging, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import boot as bootmod
from . import report
from .data import load_processed
from .metrics import holm, mcnemar_exact, verdict_table_metrics
from .probit import predict_cell
from .records import build_pairdata, load_verdicts
from .sim import KINDS, PROTOS, k_battles, n_calls, simulate_z, decide, tune_zstar
from .tuning import all_judges, check_frozen, load_tuned

log = logging.getLogger("judgestudy")


# ------------------------------------------------------------------ small helpers
def _ratio(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def _ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if len(v) else (np.nan, np.nan)


def make_resample(mask, B, rng):
    sel = np.flatnonzero(mask)
    return sel[rng.integers(0, len(sel), size=(B, len(sel)))]


def agg(C, RS):
    return C[np.arange(C.shape[0])[:, None], RS].sum(1)


# ------------------------------------------------------------------ generation / item-level summaries
def generation_summary(ctx, judges, source):
    cfg = ctx.cfg; rows, eff = [], []
    for j in judges:
        v = load_verdicts(cfg, j, source)
        rows.append(dict(judge=j, source=source, calls=len(v), invalid_after_retry=float((v.verdict == "X").mean()),
                         retried=float(v.retried.mean()), truncated=float(v.truncated.mean()),
                         tie_rate=float((v.verdict == "T").mean()), n_in=int(v.n_in.sum()), n_out=int(v.n_out.sum())))
        p = Path(cfg["paths"]["verdicts"]) / j / "efficiency.jsonl"
        if p.exists():
            e = pd.read_json(p, lines=True); e = e[e.source == source]
            eff.append(dict(judge=j, source=source, calls=int(e.calls.sum()), seconds=float(e.seconds.sum()),
                            calls_per_s=float(e.calls.sum() / max(e.seconds.sum(), 1e-9)),
                            tokens_in=int(e.n_in.sum()), tokens_out=int(e.n_out.sum()),
                            out_tokens_per_s=float(e.n_out.sum() / max(e.seconds.sum(), 1e-9)),
                            gpu=";".join(sorted(set(map(str, e.gpu))))))
    ctx.table(pd.DataFrame(rows), f"generation_quality_{source}")
    if eff:
        ctx.table(pd.DataFrame(eff), f"efficiency_{source}")
    for r in rows:
        if r["invalid_after_retry"] > 0.01:
            log.warning("[%s/%s] invalid verdict rate after retry %.2f%% exceeds the 1%% pre-specified limit: adjust the parser/prompt on DEV and regenerate",
                        r["judge"], source, 100 * r["invalid_after_retry"])


def item_level(ctx, units, pairs, judges, source):
    cfg = ctx.cfg; u = units[units.split == "test"]; rows = []
    human = dict(zip(u.unit_id, u.human)); setmap = dict(zip(u.unit_id, u.set))
    for j in judges:
        v = load_verdicts(cfg, j, source); v = v[v.unit_id.isin(human)]
        for scope in ("all", "close", "far"):
            vv = v if scope == "all" else v[v.unit_id.map(setmap) == scope]
            if len(vv) == 0:
                continue
            m = verdict_table_metrics(vv, pd.Series(human))
            rows.append(dict(judge=j, source=source, scope=scope, **m))
    df = pd.DataFrame(rows); ctx.table(df, f"item_level_metrics_{source}")
    if source == "mt":                       # human ceiling, if saved during prep
        p = Path(cfg["paths"]["processed"]) / "mt_extra.json"
        if p.exists():
            ctx.record("mt_human_ceiling", **json.loads(p.read_text()))
    return df


# ------------------------------------------------------------------ summary tables
def summary_table(Xpt, Xb, scope, slots, budgets):
    rows = []
    for si, slot in enumerate(slots):
        for pi, proto in enumerate(PROTOS):
            if (slot == "rot") != (proto == "P3"):
                continue
            for bi, n in enumerate(budgets):
                x = Xpt[si, pi, bi]
                if x[3] == 0:
                    continue
                xb = Xb[:, si, pi, bi]
                fr, cv, hc = _ratio(xb[:, 0], xb[:, 1]), _ratio(xb[:, 2], xb[:, 3]), _ratio(xb[:, 4], xb[:, 5])
                rows.append(dict(scope=scope, slot=slot, protocol=proto, budget=n, calls=n_calls(proto, n),
                                 flip_rate=float(_ratio(x[0], x[1])), flip_lo=_ci(fr)[0], flip_hi=_ci(fr)[1],
                                 coverage=float(_ratio(x[2], x[3])), cov_lo=_ci(cv)[0], cov_hi=_ci(cv)[1],
                                 hc_agreement=float(_ratio(x[4], x[5])), hc_lo=_ci(hc)[0], hc_hi=_ci(hc)[1],
                                 consensus_agreement=float(_ratio(x[6], x[2])), n_both_decisive=int(x[1])))
    return pd.DataFrame(rows)


def decomp_table(Dpt, Db, scope, judges, budgets):
    rows = []
    for ji, j in enumerate(judges):
        for ki, kind in enumerate(KINDS):
            for bi, n in enumerate(budgets):
                x = Dpt[ji, ki, bi]
                if x[3] == 0:
                    continue
                fr = _ratio(Db[:, ji, ki, bi, 0], Db[:, ji, ki, bi, 1]); lo, hi = _ci(fr)
                rows.append(dict(scope=scope, judge=j, source_of_variation=kind, budget=n, flip_rate=float(_ratio(x[0], x[1])),
                                 flip_lo=lo, flip_hi=hi, coverage=float(_ratio(x[2], x[3])), n_both_decisive=int(x[1])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ hypothesis machinery
def _rate(X, slot_i, proto_i, bud_i):
    return _ratio(X[..., slot_i, proto_i, bud_i, 0], X[..., slot_i, proto_i, bud_i, 1])


def _hc(X, slot_i, proto_i, bud_i):
    return _ratio(X[..., slot_i, proto_i, bud_i, 4], X[..., slot_i, proto_i, bud_i, 5])


def _core_mean(fn, X, core_si):
    return np.nanmean(np.stack([fn(X, si) for si in core_si], 0), axis=0)


def logistic_slope(counts, margins):
    import statsmodels.api as sm
    both = counts[:, 1]; ok = both > 0
    if ok.sum() < 5 or np.ptp(margins[ok]) < 1e-9:
        return np.nan
    y = np.column_stack([counts[ok, 0], both[ok] - counts[ok, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = sm.GLM(y, sm.add_constant(margins[ok]), family=sm.families.Binomial()).fit()
            return float(r.params[1])
        except Exception:
            return np.nan


def hypotheses(ctx, source, plist, judges, core, budgets, Cpt, Cb, Dpt, Db, tuned, rng):
    cfg = ctx.cfg; H = cfg["hypotheses"]; B = Cb.shape[0]
    slots = judges + ["rot"]; si = {s: i for i, s in enumerate(slots)}; pi = {p: i for i, p in enumerate(PROTOS)}
    close = np.array([p.set == "close" for p in plist]); allm = np.ones(len(plist), bool)
    RSc, RSa = make_resample(close, B, rng), make_resample(allm, B, rng)
    core_si = [si[j] for j in core]; out = {}
    bud1a = H["h1a"]["budget"] if H["h1a"]["budget"] in budgets else max(budgets)
    bud2 = H["h2"]["budget"] if H["h2"]["budget"] in budgets else max(budgets)
    b1, b2 = budgets.index(bud1a), budgets.index(bud2)
    Xpt_c = Cpt[0][close].sum(0); Xb_c = agg(Cb, RSc)
    Xpt_a = Cpt[0].sum(0); Xb_a = agg(Cb, RSa)
    n_hc_close = int(sum(p.human_clear for p in plist if p.set == "close")); n_hc_all = int(sum(p.human_clear for p in plist))
    ctx.record(f"human_clear_pairs/{source}", close=n_hc_close, all=n_hc_all, n_pairs=len(plist))
    if n_hc_close == 0:
        log.warning("no close pair is human-clear: agreement with human-clear decisions (S3) is evaluated on ALL human-clear pairs (%d)", n_hc_all)
    Dpt_c = Dpt[0][close].sum(0); Db_c = agg(Db, RSc)

    # ---------- H1a: order reversal flips >= threshold of decisions (close pairs, deterministic decoding) ----------
    rows = []
    thr = H["h1a"]["threshold"]
    for j in judges:
        ji = judges.index(j)
        pt = float(_ratio(Dpt_c[ji, 0, b1, 0], Dpt_c[ji, 0, b1, 1]))
        vals = _ratio(Db_c[:, ji, 0, b1, 0], Db_c[:, ji, 0, b1, 1])
        lo, hi = float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))
        rows.append(dict(judge=j, core=j in core, budget=bud1a, flip_rate=pt, lower90=lo, upper95_onesided=hi,
                         supported=bool(lo > thr), bounded_risk=bool(hi < thr)))
    h1a = pd.DataFrame(rows); ctx.table(h1a, f"H1a_{source}")
    c = h1a[h1a.core]
    out["H1a"] = "SUPPORTED" if c.supported.any() else ("BOUNDED RISK (upper bound below threshold for all core judges)" if c.bounded_risk.all() else "INCONCLUSIVE")
    ctx.record(f"H1a/{source}", verdict=out["H1a"], threshold=thr, budget=bud1a, **{f"rate_{r.judge}": r.flip_rate for r in h1a.itertuples()})

    # ---------- H1b: pair-level flip rate (P0r) falls with human margin ----------
    margins = np.array([p.margin for p in plist]); rows = []
    for name, cols in [(j, [si[j]]) for j in core] + [("pooled_core", core_si)]:
        cnt = Cpt[0][:, cols, pi["P0r"], b2, :2].sum(1)
        pt = logistic_slope(cnt, margins); bs = []
        for b in range(B):
            idx = rng.integers(0, len(plist), len(plist))
            bs.append(logistic_slope(Cb[b, idx][:, cols, pi["P0r"], b2, :2].sum(1), margins[idx]))
        lo, hi = _ci(np.array(bs))
        rows.append(dict(judge=name, budget=bud2, slope=pt, lo=lo, hi=hi, n_pairs=len(plist), supported=bool(hi < 0)))
    h1b = pd.DataFrame(rows); ctx.table(h1b, f"H1b_{source}")
    out["H1b"] = "SUPPORTED" if bool(h1b[h1b.judge == "pooled_core"].supported.iloc[0]) else "NOT SUPPORTED / INCONCLUSIVE"
    ctx.record(f"H1b/{source}", verdict=out["H1b"], **{f"slope_{r.judge}": r.slope for r in h1b.itertuples()})

    # ---------- H2 and secondary tests (close pairs, core judges pooled with equal weight) ----------
    def diff(X, a, b_, fn):
        ra = _core_mean(lambda x, s: fn(x, s, pi[a], b2), X, core_si)
        rb = _core_mean(lambda x, s: fn(x, s, pi[b_], b2), X, core_si)
        return ra, rb
    r0, r1 = diff(Xb_c, "P0r", "P1", _rate); p0, p1 = diff(Xpt_c, "P0r", "P1", _rate)
    d_boot = r0 - r1; d_pt = float(p0 - p1); lo5 = float(np.nanpercentile(d_boot, 5)); hi95 = float(np.nanpercentile(d_boot, 95))
    h2 = dict(budget=bud2, rate_P0r=float(p0), rate_P1=float(p1), diff=d_pt, lower95_onesided=lo5, upper95_onesided=hi95,
              supported=bool(lo5 > 0), practically_relevant=bool(d_pt >= H["h2"]["min_effect"]), falsified=bool(hi95 < 0))
    ctx.table(pd.DataFrame([h2]), f"H2_{source}")
    out["H2"] = "SUPPORTED" if h2["supported"] else ("FALSIFIED (P1 not better)" if h2["falsified"] else "INCONCLUSIVE")
    ctx.record(f"H2/{source}", verdict=out["H2"], **h2)

    # secondary: Holm over four
    sec, stats = [], {}
    a, b_ = diff(Xb_c, "P0r", "P2", _rate); stats["S1 P0r minus P2 flip rate (>0)"] = (a - b_, 0.0)
    a, b_ = diff(Xpt_c, "P0r", "P2", _rate); pts = {"S1 P0r minus P2 flip rate (>0)": float(a - b_)}
    p3 = _rate(Xb_c, si["rot"], pi["P3"], b2); p3pt = float(_rate(Xpt_c, si["rot"], pi["P3"], b2))
    stats["S2 P0r minus P3 flip rate (>0)"] = (r0 - p3, 0.0); pts["S2 P0r minus P3 flip rate (>0)"] = float(p0 - p3pt)
    ha = _core_mean(lambda x, s: _hc(x, s, pi["P1"], b2), Xb_a, core_si); hb = _core_mean(lambda x, s: _hc(x, s, pi["P0r"], b2), Xb_a, core_si)
    hap = _core_mean(lambda x, s: _hc(x, s, pi["P1"], b2), Xpt_a, core_si); hbp = _core_mean(lambda x, s: _hc(x, s, pi["P0r"], b2), Xpt_a, core_si)
    mg = H["noninferiority_margin"]
    stats["S3 P1 minus P0r agreement with human-clear pairs (> -margin)"] = (ha - hb, -mg); pts["S3 P1 minus P0r agreement with human-clear pairs (> -margin)"] = float(hap - hbp)
    x1, x2 = diff(Xb_c, "P1", "P1x", _rate); y1, y2 = diff(Xpt_c, "P1", "P1x", _rate)
    stats["S4 P1 minus P1x flip rate (>0 means doubling the cost helps)"] = (x1 - x2, 0.0); pts["S4 P1 minus P1x flip rate (>0 means doubling the cost helps)"] = float(y1 - y2)
    names = list(stats)
    pvals = []
    for k in names:
        v, null = stats[k]; v = v[~np.isnan(v)]
        pvals.append(float((1 + np.sum(v <= null)) / (len(v) + 1)))
    adj, rej = holm(pvals)
    for k, pv, ad, rj in zip(names, pvals, adj, rej):
        v = stats[k][0]
        sec.append(dict(test=k, point=pts[k], lo95=_ci(v)[0], hi95=_ci(v)[1], p_raw=pv, p_holm=float(ad), reject=bool(rj)))
    ctx.table(pd.DataFrame(sec), f"H2_secondary_holm_{source}")

    # ---------- H3: predictions from dev-fitted ordered probits vs observed test flip rates ----------
    rot = [j for j in cfg["judges_cfg"]["rotation_judges"] if j in judges]
    zs, _ = load_tuned(cfg, source)
    for model, key, tag in (("M0 (pre-specified)", "probit", "H3"), ("M1 random intercept (exploratory)", "probit_re", "H3_RE")):
        if key not in tuned:
            continue
        res = _h3(ctx, source, plist, judges, rot, budgets, Cpt, zs, {j: tuned[key][j]["theta"] for j in judges}, tag, B, rng, model)
        out[tag] = res["verdict"]
        if tag == "H3":
            cells_main = res["cells"]
    return out, dict(h1a=h1a, h1b=h1b, cells=cells_main, close=close, Xb_c=Xb_c, Xpt_c=Xpt_c, RSc=RSc, si=si, pi=pi, b2=b2)


def _h3(ctx, source, plist, judges, rot, budgets, Cpt, zs, thetas, tag, B, rng, model_name):
    H = ctx.cfg["hypotheses"]; slots = judges + ["rot"]; cells = []
    for pidx, P in enumerate(plist):
        f_ds = float(P.ds_order.mean())
        for sidx, slot in enumerate(slots):
            for qi, proto in enumerate(PROTOS):
                if (slot == "rot") != (proto == "P3") or proto == "P0":
                    continue
                th = {j: thetas[j] for j in rot} if slot == "rot" else {slot: thetas[slot]}
                for bi, n in enumerate(budgets):
                    x = Cpt[0][pidx, sidx, qi, bi]
                    if x[3] == 0:
                        continue
                    pf, pc = predict_cell(proto, th, P.signed_margin, n, zs[(slot, proto)], k_battles(proto, n), f_ds, len(P.unit_ids))
                    cells.append(dict(pair_id=P.pair_id, set=P.set, margin=P.margin, slot=slot, protocol=proto, budget=n, pred_flip=pf, pred_cov=pc,
                                      obs_flips=int(x[0]), obs_both=int(x[1]), obs_flip=float(_ratio(x[0], x[1])), obs_cov=float(_ratio(x[2], x[3]))))
    cdf = pd.DataFrame(cells); ctx.table(cdf, f"{tag}_cells_{source}", show=False)
    ok = cdf[(cdf.obs_both >= 20) & cdf.pred_flip.notna() & cdf.obs_flip.notna()]
    def calib(d):
        if len(d) < 5 or d.pred_flip.nunique() < 2:
            return np.nan, np.nan
        return float(np.polyfit(d.pred_flip, d.obs_flip, 1)[0]), float(spearmanr(d.pred_flip, d.obs_flip).correlation)
    sl, rho = calib(ok); bs_sl, bs_rho = [], []
    grp = {p: g for p, g in ok.groupby("pair_id")}; pids = np.array(list(grp))
    for _ in range(min(B, 500)):
        if len(pids) == 0:
            break
        samp = pd.concat([grp[p] for p in rng.choice(pids, len(pids))]); a_, b_ = calib(samp); bs_sl.append(a_); bs_rho.append(b_)
    cand = ["P0r", "P1", "P2"]; match, tot, brows = 0, 0, []
    for slot in [s for s in slots if s != "rot"]:
        for n in budgets:
            obs, pred = {}, {}
            for pr in cand:
                d = cdf[(cdf.set == "close") & (cdf.slot == slot) & (cdf.protocol == pr) & (cdf.budget == n)]
                if len(d) and d.obs_both.sum() > 0:
                    obs[pr] = d.obs_flips.sum() / d.obs_both.sum(); pred[pr] = d.pred_flip.mean()
            d3 = cdf[(cdf.set == "close") & (cdf.slot == "rot") & (cdf.protocol == "P3") & (cdf.budget == n)]
            if slot in rot and len(d3) and d3.obs_both.sum() > 0:
                obs["P3"] = d3.obs_flips.sum() / d3.obs_both.sum(); pred["P3"] = d3.pred_flip.mean()
            if len(obs) >= 2 and len(pred) >= 2:
                bo, bp = min(obs, key=obs.get), min(pred, key=pred.get); tot += 1; match += int(bo == bp)
                brows.append(dict(judge=slot, budget=n, best_observed=bo, best_predicted=bp, match=bo == bp))
    ctx.table(pd.DataFrame(brows), f"{tag}_best_protocol_{source}")
    share = match / tot if tot else np.nan; h = H["h3"]
    r = dict(model=model_name, n_cells=int(len(ok)), calibration_slope=sl, slope_lo=_ci(bs_sl)[0], slope_hi=_ci(bs_sl)[1], spearman=rho,
             rho_lo=_ci(bs_rho)[0], rho_hi=_ci(bs_rho)[1], best_protocol_share=float(share),
             slope_ok=bool(h["slope"][0] <= sl <= h["slope"][1]), rho_ok=bool(rho >= h["spearman"]), best_ok=bool(share >= h["best_protocol_share"]))
    r["supported"] = bool(r["slope_ok"] and r["rho_ok"] and r["best_ok"])
    ctx.table(pd.DataFrame([r]), f"{tag}_{source}")
    verdict = "SUPPORTED" if r["supported"] else "NOT SUPPORTED (report which of the three conditions failed)"
    ctx.record(f"{tag}/{source}", verdict=verdict, **r)
    return dict(verdict=verdict, cells=cdf)


# ------------------------------------------------------------------ risk-coverage and ablations
def risk_coverage(ctx, plist, judges, rot, zs, budget, grid, D, seed, source):
    rng = np.random.default_rng(seed); rows = []
    close = [P for P in plist if P.set == "close"]
    for slot in judges + ["rot"]:
        for proto in ["P0r", "P1", "P2", "P3"]:
            if (slot == "rot") != (proto == "P3"):
                continue
            Z = []
            for P in close:
                r = simulate_z(proto, P.S, slot, rot, budget, rng, D, P.ds_order)
                if r is not None:
                    Z.append(r)
            if not Z:
                continue
            z1 = np.concatenate([r[0] for r in Z]); z2 = np.concatenate([r[1] for r in Z])
            for g in grid:
                d1, d2 = decide(z1, g), decide(z2, g); both = (d1 != 0) & (d2 != 0)
                rows.append(dict(slot=slot, protocol=proto, zstar=g, coverage=float(((d1 != 0).mean() + (d2 != 0).mean()) / 2),
                                 flip_rate=float(((d1 != d2) & both).sum() / both.sum()) if both.sum() else np.nan))
    df = pd.DataFrame(rows); ctx.table(df, f"risk_coverage_{source}", show=False)
    return df


def ablations(ctx, source, plist, dev_list, judges, core, rot, budgets, zs_main, bud2, seed):
    """Point-estimate sensitivity checks on the P0r vs P1 contrast (close pairs, core judges)."""
    s = ctx.cfg["sim"]; rows = []
    close_idx = [i for i, p in enumerate(plist) if p.set == "close"]
    by_margin = sorted(close_idx, key=lambda i: plist[i].margin)
    variants = {"main": dict(), "t0_only": dict(gen_mode="t0"), "ties_dropped": dict(ties="drop", retune=True),
                "coverage_70": dict(cov=0.70, retune=True), "coverage_90": dict(cov=0.90, retune=True),
                "no_tie_band(z*=0)": dict(zero=True), "closest_20_pairs": dict(subset=by_margin[:20])}
    pi = {p: i for i, p in enumerate(PROTOS)}; bi = budgets.index(bud2)
    for name, v in variants.items():
        rng = np.random.default_rng(seed)
        zs = dict(zs_main)
        if v.get("zero"):
            zs = {k: 0.0 for k in zs}
        if v.get("retune"):
            zs, _ = tune_zstar([d for d in dev_list], judges, rot, v.get("cov", s["coverage"]), s["tune_budget"], rng, s["n_draws_point"],
                               ties=v.get("ties", "half"))
        sub = v.get("subset", close_idx)
        protos = ["P0r", "P1", "P1x"] if v.get("gen_mode") == "t0" else ["P0r", "P1", "P2", "P1x"]
        C, _ = bootmod.run([plist[i] for i in sub], judges, rot, PROTOS, budgets, zs, 1, s["n_draws_point"], seed, 1, False,
                           v.get("gen_mode", "sampled"), v.get("ties", "half"), label=f"ablation:{name}")
        X = C[0].sum(0)
        for j in core:
            sj = judges.index(j)
            r0, r1, r2 = (float(_rate(X, sj, pi[p], bi)) for p in ("P0r", "P1", "P2"))
            if name == "t0_only":
                r2 = np.nan                       # P2 needs sampled generations; undefined under temperature zero
            cov0 = float(_ratio(X[sj, pi["P0r"], bi, 2], X[sj, pi["P0r"], bi, 3])); cov1 = float(_ratio(X[sj, pi["P1"], bi, 2], X[sj, pi["P1"], bi, 3]))
            rows.append(dict(variant=name, judge=j, budget=budgets[bi], flip_P0r=r0, flip_P1=r1, flip_P2=r2, diff_P0r_minus_P1=r0 - r1,
                             coverage_P0r=cov0, coverage_P1=cov1, n_pairs=len(sub)))
    df = pd.DataFrame(rows); ctx.table(df, f"ablations_{source}")
    return df


# ------------------------------------------------------------------ MT-Bench replication
def anchor_replication(ctx, results):
    a, m = results.get("arena"), results.get("mt")
    if not a or not m:
        return
    rows = []
    ra, rm = a["frames"], m["frames"]
    def first(df, col): return float(df[col].iloc[0]) if len(df) else np.nan
    pairs = [("H1a order-reversal flip rate (mean over core judges)", ra["h1a"][ra["h1a"].core].flip_rate.mean(), rm["h1a"][rm["h1a"].core].flip_rate.mean()),
             ("H1b slope (pooled core judges, should be < 0)", first(ra["h1b"][ra["h1b"].judge == "pooled_core"], "slope"), first(rm["h1b"][rm["h1b"].judge == "pooled_core"], "slope")),
             ("H2 diff P0r minus P1 (should be > 0)", a["h2"]["diff"], m["h2"]["diff"])]
    for name, x, y in pairs:
        rows.append(dict(effect=name, arena=x, mt_bench=y, same_direction=bool(np.sign(x) == np.sign(y)) if not (np.isnan(x) or np.isnan(y)) else None))
    ctx.table(pd.DataFrame(rows), "anchor_replication")


# ------------------------------------------------------------------ error-analysis sheet (private)
def error_sheet(ctx, units, pairs, plist, judges, core, source, n_rows=50):
    cfg = ctx.cfg
    if source != "arena":
        return
    closest = sorted([p for p in plist if p.set == "close"], key=lambda p: p.margin)[:5]
    ids = {u for p in closest for u in p.unit_ids}
    uu = units[units.unit_id.isin(ids)].set_index("unit_id"); rows = []
    for j in core:
        v = load_verdicts(cfg, j, source); v = v[v.unit_id.isin(ids)]
        for uid, g in v.groupby("unit_id"):
            lab = g.verdict.tolist(); modal = max(set(lab), key=lab.count)
            pay = json.loads(uu.loc[uid, "payload"]); la, lb = len(pay["resp_A"]), len(pay["resp_B"])
            md = lambda t: len(re.findall(r"(\*\*|^#+ |^[-*] |```)", t, re.M))
            rows.append(dict(judge=j, unit_id=uid, pair_id=uu.loc[uid, "pair_id"], instability=1 - lab.count(modal) / len(lab),
                             len_A=la, len_B=lb, len_ratio=max(la, lb) / max(1, min(la, lb)), md_A=md(pay["resp_A"]), md_B=md(pay["resp_B"]),
                             prompt_excerpt=pay["question"][:300].replace("\n", " "), reason=""))
    df = pd.DataFrame(rows)
    if not len(df):
        return
    df["len_bin"] = pd.cut(df.len_ratio, [0, 1.25, 2, 4, np.inf], labels=["<=1.25x", "1.25-2x", "2-4x", ">4x"])
    df["format_gap"] = (df.md_A - df.md_B).abs().clip(upper=3)
    ctx.table(df.groupby(["judge", "len_bin"], observed=True).instability.agg(["mean", "count"]).reset_index(), "instability_by_length_ratio")
    ctx.table(df.groupby(["judge", "format_gap"]).instability.agg(["mean", "count"]).reset_index(), "instability_by_format_gap")
    private = Path(cfg["paths"]["private"]); private.mkdir(parents=True, exist_ok=True)
    sheet = df.sort_values("instability", ascending=False).groupby("judge").head(n_rows)
    sheet.to_csv(private / "error_coding_sheet.csv", index=False)
    log.info("error coding sheet (contains prompt excerpts: PRIVATE) -> %s", private / "error_coding_sheet.csv")


# ------------------------------------------------------------------ main entry
def analyze(ctx, source, judges=None, no_freeze_check=False, skip_boot=False, quick=False):
    cfg = ctx.cfg; s = cfg["sim"]
    if not no_freeze_check:
        check_frozen(cfg, [source])
    units, pairs = load_processed(cfg, source)
    judges = judges or all_judges(cfg, True, source)
    core = [j for j in judges if cfg["judges_cfg"]["judges"][j].get("core")]
    rot = [j for j in cfg["judges_cfg"]["rotation_judges"] if j in judges]
    zs, tuned = load_tuned(cfg, source)
    budgets = s["budgets"][source]
    ctx.log.info("analysis source=%s judges=%s core=%s rotation=%s budgets=%s", source, judges, core, rot, budgets)
    with ctx.stage(f"summaries/{source}"):
        generation_summary(ctx, judges, source)
        item_level(ctx, units, pairs, judges, source)
    PD = build_pairdata(cfg, units, pairs, judges, source, "test")
    plist = sorted(PD.values(), key=lambda p: p.pair_id)
    dev = list(build_pairdata(cfg, units, pairs, judges, source, "dev").values())
    dev = [d for d in dev if d.set == "close" or source == "mt"]
    ctx.table(pd.DataFrame([dict(pair_id=p.pair_id, set=p.set, margin=p.margin, signed_margin=p.signed_margin, human_clear=p.human_clear,
                                 n_test=len(p.unit_ids), **{f"invalid_{j}": p.n_invalid[j] for j in judges}) for p in plist]), f"test_pairs_{source}", show=False)
    B = 8 if quick else s["n_boot"]; Db_ = 20 if quick else s["n_draws_boot"]; Dp = 60 if quick else s["n_draws_point"]
    seeds = s["analysis_seeds"][:2] if quick else s["analysis_seeds"]
    with ctx.stage(f"simulate_point/{source}"):
        pts = []
        for sd in seeds:
            C, Dc = bootmod.run(plist, judges, rot, PROTOS, budgets, zs, 1, Dp, sd, 1, False, label=f"point seed {sd}")
            pts.append((C, Dc))
        Cpt, Dpt = pts[0]
    with ctx.stage(f"simulate_bootstrap/{source}"):
        Cb, Db = bootmod.run(plist, judges, rot, PROTOS, budgets, zs, B, Db_, cfg["seed"] + 100, s["n_jobs"], True, label="bootstrap")
    rng = np.random.default_rng(cfg["seed"] + 200)
    slots = judges + ["rot"]
    with ctx.stage(f"tables/{source}"):
        close = np.array([p.set == "close" for p in plist]); far = np.array([p.set == "far" for p in plist]); allm = np.ones(len(plist), bool)
        frames = []; dframes = []
        for scope, mask in (("close", close), ("far", far), ("all", allm)):
            if mask.sum() == 0:
                continue
            RS = make_resample(mask, B, rng)
            frames.append(summary_table(Cpt[0][mask].sum(0), agg(Cb, RS), scope, slots, budgets))
            dframes.append(decomp_table(Dpt[0][mask].sum(0), agg(Db, RS), scope, judges, budgets))
        ST = pd.concat(frames); DT = pd.concat(dframes)
        ctx.table(ST, f"protocol_summary_{source}", show=False); ctx.table(DT, f"decomposition_{source}", show=False)
        # seed spread of the headline quantities
        sp = []
        for sd, (C, Dc) in zip(seeds, pts):
            X = C[0][close].sum(0); pi_ = {p: i for i, p in enumerate(PROTOS)}
            b2 = budgets.index(cfg["hypotheses"]["h2"]["budget"]) if cfg["hypotheses"]["h2"]["budget"] in budgets else len(budgets) - 1
            core_si = [slots.index(j) for j in core]
            sp.append(dict(analysis_seed=sd, **{f"flip_P0r_{j}": float(_rate(X, slots.index(j), pi_["P0r"], b2)) for j in core},
                           **{f"flip_P1_{j}": float(_rate(X, slots.index(j), pi_["P1"], b2)) for j in core}))
        ctx.table(pd.DataFrame(sp), f"seed_spread_{source}")
    with ctx.stage(f"hypotheses/{source}"):
        verdicts, aux = hypotheses(ctx, source, plist, judges, core, budgets, Cpt, Cb, Dpt, Db, tuned, rng)
        h2 = pd.read_csv(ctx.dir / "tables" / f"H2_{source}.csv").iloc[0].to_dict()
    with ctx.stage(f"risk_coverage_and_ablations/{source}"):
        b2v = cfg["hypotheses"]["h2"]["budget"] if cfg["hypotheses"]["h2"]["budget"] in budgets else max(budgets)
        rc = risk_coverage(ctx, plist, judges, rot, zs, b2v, s["risk_coverage_grid"], Dp, cfg["seed"] + 300, source)
        ab = ablations(ctx, source, plist, dev, judges, core, rot, budgets, zs, b2v, cfg["seed"] + 400)
        error_sheet(ctx, units, pairs, plist, judges, core, source)
    with ctx.stage(f"figures/{source}"):
        report.figures(ctx, source, ST, DT, aux, rc, core, judges, b2v)
    ctx.record(f"analysis_done/{source}", verdicts=verdicts)
    return dict(verdicts=verdicts, h2=h2, frames=dict(h1a=aux["h1a"], h1b=aux["h1b"]))


# ------------------------------------------------------------------ sensitivity judge (e.g. Qwen3 thinking mode on a small dev subset)
def sensitivity(ctx, source, judge, baseline):
    """Compare a sensitivity judge with its baseline on exactly the units the sensitivity judge completed (all 10 calls)."""
    cfg = ctx.cfg; units, pairs = load_processed(cfg, source)
    vj, vb = load_verdicts(cfg, judge, source), load_verdicts(cfg, baseline, source)
    per = vj.groupby("unit_id").size(); ids = set(per[per >= 10].index) & set(vb.unit_id)
    if not ids:
        raise RuntimeError(f"no unit has all 10 calls for {judge}; finish generation first")
    human = pd.Series(dict(zip(units.unit_id, units.human))); rows = []
    for name, v in ((baseline, vb), (judge, vj)):
        vv = v[v.unit_id.isin(ids)]
        rows.append(dict(judge=name, n_units=len(ids), invalid_pct=100 * float((vv.verdict == "X").mean()), truncated_pct=100 * float(vv.truncated.mean()),
                         tie_pct=100 * float((vv.verdict == "T").mean()), mean_output_tokens=float(vv.n_out.mean()), total_output_tokens=int(vv.n_out.sum()),
                         **verdict_table_metrics(vv, human)))
    df = pd.DataFrame(rows); ctx.table(df, f"sensitivity_{judge}_vs_{baseline}_{source}")
    pick = lambda v: v[(v.unit_id.isin(ids)) & (v.order == "AB") & (v.gen == 0)].set_index("unit_id").verdict
    a_, b_ = pick(vj), pick(vb); common = a_.index.intersection(b_.index)
    ca = (a_[common] == human[common]).to_numpy(); cb = (b_[common] == human[common]).to_numpy()
    mc = mcnemar_exact(ca, cb)
    ratio = rows[1]["mean_output_tokens"] / max(rows[0]["mean_output_tokens"], 1e-9)
    ctx.record(f"sensitivity/{source}/{judge}_vs_{baseline}", n_units=len(ids), agree_sens=float(ca.mean()), agree_base=float(cb.mean()),
               mcnemar_only_sens_correct=mc["only_a"], mcnemar_only_base_correct=mc["only_b"], mcnemar_p=mc["p"], output_token_ratio=float(ratio))
    return df
