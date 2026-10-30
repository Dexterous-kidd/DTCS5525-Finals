"""Stages `tune` and `freeze`: everything fitted on DEVELOPMENT battles only, then locked before any test result is computed."""
import hashlib, json, logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_processed
from .probit import fit_probit, fit_probit_re
from .records import build_pairdata, load_verdicts
from .sim import tune_zstar
from .util import now, sha256_file

log = logging.getLogger("judgestudy")


def all_judges(cfg, available_only=True, source="arena"):
    names = [n for n, s in cfg["judges_cfg"]["judges"].items() if not s.get("sensitivity")]   # sensitivity judges have partial data by design
    if not available_only:
        return names
    return [n for n in names if any((Path(cfg["paths"]["verdicts"]) / n).glob(f"{source}.shard*.jsonl"))]


def core_judges(cfg):
    return [n for n, s in cfg["judges_cfg"]["judges"].items() if s.get("core")]


def tuned_path(cfg, source):
    return Path(cfg["paths"]["processed"]) / f"tuned_{source}.json"


def load_tuned(cfg, source):
    t = json.loads(tuned_path(cfg, source).read_text())
    zs = {(r["slot"], r["protocol"]): r["zstar"] for r in t["zstar"]}
    return zs, t


def tune(ctx, source, judges, coverage=None, tune_on="close"):
    cfg = ctx.cfg; s = cfg["sim"]
    coverage = coverage or s["coverage"]
    units, pairs = load_processed(cfg, source)
    dev = build_pairdata(cfg, units, pairs, judges, source, "dev")
    ids = [p for p, d in dev.items() if (tune_on == "all" or d.set == "close" or source == "mt")]
    rot = [j for j in cfg["judges_cfg"]["rotation_judges"] if j in judges]
    rng = np.random.default_rng(cfg["seed"] + 7)
    zs, diag = tune_zstar([dev[i] for i in ids], judges, rot, coverage, s["tune_budget"], rng, s["n_draws_point"])
    dd = pd.DataFrame(diag)
    ctx.table(dd, f"tune_zstar_{source}")
    unreachable = dd[~dd.coverage_ok]
    # ---- ordered probit per judge, development battles only ----
    fits, fits_re = {}, {}
    for j in judges:
        v = load_verdicts(cfg, j, source)
        v = v[v.unit_id.isin(units[units.split == "dev"].unit_id) & (v.verdict != "X")]
        v = v.merge(pairs[["pair_id", "signed_margin"]], on="pair_id")
        y = v.verdict.map({"B": 0, "T": 1, "A": 2}).to_numpy(); sgn = np.where(v.order == "AB", 1, -1)
        f = fit_probit(v.signed_margin.to_numpy(), sgn, y); fits[j] = f
        ctx.record(f"probit_fit/{source}/{j}", **{k: x for k, x in f.items() if k != "theta"})
        f1 = fit_probit_re(v.signed_margin.to_numpy(), sgn, y, v.unit_id.to_numpy()); fits_re[j] = f1
        ctx.record(f"probit_re_fit/{source}/{j}", **{k: x for k, x in f1.items() if k != "theta"})
    ctx.table(pd.DataFrame([{"judge": j, "model": "M0", **{k: x for k, x in f.items() if k != "theta"}} for j, f in fits.items()] +
                           [{"judge": j, "model": "M1_random_intercept", **{k: x for k, x in f.items() if k != "theta"}} for j, f in fits_re.items()]),
              f"probit_params_{source}")
    out = dict(source=source, time=now(), run=ctx.run_id, coverage=coverage, tune_budget=s["tune_budget"], tune_on=tune_on, judges=judges,
               rot=rot, n_pairs_tuned=len(ids), zstar=[dict(slot=k[0], protocol=k[1], zstar=v) for k, v in zs.items()],
               probit={j: f for j, f in fits.items()}, probit_re={j: f for j, f in fits_re.items()})
    tuned_path(cfg, source).write_text(json.dumps(out, indent=2))
    rows = [("Development pairs available", f"{len(ids)} pairs tuned ({tune_on})", len(ids) > 0),
            (f"Target coverage {coverage:.0%} reachable for every judge and protocol", f"{len(unreachable)} (slot, protocol) cells unreachable", len(unreachable) == 0),
            ("Probit fits converged", ", ".join(f"{j}:{f['converged']}" for j, f in fits.items()), all(f["converged"] for j, f in fits.items()))]
    verdict = "z* and probit parameters saved; next: `freeze`." if len(unreachable) == 0 else \
        f"Coverage {coverage:.0%} unreachable for some protocols: switch ALL protocols to the pre-specified fallback of {s['coverage_fallback']:.0%} (rerun tune --coverage {s['coverage_fallback']})."
    ctx.gate(f"Tuning ({source})", rows, verdict, fallback=f"coverage {s['coverage_fallback']:.0%} for all protocols")
    return out


def freeze(ctx, sources):
    """Hash every input of the primary analysis. The analyze stage refuses to run if these hashes change."""
    cfg = ctx.cfg
    files = ["configs/study.yaml", "configs/judges.yaml", cfg["prompts"]["single"], cfg["prompts"]["multi"]]
    files += [str(tuned_path(cfg, s)) for s in sources]
    snap = {f: sha256_file(f) for f in files}
    plan = dict(time=now(), run=ctx.run_id, files=snap, hypotheses=cfg["hypotheses"], seeds=cfg["generation"]["gen_seeds"])
    plan["plan_sha256"] = hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()
    p = Path(cfg["paths"]["processed"]) / "FROZEN.json"
    p.write_text(json.dumps(plan, indent=2))
    (ctx.dir / "FROZEN.json").write_text(json.dumps(plan, indent=2))
    log.info("analysis plan frozen: sha256=%s", plan["plan_sha256"])
    log.info("NEXT STEP: archive %s together with configs/ and prompts/ on Zenodo and paste the DOI into the paper BEFORE running `analyze`.", p)
    ctx.record("freeze", plan_sha256=plan["plan_sha256"])
    return plan


def check_frozen(cfg, sources):
    p = Path(cfg["paths"]["processed"]) / "FROZEN.json"
    if not p.exists():
        raise RuntimeError("analysis plan is not frozen; run `python -m judgestudy freeze` first (use --no-freeze-check only for exploration)")
    plan = json.loads(p.read_text())
    bad = [f for f, h in plan["files"].items() if not Path(f).exists() or sha256_file(f) != h]
    if bad:
        raise RuntimeError(f"files changed since the plan was frozen: {bad}. Document the deviation and refreeze, or restore the files.")
    return plan
