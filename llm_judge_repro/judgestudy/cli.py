"""Command line: python -m judgestudy <stage> [options].  See README.md for the recommended order."""
import argparse, json, shutil, sys
from pathlib import Path

import pandas as pd

from .util import load_cfg


def _ctx(args, cfg, label):
    from .runctx import RunContext
    return RunContext(cfg, run_id=args.run_id, label=label, args=vars(args))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="judgestudy")
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--run-id", default=None, help="reuse a run folder (append logs/results) instead of creating a new one")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "prepare"):
        p = sub.add_parser(name); p.add_argument("--source", choices=["arena", "mt", "both"], default="both")
        if name == "prepare":
            p.add_argument("--min-battles", type=int, default=None, help="contingency: lower the per-pair battle threshold (e.g. 120)")
    g = sub.add_parser("generate"); g.add_argument("--judge", required=True); g.add_argument("--source", choices=["arena", "mt"], required=True)
    g.add_argument("--split", choices=["dev", "test", "all"], default="all"); g.add_argument("--shard", type=int, default=0)
    g.add_argument("--n-shards", type=int, default=1); g.add_argument("--limit", type=int, default=None, help="pilot: random subset of units")
    g.add_argument("--mock", action="store_true")
    c = sub.add_parser("checks"); c.add_argument("--judge", required=True); c.add_argument("--kind", choices=["determinism", "contamination", "both"], default="both"); c.add_argument("--mock", action="store_true")
    t = sub.add_parser("tune"); t.add_argument("--source", choices=["arena", "mt"], required=True); t.add_argument("--judges", nargs="*"); t.add_argument("--coverage", type=float, default=None)
    f = sub.add_parser("freeze"); f.add_argument("--sources", nargs="*", default=["arena", "mt"])
    a = sub.add_parser("analyze"); a.add_argument("--source", choices=["arena", "mt"], required=True); a.add_argument("--judges", nargs="*")
    a.add_argument("--no-freeze-check", action="store_true"); a.add_argument("--quick", action="store_true", help="tiny bootstrap for debugging only")
    sn = sub.add_parser("sensitivity", help="compare a sensitivity judge (e.g. qwen3-8b-think) with its baseline on the units it covered")
    sn.add_argument("--source", choices=["arena", "mt"], default="arena"); sn.add_argument("--judge", default="qwen3-8b-think"); sn.add_argument("--baseline", default="qwen3-8b")
    sub.add_parser("report")
    r = sub.add_parser("release"); r.add_argument("--out", default="release")
    s = sub.add_parser("smoke", help="end-to-end test on synthetic data with mock judges (no GPU, no downloads)")
    s.add_argument("--workdir", default="smoke_run")
    args = ap.parse_args(argv)

    if args.cmd == "smoke":
        from .smoke import run_smoke
        return run_smoke(args)

    cfg = load_cfg(args.config)
    if args.cmd == "prepare" and args.min_battles:
        cfg["arena"]["min_battles_per_pair"] = args.min_battles
        cfg["arena"]["battles_per_pair"] = min(cfg["arena"]["battles_per_pair"], args.min_battles)
    ctx = _ctx(args, cfg, args.cmd + (f"-{getattr(args, 'judge', '')}" if hasattr(args, "judge") else ""))
    from . import data
    if args.cmd == "inspect":
        with ctx.stage("inspect"):
            if args.source in ("arena", "both"): data.inspect_arena(cfg)
            if args.source in ("mt", "both"): data.inspect_mt(cfg)
    elif args.cmd == "prepare":
        with ctx.stage("prepare"):
            if args.source in ("arena", "both"):
                u, p, fn = data.prep_arena(cfg); data.save_processed(cfg, "arena", u, p, fn)
                ctx.table(fn, "arena_eligibility_funnel"); ctx.table(p, "arena_selected_pairs"); ctx.table(u.groupby(["set", "split"]).size().reset_index(name="battles"), "arena_split_sizes")
                n_el = int(fn.n_pairs_ge_min.iloc[-2]) if len(fn) > 1 else 0
                ctx.gate("Foundations: Arena eligibility", [(f">= {cfg['arena']['gate_min_pairs']} eligible pairs", f"{n_el} eligible pairs", n_el >= cfg["arena"]["gate_min_pairs"])],
                         "continue" if n_el >= cfg["arena"]["gate_min_pairs"] else "lower the threshold to 120 battles (re-run prepare --min-battles 120)")
            if args.source in ("mt", "both"):
                u, p, fn, ceil = data.prep_mt(cfg); data.save_processed(cfg, "mt", u, p, fn, extra={"human_ceiling": ceil})
                ctx.table(fn, "mt_funnel"); ctx.table(p, "mt_pairs"); ctx.record("mt_human_ceiling", human_ceiling=ceil)
    elif args.cmd == "generate":
        from .generate import run_generation
        with ctx.stage(f"generate/{args.judge}/{args.source}"):
            run_generation(ctx, args.judge, args.source, args.split, args.shard, args.n_shards, args.limit, args.mock)
    elif args.cmd == "checks":
        from .checks import run_checks
        with ctx.stage(f"checks/{args.judge}"):
            run_checks(ctx, args.judge, ("determinism", "contamination") if args.kind == "both" else (args.kind,), args.mock)
    elif args.cmd == "tune":
        from .tuning import all_judges, tune
        with ctx.stage(f"tune/{args.source}"):
            tune(ctx, args.source, args.judges or all_judges(cfg, True, args.source), args.coverage)
    elif args.cmd == "freeze":
        from .tuning import freeze
        with ctx.stage("freeze"):
            freeze(ctx, args.sources)
    elif args.cmd == "analyze":
        from .analysis import analyze
        with ctx.stage(f"analyze/{args.source}"):
            analyze(ctx, args.source, args.judges, args.no_freeze_check, quick=args.quick)
    elif args.cmd == "sensitivity":
        from .analysis import sensitivity
        with ctx.stage(f"sensitivity/{args.judge}"):
            sensitivity(ctx, args.source, args.judge, args.baseline)
    elif args.cmd == "report":
        from .report import summary
        summary(ctx)
    elif args.cmd == "release":
        _release(cfg, args.out)


def _release(cfg, out):
    """Public-safe release: verdict labels and ids only (no prompts, responses or model explanations)."""
    out = Path(out); out.mkdir(exist_ok=True, parents=True)
    for jd in sorted(Path(cfg["paths"]["verdicts"]).iterdir()):
        if jd.is_dir():
            from .util import jsonl_read
            rows = []
            for fp in sorted(jd.glob("*.shard*.jsonl")):
                rows += jsonl_read(fp)
            if rows:
                pd.DataFrame(rows).to_csv(out / f"verdicts_{jd.name}.csv.gz", index=False)
            if (jd / "header.json").exists():
                shutil.copy(jd / "header.json", out / f"header_{jd.name}.json")
    proc = Path(cfg["paths"]["processed"])
    for s in ("arena", "mt"):
        if (proc / f"{s}_units.parquet").exists():
            u = pd.read_parquet(proc / f"{s}_units.parquet").drop(columns=["payload"], errors="ignore")
            u.to_csv(out / f"{s}_units_ids_only.csv.gz", index=False)
            pd.read_parquet(proc / f"{s}_pairs.parquet").to_csv(out / f"{s}_pairs.csv", index=False)
    print("release written to", out, "(check licenses: outputs of third-party systems are NOT included)")


if __name__ == "__main__":
    main()
