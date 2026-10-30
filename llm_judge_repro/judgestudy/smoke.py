"""End-to-end smoke test: synthetic data + mock judges + the real code path for every stage."""
import os, shutil, sys
from pathlib import Path

from .util import load_cfg


def run_smoke(args):
    work = Path(args.workdir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cfg = load_cfg(args.__dict__.get("config", "configs/study.yaml") if isinstance(args.__dict__, dict) else "configs/study.yaml", smoke=True)
    cfg["paths"] = {k: str(work / v) for k, v in dict(raw_arena="raw_arena", raw_mt="raw_mt", processed="processed", verdicts="verdicts",
                                                     private="private", results="results").items()}
    cfg["judges_cfg"] = {"rotation_judges": ["mock-a", "mock-b"], "judges": {
        "mock-a": {"family": "mock", "core": True, "backend": "mock"}, "mock-b": {"family": "mock", "core": True, "backend": "mock"},
        "mock-c": {"family": "mock", "core": False, "backend": "mock"}}}
    cfg["generation"]["chunk_size"] = 4000
    from . import data
    from .analysis import analyze
    from .checks import run_checks
    from .generate import run_generation
    from .mock import make_mock_arena, make_mock_mt
    from .report import summary
    from .runctx import RunContext
    from .tuning import freeze, tune
    ctx = RunContext(cfg, run_id="smoke", label="smoke", args={"smoke": True})
    judges = ["mock-a", "mock-b", "mock-c"]
    with ctx.stage("make_mock_data"):
        ctx.log.info("mock arena battles: %d", make_mock_arena(cfg["paths"]["raw_arena"]))
        ctx.log.info("mock MT-Bench rows: %d", make_mock_mt(cfg["paths"]["raw_mt"]))
    with ctx.stage("inspect"):
        data.inspect_arena(cfg); data.inspect_mt(cfg)
    with ctx.stage("prepare"):
        u, p, fn = data.prep_arena(cfg); data.save_processed(cfg, "arena", u, p, fn); ctx.table(fn, "arena_eligibility_funnel")
        ctx.table(p, "arena_selected_pairs", show=False)
        u, p, fn, ceil = data.prep_mt(cfg); data.save_processed(cfg, "mt", u, p, fn, extra={"human_ceiling": ceil}); ctx.table(fn, "mt_funnel")
    for src in ("arena", "mt"):
        for j in judges:
            with ctx.stage(f"generate/{j}/{src}"):
                run_generation(ctx, j, src, "all", mock=True)
    # resume test: running again must find nothing left to do
    run_generation(ctx, "mock-a", "arena", "all", mock=True)
    with ctx.stage("checks"):
        run_checks(ctx, "mock-a", mock=True)
    for src in ("arena", "mt"):
        with ctx.stage(f"tune/{src}"):
            tune(ctx, src, judges)
    freeze(ctx, ["arena", "mt"])
    res = {}
    for src in ("arena", "mt"):
        with ctx.stage(f"analyze/{src}"):
            res[src] = analyze(ctx, src, judges)
    from .analysis import anchor_replication
    anchor_replication(ctx, res)
    from .analysis import sensitivity
    with ctx.stage("sensitivity"):
        sensitivity(ctx, "arena", "mock-c", "mock-a")
    summary(ctx)
    ctx.log.info("SMOKE TEST COMPLETE. Verdicts: %s", {s: r["verdicts"] for s, r in res.items()})
    return 0
