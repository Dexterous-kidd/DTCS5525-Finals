"""Verdict-record generation with append-only checkpointing, retries, sharding, and full logging.

Verdict line (public-safe, no text):   data/verdicts/<judge>/<source>.shardIofN.jsonl
Raw model output (private):            data/private/raw/<judge>/<source>.shardIofN.jsonl
Efficiency log (tokens, seconds, GPU): data/verdicts/<judge>/efficiency.jsonl
Header (versions, weights, GPU):       data/verdicts/<judge>/header.json
"""
import json, logging, time
from pathlib import Path

import numpy as np
import pandas as pd

from .judges import make_judge
from .prompts import parse_shown, render, to_canonical
from .util import jsonl_append, jsonl_read, stable_hash, now

log = logging.getLogger("judgestudy")


def verdict_dir(cfg, judge):
    return Path(cfg["paths"]["verdicts"]) / judge


def existing_keys(cfg, judge, source):
    keys = set()
    for f in verdict_dir(cfg, judge).glob(f"{source}.shard*.jsonl"):
        for r in jsonl_read(f):
            keys.add((r["unit_id"], r["order"], r["gen"]))
    return keys


def build_jobs(cfg, units, split, shard, n_shards, limit, seed):
    u = units if split == "all" else units[units.split == split]
    u = u.sort_values("unit_id")
    if n_shards > 1:
        u = u[[stable_hash(x) % n_shards == shard for x in u.unit_id]]
    if limit:
        rng = np.random.default_rng(seed)
        u = u.iloc[np.sort(rng.permutation(len(u))[:limit])]
    return u


def run_generation(ctx, judge_name, source, split="all", shard=0, n_shards=1, limit=None, mock=False, judge=None):
    cfg = ctx.cfg; g = cfg["generation"]
    units = pd.read_parquet(Path(cfg["paths"]["processed"]) / f"{source}_units.parquet")
    pairs = pd.read_parquet(Path(cfg["paths"]["processed"]) / f"{source}_pairs.parquet")
    signed = dict(zip(pairs.pair_id, pairs.signed_margin))
    u = build_jobs(cfg, units, split, shard, n_shards, limit, cfg["seed"])
    done = existing_keys(cfg, judge_name, source)
    gens = [0] + list(range(1, len(g["gen_seeds"]) + 1))
    seeds = {0: 0, **{i + 1: s for i, s in enumerate(g["gen_seeds"])}}
    jobs = [(r.unit_id, r.pair_id, r.split, o, k, r.payload) for r in u.itertuples() for o in g["orders"] for k in gens
            if (r.unit_id, o, k) not in done]
    total_all = len(u) * len(g["orders"]) * len(gens)
    log.info("[%s/%s] units=%d  total calls=%d  already done=%d  to run=%d  (shard %d/%d, split=%s, limit=%s)",
             judge_name, source, len(u), total_all, total_all - len(jobs), len(jobs), shard, n_shards, split, limit)
    if not jobs:
        return {"calls": 0}
    if judge is None:                      # a judge passed in by the caller is assumed already loaded
        judge = make_judge(cfg, judge_name, mock=mock)
        judge.load()
    vdir = verdict_dir(cfg, judge_name); vdir.mkdir(parents=True, exist_ok=True)
    hdr = vdir / "header.json"
    hdr.write_text(json.dumps({**judge.header(), "time": now()}, indent=2, default=str))
    vfile = vdir / f"{source}.shard{shard}of{n_shards}.jsonl"
    rfile = Path(cfg["paths"]["private"]) / "raw" / judge_name / f"{source}.shard{shard}of{n_shards}.jsonl"
    retry_ctr = g["retry_seed_start"] + len(jsonl_read(vdir / "retry_counter.jsonl"))
    t_start = time.time(); stats = dict(calls=0, invalid=0, truncated=0, retried=0, n_in=0, n_out=0)
    chunk = g["chunk_size"]
    for i in range(0, len(jobs), chunk):
        batch = jobs[i:i + chunk]; tb = time.time()
        reqs = [dict(messages=render(p, o, cfg), temperature=0 if k == 0 else 1, seed=seeds[k],
                     meta=dict(unit_id=uid, order=o, m=signed[pid])) for uid, pid, sp, o, k, p in batch]
        outs = judge.generate(reqs)
        rows, raws, retry_idx = [], [], []
        for j, ((uid, pid, sp, o, k, _), req, out) in enumerate(zip(batch, reqs, outs)):
            shown = parse_shown(out["text"])
            rows.append(dict(judge=judge_name, source=source, unit_id=uid, pair_id=pid, split=sp, order=o, gen=k, seed=req["seed"],
                             temp=req["temperature"], shown=shown, verdict=to_canonical(shown, o), valid=shown is not None,
                             truncated=out["finish_reason"] == "length", n_in=out["n_in"], n_out=out["n_out"],
                             retried=False, retry_seed=None))
            raws.append(dict(unit_id=uid, order=o, gen=k, text=out["text"]))
            if shown is None:
                retry_idx.append(j)
        # one retry for invalid outputs, seed from the reserved pool (101, 102, ...)
        if retry_idx:
            rreqs = []
            for j in retry_idx:
                r = dict(reqs[j]); r["temperature"] = 1; r["seed"] = retry_ctr; retry_ctr += 1
                rreqs.append(r)
            jsonl_append(vdir / "retry_counter.jsonl", [{"seed": r["seed"]} for r in rreqs])
            for j, r, out in zip(retry_idx, rreqs, judge.generate(rreqs)):
                shown = parse_shown(out["text"])
                o = rows[j]["order"]
                rows[j].update(shown=shown, verdict=to_canonical(shown, o), valid=shown is not None, retried=True, retry_seed=r["seed"],
                               truncated=rows[j]["truncated"] or out["finish_reason"] == "length",
                               n_in=rows[j]["n_in"] + out["n_in"], n_out=rows[j]["n_out"] + out["n_out"])
                raws[j]["text_retry"] = out["text"]
        jsonl_append(vfile, rows); jsonl_append(rfile, raws)
        dt = time.time() - tb
        stats["calls"] += len(rows); stats["invalid"] += sum(not r["valid"] for r in rows)
        stats["truncated"] += sum(r["truncated"] for r in rows); stats["retried"] += len(retry_idx)
        stats["n_in"] += sum(r["n_in"] for r in rows); stats["n_out"] += sum(r["n_out"] for r in rows)
        el = time.time() - t_start; rate = stats["calls"] / el; left = (len(jobs) - stats["calls"]) / max(rate, 1e-9)
        jsonl_append(vdir / "efficiency.jsonl", [dict(time=now(), judge=judge_name, source=source, shard=shard, calls=len(rows),
                     n_in=sum(r["n_in"] for r in rows), n_out=sum(r["n_out"] for r in rows), seconds=round(dt, 2),
                     invalid=sum(not r["valid"] for r in rows), retried=len(retry_idx), gpu=judge.header().get("gpu"))])
        log.info("[%s/%s] %d/%d calls | %.1f calls/s | batch invalid(after retry)=%d truncated=%d | ETA %.1f min",
                 judge_name, source, stats["calls"], len(jobs), rate, sum(not r["valid"] for r in rows),
                 sum(r["truncated"] for r in rows), left / 60)
    stats["seconds"] = round(time.time() - t_start, 1)
    stats["invalid_rate"] = stats["invalid"] / max(stats["calls"], 1)
    stats["truncated_rate"] = stats["truncated"] / max(stats["calls"], 1)
    ctx.record(f"generate/{judge_name}/{source}", split=split, shard=f"{shard}/{n_shards}", **stats)
    return stats
