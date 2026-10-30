"""Progress report for the verdict record. Run from the repo root:  python scripts/status.py
Shows, per judge/source/split: unique calls done vs expected, duplicates, invalid and truncated rates, last write time.
Run it twice a few minutes apart: if 'done' does not grow, nothing is generating."""
import argparse, glob, json, os, sys, time
from pathlib import Path

import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/study.yaml")
ap.add_argument("--verdicts"); ap.add_argument("--processed")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)) if Path(a.config).exists() else {"paths": {}}
vdir = Path(a.verdicts or cfg["paths"].get("verdicts", "data/verdicts")); pdir = Path(a.processed or cfg["paths"].get("processed", "data/processed"))
if not vdir.exists():
    sys.exit(f"{vdir} not found: run this from the llm_judge_repro folder (the one containing configs/ and data/), not its parent.")
CALLS = 10                                    # 2 orders x 5 generations
exp = {}
for s in ("arena", "mt"):
    p = pdir / f"{s}_units.parquet"
    if p.exists():
        u = pd.read_parquet(p, columns=["unit_id", "split"]); exp[s] = {sp: len(g) * CALLS for sp, g in u.groupby("split")}
        exp[s]["unit_split"] = dict(zip(u.unit_id, u.split))
rows = []
for jd in sorted(p for p in vdir.iterdir() if p.is_dir()):
    for s in ("arena", "mt"):
        files = sorted(glob.glob(str(jd / f"{s}.shard*.jsonl")))
        if not files:
            continue
        keys, n, inv, trunc, last = {}, 0, 0, 0, 0.0
        for f in files:
            last = max(last, os.path.getmtime(f))
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n += 1; keys[(r["unit_id"], r["order"], r["gen"])] = r
        for r in keys.values():
            inv += r["verdict"] == "X"; trunc += bool(r["truncated"])
        by = {}
        for (uid, _, _), r in keys.items():
            by[r["split"]] = by.get(r["split"], 0) + 1
        for sp in sorted(by):
            e = exp.get(s, {}).get(sp)
            rows.append(dict(judge=jd.name, source=s, split=sp, done=by[sp], expected=e, pct=round(100 * by[sp] / e, 1) if e else None,
                             duplicates=n - len(keys) if sp == sorted(by)[0] else "", invalid_pct=round(100 * inv / max(len(keys), 1), 2),
                             truncated_pct=round(100 * trunc / max(len(keys), 1), 2), last_write_min_ago=round((time.time() - last) / 60, 1)))
print(pd.DataFrame(rows).to_string(index=False) if rows else "no verdict files yet")
print("\nnow:", time.strftime("%H:%M:%S"), "| done = unique (unit, order, generation) calls saved. A pilot with --limit shows pct below 100 by design.")
