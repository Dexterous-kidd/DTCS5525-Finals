"""Verdict record -> per-pair tensors S[m, 2, 5] of canonical battle scores.

axis 1: order (0 = AB: canonical A shown first, 1 = BA)    axis 2: gen (0 = temperature 0, 1..4 = sampled)
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .prompts import SCORE
from .util import jsonl_read

log = logging.getLogger("judgestudy")


def load_verdicts(cfg, judge, source):
    rows = []
    for f in sorted((Path(cfg["paths"]["verdicts"]) / judge).glob(f"{source}.shard*.jsonl")):
        rows += jsonl_read(f)
    if not rows:
        raise FileNotFoundError(f"no verdicts for {judge}/{source}; run `generate` first")
    df = pd.DataFrame(rows).drop_duplicates(["unit_id", "order", "gen"], keep="last")
    return df


@dataclass
class PairData:
    pair_id: str
    unit_ids: list
    S: dict            # judge -> array (m, 2, 5)
    ds_order: np.ndarray   # dataset presentation order per unit: 0 = canonical A first
    signed_margin: float
    margin: float
    human_dec: int
    human_clear: bool
    set: str
    n_invalid: dict


def build_pairdata(cfg, units, pairs, judges, source, split, allow_missing=False):
    out = {}
    verd = {j: load_verdicts(cfg, j, source) for j in judges}
    pinfo = pairs.set_index("pair_id")
    for pid, g in units[units.split == split].groupby("pair_id"):
        g = g.sort_values("unit_id"); ids = g.unit_id.tolist(); idx = {u: i for i, u in enumerate(ids)}
        S, ninv = {}, {}
        for j in judges:
            v = verd[j]; v = v[v.unit_id.isin(idx)]
            arr = np.full((len(ids), 2, 5), np.nan)
            ri = v.unit_id.map(idx).to_numpy(); oi = (v.order == "BA").astype(int).to_numpy(); gi = v.gen.to_numpy()
            arr[ri, oi, gi] = v.verdict.map(SCORE).to_numpy()
            miss = int(np.isnan(arr).sum())
            if miss and not allow_missing:
                raise ValueError(f"{j}/{pid}: {miss} missing verdict cells; finish generation first")
            arr[np.isnan(arr)] = 0.5
            S[j] = arr; ninv[j] = int((v.verdict == "X").sum())
        p = pinfo.loc[pid]
        out[pid] = PairData(pid, ids, S, g.ds_flip.to_numpy().astype(int), float(p.signed_margin), float(p.margin),
                            int(p.human_dec), bool(p.human_clear), str(p.set), ninv)
    return out
