"""Data preparation: Arena battles and MT-Bench expert items -> a common 'units' table.

units columns: source, unit_id, pair_id, split(dev|test), set(close|far), human(A|B|T), ds_flip(0|1),
               payload(JSON of canonical text fields), n_tokens
pairs columns: source, pair_id, sys_A, sys_B, n_eligible, winrate, ci_lo, ci_hi, margin, signed_margin,
               human_clear, human_dec, set
"""
import glob, json, logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("judgestudy")

WINNER_SCORE = {"A": 1.0, "T": 0.5, "B": 0.0}
MT_CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]


# ---------------------------------------------------------------- text helpers
def _text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _text(content.get("text") if content.get("text") is not None else content.get("content"))
    if isinstance(content, (list, tuple, np.ndarray)):
        return "\n".join(t for t in (_text(c) for c in list(content)) if t)
    return str(content)


def conv_msgs(conv):
    return [(str(m.get("role", "")), _text(m.get("content"))) for m in list(conv)]


def read_parquets(path, must_contain=None):
    p = Path(path)
    files = [p] if p.is_file() else sorted(Path(f) for f in glob.glob(str(p / "**" / "*.parquet"), recursive=True))
    if must_contain:
        files = [f for f in files if must_contain in str(f)]
    if not files:
        raise FileNotFoundError(f"no parquet files under {path} (filter={must_contain})")
    log.info("reading %d parquet file(s) from %s", len(files), path)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def make_token_counter(name):
    if name in (None, "approx"):
        return lambda texts: np.array([max(1, len(t) // 4) for t in texts])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    def count(texts):
        out = []
        for i in range(0, len(texts), 2048):
            enc = tok(list(texts[i:i + 2048]), add_special_tokens=False)["input_ids"]
            out.extend(len(e) for e in enc)
        return np.array(out)
    return count


def winrate_ci(scores_by_pair, n_boot, rng):
    """scores_by_pair: dict pair_id -> np.array of {1,.5,0}. Returns dict -> (wr, lo, hi)."""
    out = {}
    for pid, s in scores_by_pair.items():
        wr = s.mean()
        bs = s[rng.integers(0, len(s), size=(n_boot, len(s)))].mean(1)
        out[pid] = (wr, np.quantile(bs, 0.025), np.quantile(bs, 0.975))
    return out


def _pair_table(units, source, sys_names, n_boot, rng, set_by_pair):
    rows = []
    sc = {pid: g["human"].map(WINNER_SCORE).to_numpy() for pid, g in units.groupby("pair_id")}
    ci = winrate_ci(sc, n_boot, rng)
    for pid, (wr, lo, hi) in ci.items():
        a, b = sys_names[pid]
        clear = (lo > 0.5) or (hi < 0.5)
        rows.append(dict(source=source, pair_id=pid, sys_A=a, sys_B=b, n_eligible=len(sc[pid]), winrate=wr,
                         ci_lo=lo, ci_hi=hi, margin=abs(wr - 0.5), signed_margin=wr - 0.5, human_clear=bool(clear),
                         human_dec=int(np.sign(wr - 0.5)) if clear else 0, set=set_by_pair.get(pid, "")))
    return pd.DataFrame(rows)


def _funnel_row(step, df, minb):
    n_pairs = df.groupby("pair_key").ngroups if len(df) else 0
    ge = int((df.groupby("pair_key").size() >= minb).sum()) if len(df) else 0
    return dict(step=step, n_battles=len(df), n_pairs=n_pairs, n_pairs_ge_min=ge)


# ---------------------------------------------------------------- Arena
def inspect_arena(cfg):
    df = read_parquets(cfg["paths"]["raw_arena"])
    cols = cfg["arena"]["columns"]
    log.info("ARENA schema: %s", {c: str(df[c].dtype) for c in df.columns})
    log.info("ARENA rows: %d", len(df))
    missing = [v for k, v in cols.items() if v not in df.columns and k != "language"]
    if missing:
        log.warning("configured columns missing from the release: %s -> edit arena.columns in configs/study.yaml", missing)
    if cols["language"] not in df.columns:
        log.warning("no language column: the English-only filter will be skipped and this will be stated in Methods")
    log.info("ARENA winner values: %s", df[cols["winner"]].value_counts().to_dict())
    log.info("ARENA number of systems: %d", len(set(df[cols["model_a"]]) | set(df[cols["model_b"]])))
    log.info("ARENA first record (truncated): %s", str(df.iloc[0].to_dict())[:1500])
    return df


def prep_arena(cfg, raw=None):
    a = cfg["arena"]; c = a["columns"]; seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    df = raw if raw is not None else read_parquets(cfg["paths"]["raw_arena"])
    minb = a["min_battles_per_pair"]
    funnel = []
    d = pd.DataFrame({"battle_id": df[c["id"]].astype(str), "ma": df[c["model_a"]].astype(str), "mb": df[c["model_b"]].astype(str),
                      "winner": df[c["winner"]].astype(str), "conv_a": df[c["conv_a"]], "conv_b": df[c["conv_b"]]})
    d["language"] = df[c["language"]].astype(str) if c["language"] in df.columns else "unknown"
    d["pair_key"] = [("||".join(sorted([x, y]))) for x, y in zip(d.ma, d.mb)]
    funnel.append(_funnel_row("0 raw release", d, minb))

    # 1 single-turn
    def single(conv):
        m = conv_msgs(conv)
        return sum(r == "user" for r, _ in m) == 1 and sum(r == "assistant" for r, _ in m) == 1
    keep = np.array([single(x) and single(y) for x, y in zip(d.conv_a, d.conv_b)])
    d = d[keep].copy(); funnel.append(_funnel_row("1 single-turn", d, minb))

    # 2 verdict map
    def wmap(w):
        w = w.lower()
        if w == "model_a": return "A"
        if w == "model_b": return "B"
        if w.startswith("tie"): return "T"
        return None
    d["human_ds"] = d.winner.map(wmap)
    d = d[d.human_ds.notna()].copy(); funnel.append(_funnel_row("2 verdict in {A,B,tie variants}", d, minb))

    # 3 same-family exclusion
    fam = [f.lower() for f in a["excluded_families"]]
    bad = d.ma.str.lower().apply(lambda s: any(f in s for f in fam)) | d.mb.str.lower().apply(lambda s: any(f in s for f in fam))
    d = d[~bad].copy(); funnel.append(_funnel_row("3 same-family systems removed", d, minb))

    # extract text once
    d["prompt"] = [next((t for r, t in conv_msgs(x) if r == "user"), "") for x in d.conv_a]
    d["resp_a"] = [next((t for r, t in conv_msgs(x) if r == "assistant"), "") for x in d.conv_a]
    d["resp_b"] = [next((t for r, t in conv_msgs(x) if r == "assistant"), "") for x in d.conv_b]
    d = d[(d.prompt.str.len() > 0) & (d.resp_a.str.len() > 0) & (d.resp_b.str.len() > 0)].copy()

    # 4 token length
    counter = make_token_counter(a["length_tokenizer"])
    d["n_tokens"] = counter(d.prompt.tolist()) + counter(d.resp_a.tolist()) + counter(d.resp_b.tolist())
    d = d[d.n_tokens <= a["max_tokens"]].copy(); funnel.append(_funnel_row(f"4 <= {a['max_tokens']} tokens", d, minb))

    # 5 duplicate prompts
    d["pnorm"] = d.prompt.str.lower().str.split().str.join(" ")
    d = d.sample(frac=1.0, random_state=seed).drop_duplicates("pnorm").copy()
    funnel.append(_funnel_row("5 duplicate prompts removed", d, minb))

    # 6 language
    if a["english_only"] and (d.language != "unknown").any():
        d = d[d.language.str.lower().isin(["english", "en"])].copy()
        funnel.append(_funnel_row("6 English only", d, minb))
    else:
        log.warning("language filter skipped (no language column or english_only=false)")
        funnel.append(_funnel_row("6 language filter skipped", d, minb))

    # canonical orientation: sys_A = alphabetically first name
    d["sys_A"] = d.pair_key.str.split("||", regex=False).str[0]
    d["sys_B"] = d.pair_key.str.split("||", regex=False).str[1]
    flip = (d.ma != d.sys_A).astype(int)             # dataset showed canonical B first
    d["ds_flip"] = flip
    d["resp_A"] = np.where(flip == 1, d.resp_b, d.resp_a)
    d["resp_B"] = np.where(flip == 1, d.resp_a, d.resp_b)
    inv = {"A": "B", "B": "A", "T": "T"}
    d["human"] = [inv[h] if f == 1 else h for h, f in zip(d.human_ds, flip)]

    # eligibility, margins, selection
    cnt = d.groupby("pair_key").size()
    elig = cnt[cnt >= minb].index
    log.info("eligible pairs (>= %d battles): %d", minb, len(elig))
    de = d[d.pair_key.isin(elig)].copy()
    names = {k: tuple(k.split("||")) for k in elig}
    pid_of = {k: f"A{i:03d}" for i, k in enumerate(sorted(elig))}
    de["pair_id"] = de.pair_key.map(pid_of)
    sys_names = {pid_of[k]: names[k] for k in elig}
    tmp = _pair_table(de.assign(unit_id=de.battle_id), "arena", sys_names, a["margin_bootstrap"], rng, {})
    tmp = tmp.sort_values("margin", kind="stable").reset_index(drop=True)
    n_close, n_far = a["n_close"], a["n_far"]
    if len(tmp) >= n_close + n_far:
        close_ids, far_ids = tmp.pair_id[:n_close].tolist(), tmp.pair_id[-n_far:].tolist()
    else:
        half = len(tmp) // 2
        close_ids, far_ids = tmp.pair_id[:half].tolist(), tmp.pair_id[half:].tolist()
        log.warning("fewer than %d eligible pairs: all %d used, split at the median margin", n_close + n_far, len(tmp))
    set_by_pair = {**{p: "close" for p in close_ids}, **{p: "far" for p in far_ids}}
    funnel.append(dict(step="7 selected pairs", n_battles=int(len(close_ids + far_ids) * a["battles_per_pair"]),
                       n_pairs=len(set_by_pair), n_pairs_ge_min=len(set_by_pair)))
    pairs = tmp[tmp.pair_id.isin(set_by_pair)].copy()
    pairs["set"] = pairs.pair_id.map(set_by_pair)

    # sampling and dev/test split
    rows = []
    for pid in sorted(set_by_pair):
        g = de[de.pair_id == pid]
        take = g.iloc[rng.permutation(len(g))[:a["battles_per_pair"]]]
        for j, (_, r) in enumerate(take.iterrows()):
            payload = dict(kind="single", question=r.prompt, resp_A=r.resp_A, resp_B=r.resp_B)
            rows.append(dict(source="arena", unit_id=f"arena:{r.battle_id}", pair_id=pid, split="dev" if j < a["n_dev"] else "test",
                             set=set_by_pair[pid], human=r.human, ds_flip=int(r.ds_flip), payload=json.dumps(payload),
                             n_tokens=int(r.n_tokens), language=r.language))
    units = pd.DataFrame(rows)
    # human win rates of the selected pairs are recomputed on ALL eligible battles (not only the 160 sampled)
    return units, pairs, pd.DataFrame(funnel)


# ---------------------------------------------------------------- MT-Bench
def inspect_mt(cfg):
    df = read_parquets(cfg["paths"]["raw_mt"], must_contain="human")
    log.info("MT-Bench human split schema: %s", {c: str(df[c].dtype) for c in df.columns})
    log.info("MT-Bench rows: %d | winner values: %s", len(df), df["winner"].value_counts().to_dict())
    return df


def prep_mt(cfg, raw=None):
    seed = cfg["seed"]; rng = np.random.default_rng(seed + 1)
    df = raw if raw is not None else read_parquets(cfg["paths"]["raw_mt"], must_contain="human")
    rows = []
    for r in df.itertuples(index=False):
        a, b = sorted([r.model_a, r.model_b])
        w = str(r.winner).lower()
        v = "T" if w.startswith("tie") else ("A" if (w == "model_a") == (r.model_a == a) else "B")
        ca, cb = conv_msgs(r.conversation_a), conv_msgs(r.conversation_b)
        if r.model_a != a:
            ca, cb = cb, ca
        rows.append(dict(qid=int(r.question_id), sys_A=a, sys_B=b, turn=int(r.turn), vote=v, ca=ca, cb=cb,
                         annot=str(getattr(r, "judge", ""))))
    v = pd.DataFrame(rows)
    items = []
    for (qid, a, b, turn), g in v.groupby(["qid", "sys_A", "sys_B", "turn"]):
        votes = g.vote.tolist()
        cnt = {k: votes.count(k) for k in "ABT"}
        top = max(cnt.values()); winners = [k for k, n in cnt.items() if n == top]
        label = winners[0] if len(winners) == 1 else "T"          # split evenly -> tie
        ca, cb = g.ca.iloc[0], g.cb.iloc[0]
        ua = [t for r_, t in ca if r_ == "user"]; aa = [t for r_, t in ca if r_ == "assistant"]
        ub = [t for r_, t in cb if r_ == "user"]; ab = [t for r_, t in cb if r_ == "assistant"]
        if turn == 1:
            payload = dict(kind="single", question=ua[0], resp_A=aa[0], resp_B=ab[0])
        else:
            if len(ua) < 2 or len(aa) < 2 or len(ab) < 2:
                continue
            payload = dict(kind="multi", q1=ua[0], q2=ua[1], a1_A=aa[0], a2_A=aa[1], a1_B=ab[0], a2_B=ab[1])
        items.append(dict(source="mt", unit_id=f"mt:{qid}:{a}:{b}:{turn}", qid=qid, sys_A=a, sys_B=b, turn=turn, human=label,
                          ds_flip=0, payload=json.dumps(payload), n_votes=len(votes), votes="".join(votes), n_tokens=0))
    u = pd.DataFrame(items)
    keys = sorted(set(zip(u.sys_A, u.sys_B)))
    pid_of = {k: f"M{i:02d}" for i, k in enumerate(keys)}
    u["pair_id"] = [pid_of[(a, b)] for a, b in zip(u.sys_A, u.sys_B)]
    u["category"] = [MT_CATEGORIES[min((q - 81) // 10, 7)] for q in u.qid]
    # split by question within category
    m = cfg["mt"]; split_of = {}
    for cat in MT_CATEGORIES:
        qs = sorted(u[u.category == cat].qid.unique())
        qs = [qs[i] for i in rng.permutation(len(qs))]
        for i, q in enumerate(qs):
            split_of[q] = "dev" if i < m["n_dev_q_per_cat"] else ("test" if i < m["n_dev_q_per_cat"] + m["n_test_q_per_cat"] else "unused")
    u["split"] = u.qid.map(split_of)
    u = u[u.split != "unused"].copy()
    sys_names = {pid_of[k]: k for k in keys}
    pairs = _pair_table(u, "mt", sys_names, cfg["arena"]["margin_bootstrap"], rng, {})
    med = pairs.margin.median()
    pairs["set"] = np.where(pairs.margin <= med, "close", "far")
    u["set"] = u.pair_id.map(dict(zip(pairs.pair_id, pairs.set)))
    # human-to-human agreement ceiling (items with >= 2 votes)
    agree = []
    for votes in u.votes[u.n_votes >= 2]:
        n = len(votes)
        agree += [votes[i] == votes[j] for i in range(n) for j in range(i + 1, n)]
    ceiling = float(np.mean(agree)) if agree else float("nan")
    funnel = pd.DataFrame([dict(step="MT-Bench items (>=1 expert vote)", n_battles=len(u), n_pairs=len(pairs), n_pairs_ge_min=len(pairs)),
                           dict(step="human-to-human pairwise agreement ceiling", n_battles=len(agree), n_pairs=0, n_pairs_ge_min=0)])
    log.info("MT-Bench human-to-human agreement ceiling: %.4f (%d vote pairs)", ceiling, len(agree))
    u = u.drop(columns=["votes"])
    return u, pairs, funnel, ceiling


# ---------------------------------------------------------------- I/O
def save_processed(cfg, source, units, pairs, funnel, extra=None):
    out = Path(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
    units.to_parquet(out / f"{source}_units.parquet"); pairs.to_parquet(out / f"{source}_pairs.parquet")
    funnel.to_csv(out / f"{source}_funnel.csv", index=False)
    if extra:
        (out / f"{source}_extra.json").write_text(json.dumps(extra))


def load_processed(cfg, source):
    out = Path(cfg["paths"]["processed"])
    return pd.read_parquet(out / f"{source}_units.parquet"), pd.read_parquet(out / f"{source}_pairs.parquet")
