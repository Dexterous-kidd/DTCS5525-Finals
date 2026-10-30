"""Synthetic Arena-like and MT-Bench-like raw data (for smoke tests only; never mixed with real data)."""
from pathlib import Path

import numpy as np
import pandas as pd


def _sig(x):
    return 1 / (1 + np.exp(-x))


def make_mock_arena(path, seed=0, per_pair=420):
    rng = np.random.default_rng(seed)
    models = [f"model-{c}" for c in "abcdefghijkl"] + ["gpt-oss-20b-x", "qwen3-8b-x"]
    strength = {m: s for m, s in zip(models, np.linspace(0, 1.4, len(models)))}
    rows, bid = [], 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            for _ in range(per_pair):
                a, b = (models[i], models[j]) if rng.random() < 0.5 else (models[j], models[i])
                p = _sig(strength[a] - strength[b])
                u = rng.random()
                w = "tie" if u < 0.12 else ("tie (bothbad)" if u < 0.16 else ("model_a" if rng.random() < p else "model_b"))
                q = f"Question {bid} about topic {rng.integers(0, 500)}: explain the idea briefly."
                ra = f"Answer from {a} to {bid}. " + "Detail. " * int(rng.integers(3, 40))
                rb = f"Answer from {b} to {bid}. " + "Detail. " * int(rng.integers(3, 40))
                rows.append(dict(id=f"b{bid}", model_a=a, model_b=b, winner=w, language="English" if rng.random() < 0.9 else "Spanish",
                                 conversation_a=[{"role": "user", "content": q}, {"role": "assistant", "content": ra}],
                                 conversation_b=[{"role": "user", "content": q}, {"role": "assistant", "content": rb}]))
                bid += 1
    Path(path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(Path(path) / "train-00000.parquet")
    return len(rows)


def make_mock_mt(path, seed=1):
    rng = np.random.default_rng(seed)
    models = ["sys-1", "sys-2", "sys-3", "sys-4", "sys-5", "sys-6"]
    strength = dict(zip(models, np.linspace(0, 2.0, 6)))
    rows = []
    for qid in range(81, 161):
        for i in range(6):
            for j in range(i + 1, 6):
                a, b = (models[i], models[j]) if rng.random() < 0.5 else (models[j], models[i])
                for turn in (1, 2):
                    for k in range(int(rng.integers(1, 4))):
                        p = _sig(strength[a] - strength[b]); u = rng.random()
                        w = "tie" if u < 0.1 else ("model_a" if rng.random() < p else "model_b")
                        def conv(m):
                            c = [{"role": "user", "content": f"Q{qid} first question"}, {"role": "assistant", "content": f"{m} answer 1 to {qid}. " + "Text. " * 20}]
                            if turn == 2:
                                c += [{"role": "user", "content": f"Q{qid} follow-up"}, {"role": "assistant", "content": f"{m} answer 2 to {qid}. " + "Text. " * 20}]
                            return c
                        rows.append(dict(question_id=qid, model_a=a, model_b=b, winner=w, judge=f"expert-{k}", conversation_a=conv(a),
                                         conversation_b=conv(b), turn=turn))
    Path(path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(Path(path) / "human-00000.parquet")
    return len(rows)
