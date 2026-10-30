"""Two-level cluster bootstrap driver: battles are resampled within each pair, pairs are resampled at aggregation."""
import logging, multiprocessing as mp, time

import numpy as np

from .sim import decomp_pair, simulate_pair

log = logging.getLogger("judgestudy")
_ST = {}


def _init(st):
    global _ST
    _ST = st


def _one(args):
    b, seed = args; st = _ST; rng = np.random.default_rng(seed)
    P_list = st["pairs"]; ns = len(st["judges"]) + 1
    cells = np.zeros((len(P_list), ns, len(st["protos"]), len(st["budgets"]), 7), dtype=np.int32)
    dec = np.zeros((len(P_list), len(st["judges"]), 4, len(st["budgets"]), 7), dtype=np.int32)
    for i, P in enumerate(P_list):
        m = len(P.unit_ids); pool = rng.integers(0, m, m) if st["resample"] else None
        cells[i] = simulate_pair(P, st["judges"], st["rot"], st["protos"], st["budgets"], st["zs"], rng, st["D"], pool, st["gen_mode"], st["ties"])
        dec[i] = decomp_pair(P, st["judges"], st["budgets"], st["zs"], rng, st["D"], pool, st["ties"])
    return b, cells, dec


def run(pairs, judges, rot, protos, budgets, zs, n_rep, D, seed, n_jobs=1, resample=True, gen_mode="sampled", ties="half", label=""):
    """Returns cells [n_rep, P, slots, protos, budgets, 7] and decomposition [n_rep, P, J, 4, budgets, 7]."""
    st = dict(pairs=pairs, judges=judges, rot=rot, protos=protos, budgets=budgets, zs=zs, D=D, resample=resample,
              gen_mode=gen_mode, ties=ties)
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n_rep)]
    tasks = list(enumerate(seeds))
    C = np.zeros((n_rep, len(pairs), len(judges) + 1, len(protos), len(budgets), 7), dtype=np.int32)
    Dc = np.zeros((n_rep, len(pairs), len(judges), 4, len(budgets), 7), dtype=np.int32)
    t0 = time.time(); step = max(1, n_rep // 10); done = 0
    log.info("simulation %s: %d replicate(s) x %d pairs x %d draws, resample=%s, jobs=%d", label, n_rep, len(pairs), D, resample, n_jobs)

    def take(res):
        nonlocal done
        b, c, d = res; C[b] = c; Dc[b] = d; done += 1
        if done % step == 0 or done == n_rep:
            log.info("  %s %d/%d done (%.0fs)", label, done, n_rep, time.time() - t0)

    if n_jobs > 1 and n_rep > 1:
        with mp.get_context("fork").Pool(n_jobs, initializer=_init, initargs=(st,)) as pool:
            for res in pool.imap_unordered(_one, tasks, chunksize=2):
                take(res)
    else:
        _init(st)
        for t in tasks:
            take(_one(t))
    return C, Dc
