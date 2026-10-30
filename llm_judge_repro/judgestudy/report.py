"""Figures (grayscale-safe: distinct markers and line styles) and the auto-generated run summary."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {"P0": ("#7f7f7f", "x", ":"), "P0r": ("#d62728", "o", "-"), "P1": ("#1f77b4", "s", "--"), "P2": ("#2ca02c", "^", "-."),
         "P3": ("#9467bd", "D", "-"), "P1x": ("#ff7f0e", "v", ":")}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def figures(ctx, source, ST, DT, aux, rc, core, judges, budget):
    panels = core + (["rot"] if len(core) >= 2 else [])
    # 1. flip rate vs budget, close pairs
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), sharey=True, squeeze=False)
    for ax, slot in zip(axs[0], panels):
        d = ST[(ST.scope == "close") & (ST.slot == slot)]
        for pr, g in d.groupby("protocol"):
            c, m, ls = STYLE[pr]; g = g.sort_values("budget")
            ax.plot(g.budget, g.flip_rate, color=c, marker=m, ls=ls, label=pr)
            ax.fill_between(g.budget, g.flip_lo, g.flip_hi, color=c, alpha=0.12)
        ax.set_xscale("log", base=2); ax.set_title(slot); ax.set_xlabel("judge calls per pair (n)")
    axs[0][0].set_ylabel("decisive flip rate (close pairs)")
    hs = {}
    for ax in axs[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l, h)
    fig.legend(list(hs.values()), list(hs.keys()), loc="lower center", ncol=len(hs), frameon=False, fontsize=8)
    fig.suptitle(f"Decision reproducibility vs budget ({source})"); fig.tight_layout(rect=(0, 0.06, 1, 1)); ctx.figure(fig, f"fig_flip_vs_budget_{source}")
    # 2. pair-level flip rate vs human margin (P0r)
    cells = aux["cells"]; d = cells[(cells.protocol == "P0r") & (cells.budget == budget) & (cells.slot.isin(core)) & (cells.obs_both >= 5)]
    if len(d):
        fig, axs = plt.subplots(1, len(core), figsize=(4.2 * len(core), 3.4), sharey=True, squeeze=False)
        for ax, j in zip(axs[0], core):
            g = d[d.slot == j]
            ax.scatter(g.margin, g.obs_flip, s=18, c=np.where(g.set == "close", "#1f77b4", "#7f7f7f"), marker="o", alpha=0.8)
            if len(g) > 4:
                bins = pd.qcut(g.margin, min(5, len(g) // 3), duplicates="drop"); m = g.groupby(bins, observed=True)[["margin", "obs_flip"]].mean()
                ax.plot(m.margin, m.obs_flip, color="k", marker="s", ms=4, lw=1)
            ax.set_title(j); ax.set_xlabel("human margin |win rate - 0.5|")
        axs[0][0].set_ylabel(f"flip rate, P0r, n={budget}"); fig.tight_layout(); ctx.figure(fig, f"fig_flip_vs_margin_{source}")
    # 3. risk-coverage
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), sharey=True, squeeze=False)
    for ax, slot in zip(axs[0], panels):
        for pr, g in rc[rc.slot == slot].groupby("protocol"):
            c, m, ls = STYLE[pr]; g = g.sort_values("coverage")
            ax.plot(g.coverage, g.flip_rate, color=c, marker=m, ls=ls, ms=4, label=pr)
        ax.set_title(slot); ax.set_xlabel("coverage (share decisive)")
    axs[0][0].set_ylabel("decisive flip rate")
    hs = {}
    for ax in axs[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l, h)
    fig.legend(list(hs.values()), list(hs.keys()), loc="lower center", ncol=len(hs), frameon=False, fontsize=8)
    fig.suptitle(f"Risk-coverage, close pairs, n={budget}"); fig.tight_layout(rect=(0, 0.06, 1, 1)); ctx.figure(fig, f"fig_risk_coverage_{source}")
    # 4. predicted vs observed
    ok = cells[(cells.obs_both >= 20) & cells.pred_flip.notna()]
    if len(ok):
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        for pr, g in ok.groupby("protocol"):
            c, m, _ = STYLE[pr]; ax.scatter(g.pred_flip, g.obs_flip, s=10, c=c, marker=m, alpha=0.6, label=pr)
        mx = max(ok.pred_flip.max(), ok.obs_flip.max()) * 1.05; ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        ax.set_xlabel("predicted flip rate (ordered probit, dev-fitted)"); ax.set_ylabel("observed flip rate (test)"); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); ctx.figure(fig, f"fig_pred_vs_obs_{source}")
    # 5. decomposition
    dd = DT[(DT.budget == budget) & (DT.scope.isin(["close", "far"]))]
    if len(dd):
        kinds = ["order", "sampling", "battle", "all"]; fig, axs = plt.subplots(1, len(core), figsize=(4.0 * len(core), 3.3), sharey=True, squeeze=False)
        for ax, j in zip(axs[0], core):
            for k_, (scope, hatch) in enumerate([("close", ""), ("far", "//")]):
                g = dd[(dd.judge == j) & (dd.scope == scope)].set_index("source_of_variation").reindex(kinds)
                ax.bar(np.arange(len(kinds)) + 0.38 * k_, g.flip_rate, 0.36, yerr=[g.flip_rate - g.flip_lo, g.flip_hi - g.flip_rate], hatch=hatch,
                       color="#1f77b4" if scope == "close" else "#bbbbbb", edgecolor="k", label=scope, capsize=2)
            ax.set_xticks(np.arange(len(kinds)) + 0.19); ax.set_xticklabels(kinds); ax.set_title(j)
        axs[0][0].set_ylabel(f"flip rate at n={budget}"); axs[0][-1].legend(frameon=False); fig.tight_layout(); ctx.figure(fig, f"fig_decomposition_{source}")


def summary(ctx):
    """Collect everything the run produced into summary.md (paste-ready for documentation)."""
    d = ctx.dir; lines = [f"# Run summary: {ctx.run_id}", ""]
    man = json.loads((d / "manifest.json").read_text())
    lines += ["## Environment", f"- git commit: {man['git']['commit']} (dirty={man['git']['dirty']})", f"- GPU: {man['env']['gpu']}",
              f"- python: {man['env']['python'].splitlines()[0]}", f"- seed: {man.get('seed')}", f"- config hashes: {json.dumps(man.get('file_sha256', {}))}", ""]
    st = [json.loads(x) for x in (d / "stages.jsonl").read_text().splitlines()] if (d / "stages.jsonl").exists() else []
    if st:
        lines += ["## Stages", "| stage | status | seconds |", "|---|---|---|"] + [f"| {s['stage']} | {s['status']} | {s['seconds']} |" for s in st] + [""]
    proc = Path(ctx.cfg["paths"]["processed"])
    for f in sorted(proc.glob("*_funnel.csv")):
        lines += [f"## Eligibility funnel: {f.stem}", pd.read_csv(f).to_markdown(index=False), ""]
    res = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()] if (d / "results.jsonl").exists() else []
    hy = [r for r in res if r["name"].split("/")[0] in ("H1a", "H1b", "H2", "H3", "H3_RE")]
    if hy:
        lines += ["## Hypothesis verdicts", "| hypothesis | source | verdict |", "|---|---|---|"]
        for r in hy:
            n, s = r["name"].split("/"); lines.append(f"| {n} | {s} | {r.get('verdict')} |")
        lines.append("")
    for t in ["generation_quality", "efficiency", "item_level_metrics", "H1a", "H1b", "H2", "H2_secondary_holm", "H3", "H3_RE", "anchor_replication", "human_clear", "ablations"]:
        for f in sorted((d / "tables").glob(f"{t}*.md")):
            if f.stem.startswith(t) and (f.stem == t or f.stem[len(t):].startswith("_")):
                lines += [f"### {f.stem}", f.read_text(), ""]
    lines += ["## Figures"] + [f"- figures/{p.name}" for p in sorted((d / "figures").glob("*.png"))] + [""]
    gd = d / "gate_decisions.md"
    if gd.exists():
        lines += ["## Gate decisions", gd.read_text()]
    (d / "summary.md").write_text("\n".join(lines))
    idx = [dict(kind="table", path=str(p.relative_to(d))) for p in sorted((d / "tables").glob("*.csv"))] + \
          [dict(kind="figure", path=str(p.relative_to(d))) for p in sorted((d / "figures").glob("*.png"))]
    pd.DataFrame(idx).to_csv(d / "RESULTS_INDEX.csv", index=False)
    ctx.log.info("summary written: %s", d / "summary.md")
