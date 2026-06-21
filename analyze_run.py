# %% [markdown]
# Run analysis: sanity checks + insights.
# Run this in a Jupyter cell with:  %run analyze_run.py
# or open it in JupyterLab and run section by section. Plots show inline and are
# also saved to figs/. Point RESULTS at a merged or per-shard results_phases folder.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results_phases")
FIGS = Path("figs"); FIGS.mkdir(exist_ok=True)


def _load(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


f  = _load("phaseF_all_protocols.csv")     # one row per (model, method, protocol, seed)
fm = _load("phaseF_by_mut_dist.csv")        # per (run, mutation distance)
g  = _load("phaseG_analysis.csv")           # seed-aggregated headline quantities
gm = _load("phaseG_by_mut_dist.csv")        # per-mutation-distance, seed-aggregated
assert f is not None and g is not None, "Run this from the repo root; results_phases/ must exist."
print("models:", sorted(f["model"].unique()), "| methods:", sorted(f["method"].unique()))

# %% [markdown]
# ## 1. Sanity — did everything train?

# %%
n_err = f["error"].notna().sum()
print(f"phaseF rows: {len(f)}   errored: {n_err}   non-finite report_mae: {(~np.isfinite(f['report_mae'])).sum()}")
if n_err:
    print(f.loc[f["error"].notna(), ["model", "method", "protocol", "seed", "error"]].to_string())
print("\nruns per (model, method)  [expect 5 protocols x n_seeds]:")
print(f.groupby(["model", "method"]).size().unstack(fill_value=0))

# %% Heatmap of MAE per (model, method): a cell far brighter than its row = barely trained.
for proto, label in [("E2", "close test (E2)"), ("E4", "far test (E4)")]:
    piv = f[f.protocol == proto].groupby(["model", "method"])["report_mae"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(piv) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
    for (i, j), v in np.ndenumerate(piv.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(f"report MAE — {label}\n(a much-brighter cell = that method/model barely trained)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(FIGS / f"sanity_heatmap_{proto}.png", dpi=120); plt.show()

# %% val_id MAE per method: a run that failed to train shows a large/wild val_id MAE.
order = sorted(f["method"].unique())
fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot([f[f.method == m]["val_id_mae"].dropna().values for m in order], labels=order)
ax.set_ylabel("val_id MAE"); ax.set_title("Selection-metric (val_id) per method — high/spread = unstable training")
ax.tick_params(axis="x", rotation=40); fig.tight_layout(); fig.savefig(FIGS / "sanity_valid.png", dpi=120); plt.show()

# %% [markdown]
# ## 2. Do the methods beat the ERM baseline?  (lift = ERM - method, >0 is better)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, ttl in [(axes[0], "lift_close", "close-band (E2 vs ERM)"),
                     (axes[1], "lift_far", "far-band (E4 vs ERM)")]:
    s = g.groupby("method")[col].agg(["mean", "std"]).sort_values("mean")
    ax.barh(s.index, s["mean"], xerr=s["std"].fillna(0),
            color=["tab:green" if v > 0 else "tab:red" for v in s["mean"]], capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"lift on {ttl}\n(>0 = method beats ERM)"); ax.set_xlabel("MAE improvement over ERM")
fig.tight_layout(); fig.savefig(FIGS / "lift_vs_erm.png", dpi=120); plt.show()

# %% [markdown]
# ## 3. MAE vs mutation distance (the core OOD-degradation plot)
# Bands are the across-seed std. ERM is not in this file (it's the baseline), so these
# are method curves; compare their *level* and *slope* with distance.

# %%
if gm is not None:
    protos = [p for p in ["E2", "E3", "E4", "E5"] if p in gm["protocol"].unique()]
    for model in sorted(gm["model"].unique()):
        fig, axes = plt.subplots(1, len(protos), figsize=(4.6 * len(protos), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, proto in zip(axes, protos):
            for method, d in gm[(gm.model == model) & (gm.protocol == proto)].groupby("method"):
                d = d.sort_values("mut_dist"); sd = d.report_mae_std.fillna(0)
                ax.plot(d.mut_dist, d.report_mae_mean, marker="o", label=method)
                ax.fill_between(d.mut_dist, d.report_mae_mean - sd, d.report_mae_mean + sd, alpha=0.12)
            ax.set_title(f"{model} — {proto}"); ax.set_xlabel("mutation distance"); ax.set_ylabel("MAE")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"MAE vs mutation distance — {model}"); fig.tight_layout()
        fig.savefig(FIGS / f"mae_vs_mutdist_{model}.png", dpi=120); plt.show()

# %% [markdown]
# ## 4. Across the five protocols (E1/E2 close, E3 extrapolation, E4/E5 far)

# %%
proto_cols = [p for p in ["E1", "E2", "E3", "E4", "E5"] if p in g.columns]
piv = g.groupby("method")[proto_cols].mean()
fig, ax = plt.subplots(figsize=(8, 4))
for method, row in piv.iterrows():
    ax.plot(proto_cols, row.values, marker="o", label=method)
ax.set_ylabel("report MAE"); ax.set_xlabel("protocol")
ax.set_title("MAE across protocols (mean over models)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "protocols.png", dpi=120); plt.show()

# %% Derived gap quantities per method (mean over models)
cols = [c for c in ["lift_close", "lift_far", "G_extrap", "D_shift", "delta_close", "delta_far"] if c in g.columns]
print(g.groupby("method")[cols].mean().round(3).to_string())
