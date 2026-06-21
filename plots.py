"""Generate the thesis figure set from the collected/ tables.

Reads the outputs of collect.py (all_runs.csv, all_per_epoch.csv, all_per_mut_dist.csv)
and writes a browsable, thesis-ready tree of figures + tables:

    plots/
    ├── epochs/        per-(dataset,model) training curves: train / val_id / T_close / T_far
    │                  for the best ERM baseline and the best run of each method, plus
    │                  ERM-vs-methods overlays and the unsupervised-loss ramp mechanism panel.
    ├── experiments/   per-dataset method-vs-ERM comparisons: lift-over-ERM bars, absolute
    │                  MAE per protocol, the headline extrapolation figure, model comparison.
    ├── distance/      MAE vs mutation distance (ERM vs best method) -- the extrapolation gradient.
    ├── tables/        method x {val_id,E1..E5} value + lift heatmaps, exported to PNG/LaTeX/CSV;
    │                  plus the mechanism table (lift_close/far, G_extrap, D_shift, deltas).
    ├── diagnostics/   trust checks: val-gap confound, seed spread, prediction-collapse, scatter.
    └── splits/        the dataset-design figure: mutation-distance histogram with the bands.

"Best run" = the run selected by val_id (honest selection), NOT the lowest test error.
Every error-bar figure is emitted twice: <name>_std (spread over seeds) and <name>_sem
(standard error). Figures save as both .png (browse) and .pdf (thesis).

Usage:
    python plots.py                                   # collected/ -> plots/
    python plots.py --collected collected --out plots --artifacts /root/run_artifacts
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- conventions ------------------------------------------------------------
PROTO_ORDER = ["E1", "E2", "E3", "E4", "E5"]
PROTO_TESTPOOL = {"E1": "T_close", "E2": "T_close", "E3": "T_far",
                  "E4": "T_far", "E5": "T_far"}
METHOD_ORDER = ["erm", "cmd", "pseudo_labeling", "mean_teacher", "fixmatch", "adabn"]
METHOD_COLORS = {
    "erm": "#6c6c6c", "cmd": "#1f77b4", "pseudo_labeling": "#ff7f0e",
    "mean_teacher": "#2ca02c", "fixmatch": "#d62728", "adabn": "#9467bd",
}
CURVE_SERIES = {  # column -> (label, color, every-epoch?)
    "train_mae":   ("train",   "#999999", True),
    "val_id_mae":  ("val_id",  "#1f77b4", True),
    "T_close_mae": ("T_close", "#ff7f0e", False),
    "T_far_mae":   ("T_far",   "#d62728", False),
}
RAMP_COLS = ["train_lambda_pseudo", "train_lambda_consistency", "train_lambda_fixmatch", "train_loss_cmd"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight", "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10, "axes.titlesize": 11,
})


# ---- io helpers -------------------------------------------------------------
def _load(collected: Path, name: str) -> pd.DataFrame:
    f = collected / name
    if not f.exists():
        print(f"  (missing) {name}")
        return pd.DataFrame()
    return pd.read_csv(f)


def _save(fig, out: Path, sub: str, name: str) -> None:
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> {sub}/{name}.png")


def _methods_present(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].dropna().unique())
    return [m for m in METHOD_ORDER if m in present]


def _err(std: float, n: int, kind: str) -> float:
    if not np.isfinite(std):
        return 0.0
    return std / np.sqrt(max(n, 1)) if kind == "sem" else std


# ---- 1. epochs --------------------------------------------------------------
def _best_run_row(final: pd.DataFrame, dataset: str, model: str, method: str):
    sub = final[(final.dataset == dataset) & (final.model == model) & (final.method == method)]
    sub = sub.dropna(subset=["mae_val_id"]) if "mae_val_id" in sub.columns else sub
    if sub.empty:
        return None
    return sub.loc[sub["mae_val_id"].idxmin()]


def _plot_curve(ax, ep: pd.DataFrame, title: str) -> None:
    for col, (label, color, every) in CURVE_SERIES.items():
        if col not in ep.columns:
            continue
        s = ep[["epoch", col]].dropna()
        if s.empty:
            continue
        ax.plot(s["epoch"], s[col], marker=("." if every else "o"),
                ms=(3 if every else 5), lw=1.4, color=color, label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE / loss"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)


def epochs_figures(final, per_epoch, out: Path) -> None:
    if per_epoch.empty:
        return
    by_run = {rid: g.sort_values("epoch") for rid, g in per_epoch.groupby("run_id")}
    for ds in sorted(final.dataset.dropna().unique()):
        for model in sorted(final[final.dataset == ds].model.unique()):
            methods = _methods_present(final[(final.dataset == ds) & (final.model == model)])
            # individual best-run curves (incl. ERM baseline)
            for method in methods:
                row = _best_run_row(final, ds, model, method)
                if row is None or row["run_id"] not in by_run:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                _plot_curve(ax, by_run[row["run_id"]],
                            f"{ds} / {model} / {method}  (best by val_id: "
                            f"{row.get('protocol','?')}, seed {row.get('seed','?')})")
                _save(fig, out, "epochs", f"{ds}_{model}_{method}")
            # ERM-vs-methods overlay on T_far and val_id
            for series, ylab in (("T_far_mae", "T_far MAE"), ("val_id_mae", "val_id MAE")):
                fig, ax = plt.subplots(figsize=(6.5, 4))
                drew = False
                for method in methods:
                    row = _best_run_row(final, ds, model, method)
                    if row is None or row["run_id"] not in by_run:
                        continue
                    ep = by_run[row["run_id"]]
                    if series not in ep.columns:
                        continue
                    s = ep[["epoch", series]].dropna()
                    if s.empty:
                        continue
                    ax.plot(s["epoch"], s[series], marker=".", lw=1.5,
                            color=METHOD_COLORS.get(method, None), label=method)
                    drew = True
                if drew:
                    ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
                    ax.set_title(f"{ds} / {model}: {ylab} by method (best run each)", fontsize=10)
                    ax.legend(fontsize=8)
                    _save(fig, out, "epochs", f"{ds}_{model}_overlay_{series}")
                else:
                    plt.close(fig)
            # mechanism panel: unsupervised-loss ramp vs T_far, for each adaptation method
            for method in [m for m in methods if m != "erm"]:
                row = _best_run_row(final, ds, model, method)
                if row is None or row["run_id"] not in by_run:
                    continue
                ep = by_run[row["run_id"]]
                ramp = next((c for c in RAMP_COLS if c in ep.columns and ep[c].notna().any()), None)
                if ramp is None or "T_far_mae" not in ep.columns:
                    continue
                fig, ax = plt.subplots(figsize=(6.5, 4))
                s = ep[["epoch", "T_far_mae"]].dropna()
                ax.plot(s["epoch"], s["T_far_mae"], marker="o", color="#d62728", label="T_far MAE")
                ax.set_xlabel("epoch"); ax.set_ylabel("T_far MAE", color="#d62728")
                ax2 = ax.twinx(); ax2.grid(False)
                r = ep[["epoch", ramp]].dropna()
                ax2.plot(r["epoch"], r[ramp], color="#2ca02c", lw=1.4, label=ramp)
                ax2.set_ylabel(ramp, color="#2ca02c")
                ax.set_title(f"{ds} / {model} / {method}: T_far vs {ramp}", fontsize=9)
                _save(fig, out, "epochs", f"{ds}_{model}_{method}_mechanism")


# ---- lift helper (paired on seed) -------------------------------------------
def _lift_table(dm: pd.DataFrame) -> pd.DataFrame:
    """Per (method, protocol): lift over ERM (paired on test_pool+seed)."""
    if "report_metric" not in dm.columns:
        return pd.DataFrame()
    erm = (dm[dm.method == "erm"][["test_pool", "seed", "report_metric"]]
           .rename(columns={"report_metric": "erm"}))
    meth = dm[dm.method != "erm"].merge(erm, on=["test_pool", "seed"], how="left")
    if meth.empty:
        return pd.DataFrame()
    meth["lift"] = meth["erm"] - meth["report_metric"]
    out = (meth.groupby(["method", "protocol"])
                .agg(lift_mean=("lift", "mean"), lift_std=("lift", "std"),
                     n=("seed", "nunique")).reset_index())
    return out


# ---- 2. experiments ---------------------------------------------------------
def experiments_figures(final, out: Path) -> None:
    for ds in sorted(final.dataset.dropna().unique()):
        dd = final[final.dataset == ds]
        models = sorted(dd.model.unique())
        for kind in ("std", "sem"):
            # lift-over-ERM bars, subplot per model
            fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), squeeze=False)
            any_bar = False
            for ax, model in zip(axes[0], models):
                lt = _lift_table(dd[dd.model == model])
                if lt.empty:
                    ax.set_visible(False); continue
                methods = [m for m in METHOD_ORDER if m in set(lt.method)]
                x = np.arange(len(PROTO_ORDER)); w = 0.8 / max(len(methods), 1)
                for i, m in enumerate(methods):
                    vals, errs = [], []
                    for p in PROTO_ORDER:
                        r = lt[(lt.method == m) & (lt.protocol == p)]
                        vals.append(r.lift_mean.iloc[0] if len(r) else np.nan)
                        errs.append(_err(r.lift_std.iloc[0], int(r.n.iloc[0]), kind) if len(r) else 0)
                    ax.bar(x + i * w, vals, w, yerr=errs, capsize=2,
                           color=METHOD_COLORS.get(m), label=m)
                    any_bar = True
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xticks(x + w * (len(methods) - 1) / 2); ax.set_xticklabels(PROTO_ORDER)
                ax.set_title(f"{model}", fontsize=10); ax.set_ylabel("lift over ERM (MAE)")
                ax.legend(fontsize=7)
            if any_bar:
                fig.suptitle(f"{ds}: method lift over ERM (>0 beats baseline, ±{kind})", fontsize=11)
                _save(fig, out, "experiments", f"{ds}_lift_{kind}")
            else:
                plt.close(fig)

        # absolute MAE per protocol per model (std bars) + ERM reference lines
        fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), squeeze=False)
        for ax, model in zip(axes[0], models):
            dm = dd[dd.model == model]
            methods = [m for m in _methods_present(dm) if m != "erm"]
            agg = (dm[dm.method != "erm"].groupby(["method", "protocol"])
                   ["report_metric"].agg(["mean", "std"]).reset_index())
            x = np.arange(len(PROTO_ORDER)); w = 0.8 / max(len(methods), 1)
            for i, m in enumerate(methods):
                vals = [agg[(agg.method == m) & (agg.protocol == p)]["mean"].iloc[0]
                        if len(agg[(agg.method == m) & (agg.protocol == p)]) else np.nan
                        for p in PROTO_ORDER]
                errs = [agg[(agg.method == m) & (agg.protocol == p)]["std"].iloc[0]
                        if len(agg[(agg.method == m) & (agg.protocol == p)]) else 0
                        for p in PROTO_ORDER]
                ax.bar(x + i * w, vals, w, yerr=np.nan_to_num(errs), capsize=2,
                       color=METHOD_COLORS.get(m), label=m)
            erm = dm[dm.method == "erm"]
            for pool, ls in (("T_close", "--"), ("T_far", ":")):
                v = erm[erm.test_pool == pool]["report_metric"].mean()
                if np.isfinite(v):
                    ax.axhline(v, color="k", lw=1, ls=ls, label=f"ERM {pool}")
            ax.set_xticks(x + w * (max(len(methods), 1) - 1) / 2); ax.set_xticklabels(PROTO_ORDER)
            ax.set_title(model, fontsize=10); ax.set_ylabel("report MAE")
            ax.legend(fontsize=7)
        fig.suptitle(f"{ds}: absolute report MAE per protocol", fontsize=11)
        _save(fig, out, "experiments", f"{ds}_abs_mae")

        # headline: best model (lowest ERM far), E3 extrapolation, ERM_far vs methods
        ermfar = (dd[(dd.method == "erm") & (dd.test_pool == "T_far")]
                  .groupby("model")["report_metric"].mean())
        if not ermfar.empty:
            best_model = ermfar.idxmin()
            dm = dd[dd.model == best_model]
            e3 = (dm[dm.protocol == "E3"].groupby("method")["report_metric"]
                  .agg(["mean", "std"]).reset_index())
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ["ERM (far)"] + list(e3.method)
            vals = [ermfar[best_model]] + list(e3["mean"])
            errs = [0] + list(np.nan_to_num(e3["std"]))
            colors = [METHOD_COLORS["erm"]] + [METHOD_COLORS.get(m) for m in e3.method]
            ax.bar(labels, vals, yerr=errs, capsize=3, color=colors)
            ax.set_ylabel("T_far MAE (E3 extrapolation)")
            ax.set_title(f"{ds}: extrapolation — best model ({best_model})", fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            _save(fig, out, "experiments", f"{ds}_headline_extrapolation")

            # model comparison: ERM far MAE per architecture
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(ermfar.index, ermfar.values, color="#6c6c6c")
            ax.set_ylabel("ERM T_far MAE"); ax.set_title(f"{ds}: architecture comparison (ERM far)", fontsize=10)
            _save(fig, out, "experiments", f"{ds}_model_comparison")


# ---- 3. distance ------------------------------------------------------------
def distance_figures(final, per_mut, out: Path) -> None:
    if per_mut.empty:
        return
    for ds in sorted(per_mut.dataset.dropna().unique()):
        for model in sorted(per_mut[per_mut.dataset == ds].model.unique()):
            pm = per_mut[(per_mut.dataset == ds) & (per_mut.model == model) & (per_mut.split == "T_far")]
            if pm.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.2))
            # ERM far reference
            erm = pm[pm.method == "erm"]
            if not erm.empty:
                g = erm.groupby("mut_dist")["mae"].agg(["mean", "std"]).reset_index()
                ax.errorbar(g.mut_dist, g["mean"], yerr=np.nan_to_num(g["std"]),
                            marker="s", color="#6c6c6c", capsize=2, label="ERM (far)")
            # best method per (each method), protocol E3 (extrapolation) if available else any far
            for m in [x for x in _methods_present(pm) if x != "erm"]:
                mm = pm[(pm.method == m) & (pm.protocol == "E3")]
                if mm.empty:
                    mm = pm[pm.method == m]
                g = mm.groupby("mut_dist")["mae"].agg(["mean", "std"]).reset_index()
                if g.empty:
                    continue
                ax.errorbar(g.mut_dist, g["mean"], yerr=np.nan_to_num(g["std"]),
                            marker="o", color=METHOD_COLORS.get(m), capsize=2, label=m)
            ax.set_xlabel("mutation distance"); ax.set_ylabel("MAE")
            ax.set_title(f"{ds} / {model}: far-OOD MAE vs distance (E3 extrapolation)", fontsize=10)
            ax.legend(fontsize=8)
            _save(fig, out, "distance", f"{ds}_{model}_mae_vs_distance")


# ---- 4. tables --------------------------------------------------------------
def _matrix(final_dm: pd.DataFrame) -> pd.DataFrame:
    """rows = methods (+ERM ref), cols = val_id,E1..E5 ; values = mean (±std in a parallel frame)."""
    methods = _methods_present(final_dm)
    rows_mean, rows_std = {}, {}
    for m in methods:
        dm = final_dm[final_dm.method == m]
        mean_row, std_row = {}, {}
        # val_id column
        mean_row["val_id"] = dm["mae_val_id"].mean() if "mae_val_id" in dm else np.nan
        std_row["val_id"] = dm["mae_val_id"].std() if "mae_val_id" in dm else np.nan
        for p in PROTO_ORDER:
            if m == "erm":  # ERM has no E1..E5; fill from its matching-band baseline
                pool = PROTO_TESTPOOL[p]
                v = dm[dm.test_pool == pool]["report_metric"]
            else:
                v = dm[dm.protocol == p]["report_metric"]
            mean_row[p] = v.mean() if len(v) else np.nan
            std_row[p] = v.std() if len(v) else np.nan
        rows_mean[m] = mean_row; rows_std[m] = std_row
    cols = ["val_id"] + PROTO_ORDER
    return (pd.DataFrame(rows_mean).T[cols], pd.DataFrame(rows_std).T[cols])


def tables_figures(final, out: Path) -> None:
    tdir = out / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    for ds in sorted(final.dataset.dropna().unique()):
        for model in sorted(final[final.dataset == ds].model.unique()):
            dm = final[(final.dataset == ds) & (final.model == model)]
            mean_df, std_df = _matrix(dm)
            if mean_df.empty:
                continue
            tag = f"{ds}_{model}"
            # CSV + LaTeX (mean ± std)
            disp = mean_df.round(3).astype(str) + " ± " + std_df.round(3).astype(str)
            disp.to_csv(tdir / f"{tag}_table.csv")
            try:
                (tdir / f"{tag}_table.tex").write_text(
                    disp.to_latex(caption=f"{ds} / {model}: MAE per protocol (mean ± std over seeds)",
                                  label=f"tab:{tag}"), encoding="utf-8")
            except Exception:
                pass
            # value heatmap
            _heatmap(mean_df, f"{ds} / {model}: report MAE", out, f"{tag}_heatmap_value",
                     cmap="viridis_r", fmt="{:.3f}")
            # lift heatmap (ERM-band reference minus method), green = beats ERM
            ref = mean_df.loc["erm"] if "erm" in mean_df.index else None
            if ref is not None:
                lift = mean_df.drop(index="erm").rsub(ref)  # ref - method
                if not lift.empty:
                    _heatmap(lift, f"{ds} / {model}: lift over ERM (>0 better)", out,
                             f"{tag}_heatmap_lift", cmap="RdYlGn", fmt="{:+.3f}", center0=True)
            # mechanism table
            _mechanism_table(dm, tdir, tag)


def _heatmap(df: pd.DataFrame, title: str, out: Path, name: str, cmap, fmt, center0=False) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * df.shape[1] + 2, 0.6 * df.shape[0] + 2))
    arr = df.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(arr)) if center0 else None
    im = ax.imshow(arr, cmap=cmap, aspect="auto",
                   vmin=(-vmax if center0 else None), vmax=(vmax if center0 else None))
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, fmt.format(arr[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out, "tables", name)


def _mechanism_table(dm: pd.DataFrame, tdir: Path, tag: str) -> None:
    def cell(method, proto):
        if method == "erm":
            return dm[(dm.method == "erm") & (dm.test_pool == PROTO_TESTPOOL[proto])]["report_metric"].mean()
        return dm[(dm.method == method) & (dm.protocol == proto)]["report_metric"].mean()
    rows = []
    for m in [x for x in _methods_present(dm) if x != "erm"]:
        rows.append({
            "method": m,
            "lift_close": cell("erm", "E2") - cell(m, "E2"),
            "lift_far":   cell("erm", "E4") - cell(m, "E4"),
            "G_extrap":   cell(m, "E3") - cell(m, "E4"),
            "D_shift":    cell(m, "E3") - cell(m, "E2"),
            "delta_close": cell(m, "E2") - cell(m, "E1"),
            "delta_far":   cell(m, "E4") - cell(m, "E5"),
        })
    if rows:
        pd.DataFrame(rows).round(4).to_csv(tdir / f"{tag}_mechanism.csv", index=False)


# ---- 5. diagnostics ---------------------------------------------------------
def diagnostics_figures(final, out: Path, artifacts: Path | None) -> None:
    for ds in sorted(final.dataset.dropna().unique()):
        for model in sorted(final[final.dataset == ds].model.unique()):
            dm = final[(final.dataset == ds) & (final.model == model)]
            tag = f"{ds}_{model}"
            # val-gap confound: method val_id - ERM val_id
            if "mae_val_id" in dm.columns:
                erm_v = dm[dm.method == "erm"]["mae_val_id"].mean()
                gaps = (dm[dm.method != "erm"].groupby("method")["mae_val_id"].mean() - erm_v)
                if len(gaps):
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.bar(gaps.index, gaps.values, color=[METHOD_COLORS.get(m) for m in gaps.index])
                    ax.axhline(0, color="k", lw=0.8)
                    ax.set_ylabel("val_id(method) - val_id(ERM)")
                    ax.set_title(f"{tag}: encoder/val gap (≈0 = same source fit)", fontsize=9)
                    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
                    _save(fig, out, "diagnostics", f"{tag}_val_gap")
            # collapse: report MAE / naive_mae on T_far (→1 = predicts the mean)
            if "naive_mae_T_far" in dm.columns:
                far = dm[dm.test_pool == "T_far"].copy()
                far["ratio"] = far["mae_T_far"] / far["naive_mae_T_far"]
                g = far.groupby("method")["ratio"].mean()
                if len(g):
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.bar(g.index, g.values, color=[METHOD_COLORS.get(m) for m in g.index])
                    ax.axhline(1.0, color="r", lw=1, ls="--", label="mean-predictor")
                    ax.set_ylabel("MAE / naive_MAE (T_far)")
                    ax.set_title(f"{tag}: collapse check (→1 = predicts mean)", fontsize=9)
                    ax.legend(fontsize=8); plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
                    _save(fig, out, "diagnostics", f"{tag}_collapse")
            # seed spread of report_metric per protocol
            if {"protocol", "report_metric", "seed"}.issubset(dm.columns):
                fig, ax = plt.subplots(figsize=(6, 3.5))
                for m in [x for x in _methods_present(dm) if x != "erm"]:
                    sp = dm[dm.method == m].groupby("protocol")["report_metric"].std()
                    sp = sp.reindex(PROTO_ORDER)
                    ax.plot(PROTO_ORDER, sp.values, marker="o", color=METHOD_COLORS.get(m), label=m)
                ax.set_ylabel("std of report MAE over seeds")
                ax.set_title(f"{tag}: seed spread (are 2 seeds enough?)", fontsize=9)
                ax.legend(fontsize=8)
                _save(fig, out, "diagnostics", f"{tag}_seed_spread")
            # prediction scatter (needs artifacts dir)
            if artifacts is not None:
                _scatter_best(dm, out, tag, artifacts)


def _scatter_best(dm: pd.DataFrame, out: Path, tag: str, artifacts: Path) -> None:
    row = None
    cand = dm[dm.method != "erm"].dropna(subset=["mae_val_id"]) if "mae_val_id" in dm else dm
    if not cand.empty:
        row = cand.loc[cand["mae_val_id"].idxmin()]
    if row is None or "run_dir" not in row:
        return
    pf = Path(str(row["run_dir"])) / "evaluation" / "predictions.csv"
    if not pf.exists():
        return
    pred = pd.read_csv(pf)
    splits = [s for s in ["val_id", "T_close", "T_far"] if s in set(pred.split)]
    if not splits:
        return
    fig, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 4), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        d = pred[pred.split == sp]
        ax.scatter(d.y_true, d.y_pred, s=5, alpha=0.3)
        lo, hi = d.y_true.min(), d.y_true.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{sp} (ρ={d.y_true.corr(d.y_pred):.2f})", fontsize=9)
        ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
    fig.suptitle(f"{tag}: {row['method']} predictions", fontsize=10)
    _save(fig, out, "diagnostics", f"{tag}_scatter")


# ---- 6. splits --------------------------------------------------------------
def splits_figures(final, out: Path) -> None:
    try:
        from bioseq_ood.data.load_data import _dataset_splits_dir
    except Exception:
        return
    band_color = {"train": "#6c6c6c", "val_id": "#1f77b4", "val_ood": "#9edae5",
                  "target_close": "#ff7f0e", "target_test": "#d62728", "test": "#aa3377"}
    for ds in sorted(final.dataset.dropna().unique()):
        d = _dataset_splits_dir(ds)
        frames = []
        for s, c in band_color.items():
            f = d / f"{s}.csv"
            if f.exists():
                df = pd.read_csv(f)
                if "mut_dist" in df.columns:
                    frames.append((s, c, df["mut_dist"].to_numpy()))
        if not frames:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        mx = int(max(arr.max() for _, _, arr in frames))
        bins = np.arange(0.5, mx + 1.5)
        for s, c, arr in frames:
            ax.hist(arr, bins=bins, alpha=0.55, color=c, label=f"{s} (n={len(arr)})")
        ax.set_xlabel("mutation distance"); ax.set_ylabel("count")
        ax.set_title(f"{ds}: split design by mutation distance", fontsize=10)
        ax.legend(fontsize=7)
        _save(fig, out, "splits", f"{ds}_mut_dist_bands")


# ---- driver -----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collected", default="collected", help="Dir with the collect.py CSVs.")
    ap.add_argument("--out", default="plots", help="Output dir for the figure tree.")
    ap.add_argument("--artifacts", default=None, help="Run-artifacts dir (enables prediction scatters).")
    args = ap.parse_args()

    collected = Path(args.collected); out = Path(args.out)
    runs = _load(collected, "all_runs.csv")
    per_epoch = _load(collected, "all_per_epoch.csv")
    per_mut = _load(collected, "all_per_mut_dist.csv")
    if runs.empty:
        print("No all_runs.csv -- nothing to plot."); return

    final = runs[runs.stage == "final"].copy() if "stage" in runs.columns else runs.copy()
    if "status" in final.columns:
        final = final[final["status"].isna() | (final["status"] == "ok")]
    artifacts = Path(args.artifacts) if args.artifacts else None

    steps = [
        ("epochs", lambda: epochs_figures(final, per_epoch, out)),
        ("experiments", lambda: experiments_figures(final, out)),
        ("distance", lambda: distance_figures(final, per_mut, out)),
        ("tables", lambda: tables_figures(final, out)),
        ("diagnostics", lambda: diagnostics_figures(final, out, artifacts)),
        ("splits", lambda: splits_figures(final, out)),
    ]
    for name, fn in steps:
        print(f"[{name}]")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
        except Exception as exc:  # one bad group must not kill the rest
            print(f"  [WARN] {name} failed: {exc!r}")
    print(f"\nDone -> {out}/")


if __name__ == "__main__":
    main()
