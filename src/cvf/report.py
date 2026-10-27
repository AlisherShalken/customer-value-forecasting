"""Model report: backtest tables and diagnostic figures (reports/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cvf.backtest import summarise  # noqa: E402
from cvf.config import Settings  # noqa: E402
from cvf.evaluation import calibration_table, capture_at  # noqa: E402
from cvf.models.gbm import GBMModel  # noqa: E402
from cvf.policy import break_even_revenue, profit_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#a3a29c"
SERIES = {"lightgbm": "#2a78d6", "baseline": "#eb6834", "baseline_2": "#1baf7a"}

LABELS = {
    "lightgbm": "LightGBM",
    "rfm_logit": "RFM logistic regression",
    "run_rate": "Run-rate (revenue 365d / 4)",
    "last_quarter": "Last quarter revenue",
}

HEADLINE = {
    "propensity": ["roc_auc", "pr_auc", "log_loss", "brier", "ece", "lift@10%"],
    "revenue": ["norm_gini", "spearman", "capture@10%", "capture@20%", "mae", "rmse",
                "total_ratio"],
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
    })


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def plot_calibration(holdout: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle="--", label="Perfect")
    for col, key, label in [("p_purchase", "lightgbm", "LightGBM, calibrated"),
                            ("p_purchase_raw", "baseline", "LightGBM, raw")]:
        t = calibration_table(holdout["purchased"].to_numpy(), holdout[col].to_numpy())
        ax.plot(t["predicted"], t["observed"], marker="o", markersize=5, color=SERIES[key],
                label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability (decile mean)",
           ylabel="Observed purchase rate", title="Calibration on holdout")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_capture(holdout: pd.DataFrame, path: Path) -> str:
    y = holdout["revenue"].to_numpy()
    shares = np.linspace(0.01, 1, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(shares, [capture_at(y, y, s) for s in shares], color=REFERENCE, linewidth=1,
            linestyle="--", label="Perfect ordering")
    for col, key, label in [("expected_revenue", "lightgbm", LABELS["lightgbm"]),
                            ("revenue_run_rate", "baseline", LABELS["run_rate"]),
                            ("revenue_last_quarter", "baseline_2", LABELS["last_quarter"])]:
        ax.plot(shares, [capture_at(y, holdout[col].to_numpy(), s) for s in shares],
                color=SERIES[key], label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Share of customers contacted (ranked by score)",
           ylabel="Share of next-90-day revenue captured",
           title="Revenue capture on holdout")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_profit(holdout: pd.DataFrame, settings: Settings, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    realised = holdout["revenue"].to_numpy()
    for col, key, label in [("expected_revenue", "lightgbm", LABELS["lightgbm"]),
                            ("revenue_run_rate", "baseline", LABELS["run_rate"])]:
        curve = profit_curve(holdout[col].to_numpy(), realised, settings.policy, n_points=200)
        ax.plot(curve["share"], curve["realised_profit"], color=SERIES[key],
                label=f"Ranked by {label}")
    chosen = (holdout["expected_revenue"] > break_even_revenue(settings.policy)).mean()
    ax.axvline(chosen, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate(f"policy: {chosen:.0%} contacted", xy=(chosen, ax.get_ylim()[0]),
                xytext=(6, 8), textcoords="offset points", color=INK_2, fontsize=9)
    ax.axhline(0, color=REFERENCE, linewidth=1)
    ax.set(xlim=(0, 1), xlabel="Share of customers contacted",
           ylabel="Realised campaign profit, GBP", title="Campaign profit on holdout")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_importance(model: GBMModel, holdout_features: pd.DataFrame, path: Path,
                    top: int = 15) -> str:
    contrib = model.contributions(holdout_features).drop(columns="bias")
    imp = contrib.abs().mean().sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(imp.index, imp.to_numpy(), color=SERIES["lightgbm"], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP value| (log-odds)",
           title="Purchase propensity: top drivers")
    return _save(fig, path)


def plot_stability(metrics: pd.DataFrame, path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    panels = [("propensity", "roc_auc", "ROC AUC (propensity)", ["lightgbm", "rfm_logit"]),
              ("revenue", "norm_gini", "Normalised Gini (revenue)", ["lightgbm", "run_rate"])]
    for ax, (task, metric, title, models) in zip(axes, panels, strict=True):
        for model, key in zip(models, ["lightgbm", "baseline"], strict=True):
            s = metrics[(metrics["task"] == task) & (metrics["metric"] == metric)
                        & (metrics["model"] == model)]
            ax.plot(pd.to_datetime(s["cutoff"]), s["value"], marker="o", markersize=6,
                    color=SERIES[key], label=LABELS[model])
        ax.set_title(title)
        ax.set_xticks(pd.to_datetime(sorted(metrics["cutoff"].unique())))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        ax.legend(loc="lower left")
    fig.suptitle("Backtest by cutoff", x=0.01, ha="left", fontsize=12, color=INK)
    return _save(fig, path)


def _metrics_table(metrics: pd.DataFrame, task: str) -> str:
    s = summarise(metrics).query("task == @task and metric in @HEADLINE[@task]")
    s = s.assign(cell=s["mean"].map("{:.3f}".format) + " ± " + s["std"].map("{:.3f}".format))
    table = s.pivot(index="metric", columns="model", values="cell")
    table = table.reindex(HEADLINE[task])[["lightgbm"] + [c for c in table if c != "lightgbm"]]
    table.columns = [LABELS[c] for c in table.columns]
    return table.to_markdown()


def write_report(metrics: pd.DataFrame, predictions: pd.DataFrame,
                 holdout_models: dict[str, GBMModel], holdout_features: pd.DataFrame,
                 settings: Settings, extra: dict | None = None) -> Path:
    _style()
    out = settings.reports_dir
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "backtest_metrics.csv", index=False)

    cutoff = settings.evaluation.holdout_cutoff
    holdout = predictions[predictions["snapshot_date"] == cutoff]
    figs = {
        "calibration": plot_calibration(holdout, fig_dir / "calibration.png"),
        "capture": plot_capture(holdout, fig_dir / "revenue_capture.png"),
        "profit": plot_profit(holdout, settings, fig_dir / "campaign_profit.png"),
        "importance": plot_importance(holdout_models["propensity"], holdout_features,
                                      fig_dir / "feature_importance.png"),
        "stability": plot_stability(metrics, fig_dir / "backtest_stability.png"),
    }

    policy = settings.policy
    threshold = break_even_revenue(policy)
    contact = holdout["expected_revenue"] > threshold
    rate = policy.assumed_uplift * policy.gross_margin
    profit_policy = (rate * holdout.loc[contact, "revenue"].sum()
                     - policy.contact_cost * contact.sum())
    profit_all = rate * holdout["revenue"].sum() - policy.contact_cost * len(holdout)

    lines = [
        "# Model report",
        "",
        f"Backtest cutoffs: {', '.join(str(c) for c in settings.evaluation.backtest_cutoffs)}. "
        f"Holdout cutoff: {cutoff} ({len(holdout):,} active customers, "
        f"purchase rate {holdout['purchased'].mean():.1%}, "
        f"revenue £{holdout['revenue'].sum():,.0f}).",
        "",
        "Values are mean ± std across backtest cutoffs; each cutoff retrains the full "
        "production procedure on data available at that date.",
        "",
        "## Purchase propensity (90 days)",
        "",
        _metrics_table(metrics, "propensity"),
        "",
        "## Revenue (90 days)",
        "",
        _metrics_table(metrics, "revenue"),
        "",
        "## Campaign policy on holdout",
        "",
        f"Contact when expected 90-day revenue > £{threshold:,.2f} "
        f"(contact cost £{policy.contact_cost}, margin {policy.gross_margin:.0%}, "
        f"assumed uplift {policy.assumed_uplift:.0%}).",
        "",
        "| | Contacted | Revenue captured | Realised profit |",
        "|---|---|---|---|",
        f"| Model policy | {contact.mean():.1%} | "
        f"{holdout.loc[contact, 'revenue'].sum() / holdout['revenue'].sum():.1%} | "
        f"£{profit_policy:,.0f} |",
        f"| Contact everyone | 100.0% | 100.0% | £{profit_all:,.0f} |",
        "",
        "## Figures",
        "",
        *[f"![{name}](figures/{file})" for name, file in figs.items()],
        "",
    ]
    if extra:
        lines += ["## Run metadata", "", "```json", pd.Series(extra).to_json(indent=2), "```", ""]
    path = out / "model_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
