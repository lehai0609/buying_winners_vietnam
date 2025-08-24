"""
Report generation for buying-winners-vietnam.

Primary entry:
  generate_report(results: dict, config: dict, out_dir: str) -> dict

Expectations for `results` (any missing keys are handled gracefully):
- equity_curve: pd.Series (indexed by date) or list-like
- monthly_returns: pd.Series or pd.DataFrame
- benchmark_returns: pd.Series (optional)
- metrics: dict
- stats: dict
- grid_results: pd.DataFrame (optional)
- subperiod_stats: pd.DataFrame (optional)
- weights, trades, turnover, costs (optional)

Outputs are written under out_dir/artifacts and a manifest of created files is returned.
"""

from __future__ import annotations
import os
from pathlib import Path
import json
from typing import Dict, Any
import warnings

import pandas as pd
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Optional templating
try:
    from jinja2 import Template  # type: ignore
    _HAS_JINJA = True
except Exception:
    _HAS_JINJA = False

try:
    import markdown as md  # type: ignore
    _HAS_MARKDOWN = True
except Exception:
    _HAS_MARKDOWN = False


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_series(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        # If single column DF, convert to Series
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        raise ValueError("Cannot coerce DataFrame with >1 column to Series")
    # Attempt to build Series from list-like
    try:
        return pd.Series(obj)
    except Exception:
        return None


def _safe_savefig(fig, path: Path, dpi=150):
    try:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    finally:
        plt.close(fig)


def _plot_equity(equity: pd.Series, benchmark: pd.Series | None, out_path: Path, dpi=150):
    fig, ax = plt.subplots(figsize=(10, 5))
    if equity is not None:
        equity = equity.sort_index()
        ax.plot(equity.index, equity.values, label="Strategy")
    if benchmark is not None:
        benchmark = benchmark.sort_index()
        ax.plot(benchmark.index, benchmark.values, label="Benchmark")
    ax.set_ylabel("Cumulative Return / Equity")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    _safe_savefig(fig, out_path, dpi=dpi)


def _plot_drawdown(equity: pd.Series, out_path: Path, dpi=150):
    if equity is None:
        return
    eq = equity.sort_index()
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(drawdown.index, drawdown.values)
    ax.fill_between(drawdown.index, drawdown.values, 0, where=drawdown.values < 0, color="red", alpha=0.3)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True)
    _safe_savefig(fig, out_path, dpi=dpi)


def _plot_heatmap(grid_df, out_path: Path, dpi=150):
    if grid_df is None or grid_df.empty:
        return
    # grid_df is expected to have multi-index or columns representing J, K or be pivotable
    try:
        pivot = grid_df.copy()
        if isinstance(pivot.index, pd.MultiIndex) or "J" in pivot.columns and "K" in pivot.columns:
            if "value" in pivot.columns:
                heat = pivot.pivot(index="J", columns="K", values="value")
            else:
                # Try first numeric column
                numeric_cols = pivot.select_dtypes("number").columns
                if len(numeric_cols) == 0:
                    return
                heat = pivot.pivot(index="J", columns="K", values=numeric_cols[0])
        else:
            # If already pivoted
            heat = pivot
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(heat, annot=True, fmt=".2f", ax=ax, cmap="vlag", center=0)
        ax.set_title("J x K heatmap")
        _safe_savefig(fig, out_path, dpi=dpi)
    except Exception:
        warnings.warn("Failed to render heatmap; input format not recognized")


def generate_report(results: Dict[str, Any], config: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """
    Generate report artifacts under out_dir/artifacts and return dict of created paths.
    """
    out = Path(out_dir)
    artifacts = out / "artifacts"
    _ensure_dir(artifacts)

    created = {}

    # Save metrics and stats
    metrics = results.get("metrics", {})
    stats = results.get("stats", {})
    grid = results.get("grid_results")
    subperiod = results.get("subperiod_stats")

    (artifacts / "tables").mkdir(exist_ok=True)
    (artifacts / "figures").mkdir(exist_ok=True)

    # Write metrics & stats JSON
    metrics_path = artifacts / "tables" / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    created["metrics_json"] = str(metrics_path.relative_to(out))

    stats_path = artifacts / "tables" / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    created["stats_json"] = str(stats_path.relative_to(out))

    # Save monthly returns and equity
    mr = results.get("monthly_returns")
    try:
        if mr is not None:
            if isinstance(mr, (pd.Series, pd.DataFrame)):
                mr.to_csv(artifacts / "tables" / "monthly_returns.csv")
            else:
                pd.DataFrame(mr).to_csv(artifacts / "tables" / "monthly_returns.csv", index=False)
            created["monthly_returns_csv"] = str((artifacts / "tables" / "monthly_returns.csv").relative_to(out))
    except Exception:
        warnings.warn("Failed to write monthly_returns")

    equity = _to_series(results.get("equity_curve"))
    if equity is not None:
        try:
            eq_df = equity.copy()
            # If index is not string, coerce
            eq_df.to_csv(artifacts / "tables" / "equity_curve.csv")
            created["equity_curve_csv"] = str((artifacts / "tables" / "equity_curve.csv").relative_to(out))
        except Exception:
            warnings.warn("Failed to write equity_curve")

    # grid_results
    if grid is not None:
        try:
            if isinstance(grid, pd.DataFrame):
                grid.to_csv(artifacts / "tables" / "grid_results.csv")
                created["grid_results_csv"] = str((artifacts / "tables" / "grid_results.csv").relative_to(out))
            else:
                pd.DataFrame(grid).to_csv(artifacts / "tables" / "grid_results.csv", index=False)
                created["grid_results_csv"] = str((artifacts / "tables" / "grid_results.csv").relative_to(out))
        except Exception:
            warnings.warn("Could not save grid_results")

    if subperiod is not None:
        try:
            if isinstance(subperiod, pd.DataFrame):
                subperiod.to_csv(artifacts / "tables" / "subperiod_stats.csv")
                created["subperiod_stats_csv"] = str((artifacts / "tables" / "subperiod_stats.csv").relative_to(out))
        except Exception:
            warnings.warn("Could not save subperiod_stats")

    # Plots
    fig_equity = artifacts / "figures" / "equity_vs_benchmark.png"
    try:
        bench = _to_series(results.get("benchmark_returns"))
        _plot_equity(equity, bench, fig_equity)
        created["equity_plot"] = str(fig_equity.relative_to(out))
    except Exception:
        warnings.warn("Failed to create equity vs benchmark plot")

    fig_draw = artifacts / "figures" / "drawdown.png"
    try:
        _plot_drawdown(equity, fig_draw)
        created["drawdown_plot"] = str(fig_draw.relative_to(out))
    except Exception:
        warnings.warn("Failed to create drawdown plot")

    heat_path = artifacts / "figures" / "jk_sharpe_heatmap.png"
    try:
        _plot_heatmap(grid, heat_path)
        if heat_path.exists():
            created["jk_heatmap"] = str(heat_path.relative_to(out))
    except Exception:
        warnings.warn("Failed to create JK heatmap")

    # Simple markdown report
    report_md = artifacts / "report.md"
    title = config.get("html", {}).get("title", "Momentum Research Report")
    md_lines = [f"# {title}", "", "## Summary Metrics", ""]
    if metrics:
        for k, v in metrics.items():
            md_lines.append(f"- **{k}**: {v}")
    else:
        md_lines.append("Metrics: None")

    md_lines.append("")
    md_lines.append("## Artifacts")
    for k, v in created.items():
        md_lines.append(f"- {k}: {v}")

    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    created["report_md"] = str(report_md.relative_to(out))

    # Optionally render HTML if jinja2/markdown available (simple conversion)
    report_html = artifacts / "report.html"
    try:
        if _HAS_JINJA:
            tpl = Template(report_md.read_text(encoding="utf-8"))
            html = tpl.render(metrics=metrics, created=created)
            report_html.write_text(html, encoding="utf-8")
        elif _HAS_MARKDOWN:
            html = md.markdown(report_md.read_text(encoding="utf-8"))
            report_html.write_text(html, encoding="utf-8")
        else:
            # Minimal HTML wrapper
            html_body = "<pre>" + report_md.read_text(encoding="utf-8") + "</pre>"
            report_html.write_text(html_body, encoding="utf-8")
        created["report_html"] = str(report_html.relative_to(out))
    except Exception:
        warnings.warn("Failed to render HTML report")

    # Write an index manifest
    manifest_path = out / "manifest.json"
    manifest = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "artifacts": created,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return created
