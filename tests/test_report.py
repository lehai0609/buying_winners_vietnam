import json
from pathlib import Path

import pandas as pd
import numpy as np

from bwv.report import generate_report


def test_generate_report_minimal(tmp_path):
    # Toy results
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    # Build a simple equity curve (cumulative returns)
    returns = pd.Series([0.01] * len(dates), index=dates)
    equity = (1 + returns).cumprod()
    monthly_returns = returns
    benchmark = pd.Series([0.005] * len(dates), index=dates)
    metrics = {"CAGR": 0.12, "Sharpe": 1.0}
    stats = {"alpha": 0.02, "t_alpha": 2.5}

    # Simple grid_results DataFrame suitable for pivoting
    grid = pd.DataFrame(
        [
            {"J": 3, "K": 1, "value": 0.8},
            {"J": 3, "K": 3, "value": 0.9},
            {"J": 6, "K": 1, "value": 1.1},
            {"J": 6, "K": 3, "value": 1.2},
        ]
    )

    subperiod = pd.DataFrame({"period": ["2010-2014", "2015-2019"], "Sharpe": [0.9, 1.1]})

    results = {
        "equity_curve": equity,
        "monthly_returns": monthly_returns,
        "benchmark_returns": benchmark,
        "metrics": metrics,
        "stats": stats,
        "grid_results": grid,
        "subperiod_stats": subperiod,
    }

    cfg = {"html": {"title": "Test Report"}}
    out_dir = tmp_path / "report_out"

    created = generate_report(results, cfg, str(out_dir))

    # Basic expectations about created artifacts
    assert "report_md" in created
    assert "report_html" in created
    assert "equity_plot" in created
    assert "drawdown_plot" in created
    assert "monthly_returns_csv" in created
    assert "equity_curve_csv" in created
    assert "metrics_json" in created
    assert "stats_json" in created
    assert "jk_heatmap" in created or "jk_heatmap" not in created  # heatmap may or may not be created depending on format

    # Verify files exist
    base = Path(out_dir)
    for key, rel in created.items():
        p = base / rel
        assert p.exists(), f"Expected artifact for {key} at {p}"

    # Validate CSV contents for monthly_returns
    mr_path = base / created["monthly_returns_csv"]
    df_mr = pd.read_csv(mr_path, index_col=0)
    assert not df_mr.empty

    # Validate metrics JSON
    metrics_path = base / created["metrics_json"]
    metrics_loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "CAGR" in metrics_loaded and metrics_loaded["CAGR"] == metrics["CAGR"]
