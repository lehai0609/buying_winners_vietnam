import json
import os
from pathlib import Path
import tempfile

import pytest

from bwv import experiment_tracking


def test_start_log_artifact_and_end_run(tmp_path):
    # Minimal config
    config = {"strategy": {"J": 12, "K": 3}, "seed": 42}
    base_dir = str(tmp_path / "results")

    # Start run
    run_id, run_dir = experiment_tracking.start_run(config, tracking_cfg={"base_dir": base_dir}, base_dir=base_dir)
    run_path = Path(run_dir)
    assert run_path.exists()
    assert (run_path / "config.json").exists()
    assert (run_path / "manifest.json").exists()
    assert (run_path / "metrics.csv").exists()
    assert (run_path / "artifacts").exists()

    # Log params and metrics
    params = {"param1": "value1"}
    metrics = {"sharpe": 1.23, "cagr": 0.15}
    experiment_tracking.log_params_metrics(run_id, params=params, metrics=metrics, step=0, base_dir=base_dir)

    # Metrics file should contain entries
    metrics_lines = (run_path / "metrics.csv").read_text(encoding="utf-8").splitlines()
    # header + at least two metrics rows
    assert metrics_lines[0].split(",") == ["timestamp", "step", "key", "value"]
    assert any("sharpe" in line for line in metrics_lines)

    # Create a temp artifact and log it
    tmpfile = tmp_path / "dummy.txt"
    tmpfile.write_text("hello")
    experiment_tracking.log_artifact(run_id, str(tmpfile), artifact_name="dummy.txt", base_dir=base_dir)

    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    assert "artifacts" in manifest
    assert any("dummy.txt" in a for a in manifest["artifacts"])

    # End run
    experiment_tracking.end_run(run_id, status="FINISHED", base_dir=base_dir)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FINISHED"
    assert "ended_at" in manifest
