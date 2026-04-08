"""Integration test for the batch_healthcheck script (scripts/batch_healthcheck.py).

Runs the actual script as a subprocess with sample data, verifying it
produces valid HTML healthcheck reports via Quarto rendering.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "batch_healthcheck.py"


@pytest.fixture
def hc_layout(tmp_path):
    """Create a realistic HC/ directory from the repo's sample CSVs."""
    model_csv = DATA_DIR / "pr_data_dm_admmart_mdl_fact.csv"
    pred_csv = DATA_DIR / "pr_data_dm_admmart_pred.csv"
    if not model_csv.exists():
        pytest.skip("Sample CSV data not available")

    hc_dir = tmp_path / "SampleCustomer" / "HC"
    hc_dir.mkdir(parents=True)

    pl.read_csv(model_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")
    if pred_csv.exists():
        pl.read_csv(pred_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_PRED.parquet")

    return tmp_path


@pytest.mark.slow
def test_batch_healthcheck_produces_html(hc_layout, tmp_path):
    """Run batch_healthcheck.py and verify it produces an HTML report."""
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(hc_layout),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Verify HTML report was created
    html_files = list(output_dir.glob("*.html"))
    assert len(html_files) >= 1, f"Expected at least 1 HTML file in {output_dir}, found {len(html_files)}"

    # Verify the HTML file is non-trivial
    for html_file in html_files:
        size_kb = html_file.stat().st_size / 1024
        assert size_kb > 100, f"HTML file {html_file.name} is suspiciously small: {size_kb:.1f} KB"

    # Verify summary CSV was created
    summary = output_dir / "summary.csv"
    assert summary.exists(), "summary.csv was not created"
    df = pl.read_csv(summary)
    assert len(df) == 1
    assert df["Status"][0] == "Success"
