"""Integration test for the batch_healthcheck script (scripts/batch_healthcheck.py).

Runs the actual script as a subprocess with sample data, verifying it
produces valid HTML healthcheck reports via Quarto rendering in both
CDN and full-embed modes.
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


def _run_batch(hc_layout, output_dir):
    """Run batch_healthcheck.py and return the subprocess result."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(hc_layout),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_batch_healthcheck_cdn(hc_layout, tmp_path):
    """Run batch_healthcheck.py and verify the CDN report is produced."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Verify CDN HTML report was created
    cdn_files = list(output_dir.glob("*_cdn.html"))
    assert len(cdn_files) >= 1, f"No CDN report found, files: {[f.name for f in output_dir.glob('*.html')]}"

    for html_file in cdn_files:
        size_kb = html_file.stat().st_size / 1024
        assert size_kb > 100, f"{html_file.name} is suspiciously small: {size_kb:.1f} KB"

    # Verify summary CSV
    summary = output_dir / "summary.csv"
    assert summary.exists(), "summary.csv was not created"
    df = pl.read_csv(summary)
    assert len(df) == 1
    assert df["CDN_Status"][0] == "Success"
    assert df["CDN_HTML_MB"][0] > 0


@pytest.mark.slow
def test_batch_healthcheck_full_embed(hc_layout, tmp_path):
    """Verify the full-embed report is produced and larger than CDN."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    cdn_files = list(output_dir.glob("*_cdn.html"))
    full_files = list(output_dir.glob("*_full.html"))
    assert len(cdn_files) >= 1, f"No CDN report, files: {[f.name for f in output_dir.glob('*.html')]}"
    assert len(full_files) >= 1, f"No full-embed report, files: {[f.name for f in output_dir.glob('*.html')]}"

    cdn_size = cdn_files[0].stat().st_size
    full_size = full_files[0].stat().st_size
    print(f"CDN report:        {cdn_size / (1024 * 1024):.1f} MB")
    print(f"Full-embed report: {full_size / (1024 * 1024):.1f} MB")
    print(f"Ratio:             {full_size / cdn_size:.1f}x")
    assert full_size > cdn_size, "Full-embed report should be larger than CDN report"

    # Verify summary CSV has both modes
    df = pl.read_csv(output_dir / "summary.csv")
    assert df["FullEmbed_Status"][0] == "Success"
    assert df["FullEmbed_HTML_MB"][0] > 0
