"""Tests for the batch_healthcheck script (scripts/batch_healthcheck.py)."""

import sys
from pathlib import Path

import polars as pl
import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from batch_healthcheck import find_data_directories, get_file_size_mb

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TestFindDataDirectories:
    """Tests for dataset discovery logic."""

    def _touch(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 100)

    def test_finds_data_at_root(self, tmp_path):
        self._touch(tmp_path / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")
        self._touch(tmp_path / "PR_DATA_DM_ADMMART_PRED.parquet")

        result = find_data_directories(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == tmp_path.name
        assert result[0]["model_file"].name == "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
        assert result[0]["predictor_file"].name == "PR_DATA_DM_ADMMART_PRED.parquet"

    def test_finds_data_in_hc_subdir(self, tmp_path):
        self._touch(tmp_path / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")

        result = find_data_directories(tmp_path)
        assert len(result) == 1
        assert result[0]["data_dir"] == tmp_path / "HC"
        assert result[0]["predictor_file"] is None

    def test_finds_multiple_datasets(self, tmp_path):
        self._touch(tmp_path / "CustomerA" / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")
        self._touch(tmp_path / "CustomerB" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")

        result = find_data_directories(tmp_path)
        names = {d["name"] for d in result}
        assert names == {"CustomerA", "CustomerB"}

    def test_returns_empty_when_no_data(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        assert find_data_directories(tmp_path) == []

    def test_wildcard_pattern_match(self, tmp_path):
        self._touch(tmp_path / "Custom_MDL_FACT.parquet")

        result = find_data_directories(tmp_path)
        assert len(result) == 1
        assert "MDL_FACT" in result[0]["model_file"].name


class TestGetFileSizeMb:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00" * 1024 * 1024)
        assert abs(get_file_size_mb(f) - 1.0) < 0.01

    def test_none_returns_zero(self):
        assert get_file_size_mb(None) == 0.0

    def test_missing_file_returns_zero(self, tmp_path):
        assert get_file_size_mb(tmp_path / "nope.txt") == 0.0


class TestDiscoverAndLoadRealData:
    """Integration test using the repo's sample CSV data in a HC/ layout."""

    @pytest.fixture
    def hc_layout(self, tmp_path):
        """Create a realistic HC/ directory from the repo's sample CSVs."""
        model_csv = DATA_DIR / "pr_data_dm_admmart_mdl_fact.csv"
        pred_csv = DATA_DIR / "pr_data_dm_admmart_pred.csv"
        if not model_csv.exists():
            pytest.skip("Sample CSV data not available")

        hc_dir = tmp_path / "SampleCustomer" / "HC"
        hc_dir.mkdir(parents=True)

        # Convert CSVs to parquet with expected file names
        pl.read_csv(model_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")
        if pred_csv.exists():
            pl.read_csv(pred_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_PRED.parquet")

        return tmp_path

    def test_discover_real_data(self, hc_layout):
        result = find_data_directories(hc_layout)
        assert len(result) == 1
        assert result[0]["name"] == "SampleCustomer"
        assert result[0]["data_dir"] == hc_layout / "SampleCustomer" / "HC"
        assert result[0]["model_file"].exists()
        assert result[0]["model_file"].stat().st_size > 0

    def test_load_datamart_from_discovered_data(self, hc_layout):
        from pdstools import ADMDatamart

        datasets = find_data_directories(hc_layout)
        ds = datasets[0]

        datamart = ADMDatamart.from_ds_export(
            model_filename=str(ds["model_file"]),
            predictor_filename=str(ds["predictor_file"]) if ds["predictor_file"] else None,
        )

        models = datamart.model_data.collect()
        assert len(models) > 0, "Datamart should contain models"
