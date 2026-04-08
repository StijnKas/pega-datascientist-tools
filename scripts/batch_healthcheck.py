#!/usr/bin/env python
"""Batch HealthCheck Report Generator.

Generate ADM HealthCheck reports for multiple datasets.

This script discovers ADM model and predictor data files, generates HealthCheck
reports, and creates a summary of results with error detection.

Usage Examples
--------------
Process all datasets in a directory:
    python batch_healthcheck.py /path/to/data

Process a single dataset:
    python batch_healthcheck.py /path/to/data/CustomerA

Specify output directory:
    python batch_healthcheck.py /path/to/data --output ./reports

Process specific datasets by name:
    python batch_healthcheck.py /path/to/data --datasets CustomerA CustomerB

Directory Structure
-------------------
The script automatically discovers data in these patterns:
- /path/to/data/Dataset1/HC/*.parquet
- /path/to/data/Dataset2/HC/*.parquet
- /path/to/data/HC/*.parquet (if single dataset)
- /path/to/data/*.parquet (if files at root)

Required files:
- Model file: PR_DATA_DM_ADMMART_MDL_FACT.parquet (or *MDL_FACT.parquet)
- Predictor file: PR_DATA_DM_ADMMART_PRED.parquet (optional, or *PRED.parquet)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import polars as pl
from pdstools import ADMDatamart
from pdstools.utils.report_utils import check_report_for_errors


# Default file name patterns
MODEL_FILE_PATTERNS = ["PR_DATA_DM_ADMMART_MDL_FACT.parquet", "*MDL_FACT.parquet"]
PREDICTOR_FILE_PATTERNS = ["PR_DATA_DM_ADMMART_PRED.parquet", "*PRED.parquet"]


def find_data_directories(root_path: Path) -> list[dict]:
    """Discover directories containing ADM data files.

    Parameters
    ----------
    root_path : Path
        Root directory to search for data

    Returns
    -------
    list[dict]
        List of dictionaries with keys: name, data_dir, model_file, predictor_file
    """
    datasets = []

    # Check if root_path itself contains data files
    for pattern in MODEL_FILE_PATTERNS:
        model_files = list(root_path.glob(pattern))
        if model_files:
            # Found data at root level
            model_file = model_files[0]
            predictor_file = None
            for pred_pattern in PREDICTOR_FILE_PATTERNS:
                pred_files = list(root_path.glob(pred_pattern))
                if pred_files:
                    predictor_file = pred_files[0]
                    break

            datasets.append(
                {
                    "name": root_path.name,
                    "data_dir": root_path,
                    "model_file": model_file,
                    "predictor_file": predictor_file,
                }
            )
            return datasets  # If we found data at root, don't search subdirs

    # Check for HC subdirectory at root level
    hc_dir = root_path / "HC"
    if hc_dir.exists() and hc_dir.is_dir():
        for pattern in MODEL_FILE_PATTERNS:
            model_files = list(hc_dir.glob(pattern))
            if model_files:
                model_file = model_files[0]
                predictor_file = None
                for pred_pattern in PREDICTOR_FILE_PATTERNS:
                    pred_files = list(hc_dir.glob(pred_pattern))
                    if pred_files:
                        predictor_file = pred_files[0]
                        break

                datasets.append(
                    {
                        "name": root_path.name,
                        "data_dir": hc_dir,
                        "model_file": model_file,
                        "predictor_file": predictor_file,
                    }
                )
                return datasets

    # Search subdirectories for HC folders or direct data
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Check subdir/HC pattern
        hc_dir = subdir / "HC"
        if hc_dir.exists() and hc_dir.is_dir():
            for pattern in MODEL_FILE_PATTERNS:
                model_files = list(hc_dir.glob(pattern))
                if model_files:
                    model_file = model_files[0]
                    predictor_file = None
                    for pred_pattern in PREDICTOR_FILE_PATTERNS:
                        pred_files = list(hc_dir.glob(pred_pattern))
                        if pred_files:
                            predictor_file = pred_files[0]
                            break

                    datasets.append(
                        {
                            "name": subdir.name,
                            "data_dir": hc_dir,
                            "model_file": model_file,
                            "predictor_file": predictor_file,
                        }
                    )
                    break

        # Check subdir directly for data files
        if not any(d["name"] == subdir.name for d in datasets):
            for pattern in MODEL_FILE_PATTERNS:
                model_files = list(subdir.glob(pattern))
                if model_files:
                    model_file = model_files[0]
                    predictor_file = None
                    for pred_pattern in PREDICTOR_FILE_PATTERNS:
                        pred_files = list(subdir.glob(pred_pattern))
                        if pred_files:
                            predictor_file = pred_files[0]
                            break

                    datasets.append(
                        {
                            "name": subdir.name,
                            "data_dir": subdir,
                            "model_file": model_file,
                            "predictor_file": predictor_file,
                        }
                    )
                    break

    return datasets


def get_file_size_mb(file_path: Path | None) -> float:
    """Get file size in MB."""
    if file_path and file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def _generate_report(
    datamart: ADMDatamart,
    name: str,
    output_dir: Path,
    *,
    full_embed: bool,
) -> tuple[float, str | None, str | None]:
    """Generate a single HealthCheck report and return (size_mb, status, errors)."""
    label = "full-embed" if full_embed else "CDN"
    suffix = "_full" if full_embed else "_cdn"
    print(f"  → Generating report ({label})...")

    try:
        output_path = datamart.generate.health_check(
            name=name.lower().replace(" ", "_").replace(".", "_") + suffix,
            title=f"ADM Health Check - {name}",
            subtitle=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            output_dir=str(output_dir),
            full_embed=full_embed,
        )

        html_path = Path(output_path)
        size_mb = get_file_size_mb(html_path)
        print(f"  ✓ Report ({label}): {size_mb:.1f} MB")

        # Scan HTML for errors
        html_errors = check_report_for_errors(html_path)
        if html_errors:
            errors_str = "; ".join(html_errors)
            print(f"  ⚠ HTML errors ({label}):")
            for error in html_errors:
                print(f"    - {error}")
            return size_mb, "Success (with errors)", errors_str

        print(f"  ✓ No errors in HTML ({label})")
        return size_mb, "Success", None

    except Exception as e:
        print(f"  ✗ Error ({label}): {e}")
        import traceback

        traceback.print_exc()
        return 0.0, "Error", str(e)


def process_dataset(
    dataset: dict,
    output_dir: Path,
) -> dict:
    """Process a single dataset and generate HealthCheck reports.

    Generates two reports per dataset: one with CDN mode (smaller, no esbuild
    needed) and one with full embed (larger, standalone). Both sizes are
    reported for comparison.

    Parameters
    ----------
    dataset : dict
        Dataset information (name, data_dir, model_file, predictor_file)
    output_dir : Path
        Directory for output reports

    Returns
    -------
    dict
        Processing results with status and metrics
    """
    name = dataset["name"]
    print(f"\n{'=' * 60}")
    print(f"Processing: {name}")
    print(f"{'=' * 60}")
    print(f"  Data directory: {dataset['data_dir']}")

    result = {
        "Dataset": name,
        "Model_File_MB": 0.0,
        "Predictor_File_MB": 0.0,
        "CDN_HTML_MB": 0.0,
        "CDN_Status": "Not Found",
        "CDN_Errors": None,
        "FullEmbed_HTML_MB": 0.0,
        "FullEmbed_Status": "Not Found",
        "FullEmbed_Errors": None,
    }

    model_file = dataset["model_file"]
    predictor_file = dataset["predictor_file"]

    # Get input file sizes
    result["Model_File_MB"] = get_file_size_mb(model_file)
    result["Predictor_File_MB"] = get_file_size_mb(predictor_file)

    print(f"  ✓ Model file: {result['Model_File_MB']:.1f} MB")
    if predictor_file:
        print(f"  ✓ Predictor file: {result['Predictor_File_MB']:.1f} MB")
    else:
        print("  ℹ No predictor file found")

    try:
        # Create ADMDatamart
        print("  → Loading datamart...")
        datamart = ADMDatamart.from_ds_export(
            model_filename=str(model_file),
            predictor_filename=str(predictor_file) if predictor_file else None,
        )

        print(f"  ✓ Datamart loaded: {len(datamart.model_data.collect())} models")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate both CDN and full-embed variants
        cdn_mb, cdn_status, cdn_errors = _generate_report(
            datamart,
            name,
            output_dir,
            full_embed=False,
        )
        result["CDN_HTML_MB"] = cdn_mb
        result["CDN_Status"] = cdn_status
        result["CDN_Errors"] = cdn_errors

        embed_mb, embed_status, embed_errors = _generate_report(
            datamart,
            name,
            output_dir,
            full_embed=True,
        )
        result["FullEmbed_HTML_MB"] = embed_mb
        result["FullEmbed_Status"] = embed_status
        result["FullEmbed_Errors"] = embed_errors

        # Print size comparison
        if cdn_mb > 0 and embed_mb > 0:
            ratio = embed_mb / cdn_mb
            print(f"  ℹ Size comparison: CDN {cdn_mb:.1f} MB vs full-embed {embed_mb:.1f} MB ({ratio:.1f}x)")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        result["CDN_Status"] = "Error"
        result["CDN_Errors"] = str(e)
        result["FullEmbed_Status"] = "Error"
        result["FullEmbed_Errors"] = str(e)
        import traceback

        traceback.print_exc()

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch generate ADM HealthCheck reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/customers
  %(prog)s /path/to/customers --output ./reports
  %(prog)s /path/to/customers --datasets CustomerA CustomerB
  %(prog)s /path/to/single_customer/HC

For more information, see:
  https://github.com/pegasystems/pega-datascientist-tools
        """,
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to directory containing datasets (with HC folders) or a single dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./healthcheck_reports"),
        help="Output directory for generated reports (default: ./healthcheck_reports)",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        help="Specific dataset names to process (default: process all found)",
    )

    args = parser.parse_args()

    # Validate input path
    if not args.data_path.exists():
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)

    if not args.data_path.is_dir():
        print(f"Error: Data path is not a directory: {args.data_path}")
        sys.exit(1)

    # Discover datasets
    print(f"\n{'=' * 60}")
    print("Discovering datasets...")
    print(f"{'=' * 60}")
    print(f"Searching in: {args.data_path.absolute()}")

    all_datasets = find_data_directories(args.data_path)

    if not all_datasets:
        print("\nNo datasets found!")
        print("\nExpected file patterns:")
        print(f"  Model: {', '.join(MODEL_FILE_PATTERNS)}")
        print(f"  Predictor: {', '.join(PREDICTOR_FILE_PATTERNS)}")
        print("\nExpected directory structures:")
        print("  - /path/to/data/Dataset1/HC/*.parquet")
        print("  - /path/to/data/Dataset1/*.parquet")
        print("  - /path/to/data/HC/*.parquet")
        print("  - /path/to/data/*.parquet")
        sys.exit(1)

    # Filter datasets if specific ones requested
    if args.datasets:
        requested = set(args.datasets)
        datasets_to_process = [d for d in all_datasets if d["name"] in requested]

        if not datasets_to_process:
            print("\nError: None of the requested datasets found")
            print(f"Requested: {', '.join(args.datasets)}")
            print(f"Available: {', '.join(d['name'] for d in all_datasets)}")
            sys.exit(1)

        # Warn about datasets not found
        found_names = {d["name"] for d in datasets_to_process}
        for name in requested - found_names:
            print(f"Warning: Dataset '{name}' not found, skipping")
    else:
        datasets_to_process = all_datasets

    # Display summary
    print(f"\nFound {len(all_datasets)} dataset(s):")
    for ds in all_datasets:
        marker = "→" if ds in datasets_to_process else " "
        print(f"  {marker} {ds['name']}")

    print(f"\n{'=' * 60}")
    print("Batch HealthCheck Report Generator")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output.absolute()}")
    print(f"Datasets to process: {len(datasets_to_process)}")

    # Process all datasets
    results = []
    summary_file = args.output / "summary.csv"
    for i, dataset in enumerate(datasets_to_process, 1):
        print(f"\n[{i}/{len(datasets_to_process)}]")
        result = process_dataset(dataset, args.output)
        results.append(result)

        # Update summary CSV after each dataset
        df_incremental = pl.DataFrame(results)
        df_incremental.write_csv(summary_file)
        print(f"  ✓ Summary updated: {summary_file}")

    # Create summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    df = pl.DataFrame(results)

    # Format the table for display
    summary_table = df.select(
        [
            pl.col("Dataset"),
            pl.col("Model_File_MB").round(1).alias("Model (MB)"),
            pl.col("Predictor_File_MB").round(1).alias("Pred (MB)"),
            pl.col("CDN_HTML_MB").round(1).alias("CDN (MB)"),
            pl.col("CDN_Status").alias("CDN Status"),
            pl.col("FullEmbed_HTML_MB").round(1).alias("Embed (MB)"),
            pl.col("FullEmbed_Status").alias("Embed Status"),
        ]
    )

    print(summary_table)

    # Show HTML errors if any
    for mode, col in [("CDN", "CDN_Errors"), ("Full-embed", "FullEmbed_Errors")]:
        errors_df = df.filter(pl.col(col).is_not_null())
        if len(errors_df) > 0:
            print(f"\n{'=' * 60}")
            print(f"HTML Errors Detected ({mode})")
            print(f"{'=' * 60}")
            for row in errors_df.iter_rows(named=True):
                print(f"\n{row['Dataset']}:")
                for error in row[col].split("; "):
                    print(f"  - {error}")

    # Summary CSV already saved incrementally during processing
    print(f"\n✓ Final summary: {summary_file}")

    # Print statistics
    cdn_success = (df["CDN_Status"] == "Success").sum()
    cdn_errors = (df["CDN_Status"] == "Success (with errors)").sum()
    cdn_failed = len(df) - cdn_success - cdn_errors
    embed_success = (df["FullEmbed_Status"] == "Success").sum()
    embed_errors = (df["FullEmbed_Status"] == "Success (with errors)").sum()
    embed_failed = len(df) - embed_success - embed_errors

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  CDN mode:        {cdn_success} success, {cdn_errors} with errors, {cdn_failed} failed")
    print(f"  Full-embed mode: {embed_success} success, {embed_errors} with errors, {embed_failed} failed")
    print(
        f"Total input size:  {df['Model_File_MB'].sum():.1f} MB models, {df['Predictor_File_MB'].sum():.1f} MB predictors"
    )
    print(f"Total CDN output:  {df['CDN_HTML_MB'].sum():.1f} MB")
    print(f"Total embed output: {df['FullEmbed_HTML_MB'].sum():.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
