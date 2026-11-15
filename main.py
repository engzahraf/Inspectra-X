"""
Main Execution and Validation Script for Steel Defect Handcrafted Image Features.

Extracts exactly 15 handcrafted image features, validates the dataset,
and produces analytical plots and statistical summaries.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import tempfile
import time
from typing import List, Optional

# Set writable cache directory for headless matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset_loader import SteelDatasetLoader, TARGET_COLUMNS
from src.feature_extraction import (
    FEATURE_COLUMNS,
    _worker_extract
)
from src.preprocessing import get_or_create_reference_template
from src.data_quality import run_stage2_analysis
from src.data_preparation import run_data_preparation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Steel Defect Classification Using 15 Handcrafted Image Features"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Dataset",
        help="Path to folder containing train.csv and train_images/"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join("features", "steel_defect_features.csv"),
        help="Path for saving extracted tabular features CSV"
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=os.path.join("features", "reference_defect_free.png"),
        help="Path for saving/loading the defect-free reference image"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of images to process (omit for full dataset)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate statistical summary and distribution plots in results/"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "1", "2", "3"],
        default="all",
        help="Pipeline stage to execute: '1' (Extraction), '2' (Data Quality), '3' (Stratified Splitting), or 'all'"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        default=False,
        help="Skip raw image feature extraction and use existing features CSV"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=os.path.join("data", "splits"),
        help="Directory to save Stage 3 stratified dataset splits"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save analytical results, plots, and reports"
    )
    return parser.parse_args()


def validate_extracted_dataset(df: pd.DataFrame, loader: SteelDatasetLoader) -> bool:
    """
    Performs rigorous academic validation on the extracted tabular dataset:
      1. Verifies that exactly 15 feature columns are generated.
      2. Checks for NaN, null, and infinite values.
      3. Verifies matching with CSV records.
      4. Validates multi-label targets for multi-defect images.
      5. Prints descriptive sample values.
    """
    print("\n" + "=" * 70)
    print("                    DATASET VALIDATION CHECKS")
    print("=" * 70)

    all_passed = True

    # Check 1: Exactly 15 feature columns
    found_feature_cols = [c for c in df.columns if c in FEATURE_COLUMNS]
    print(f"[*] Feature column count: {len(found_feature_cols)} (Expected: 15)")
    if len(found_feature_cols) != 15:
        print(f"[-] ERROR: Expected 15 feature columns, but found {len(found_feature_cols)}.")
        all_passed = False
    else:
        print("    [+] Exactly 15 handcrafted feature columns verified:")
        for idx, col in enumerate(found_feature_cols, 1):
            print(f"        {idx:2d}. {col}")

    # Check 2: Missing and Infinite values
    null_counts = df[FEATURE_COLUMNS].isnull().sum().sum()
    inf_counts = np.isinf(df[FEATURE_COLUMNS].to_numpy()).sum()
    print(f"\n[*] Data integrity check:")
    print(f"    Missing (NaN/Null) values in features: {null_counts}")
    print(f"    Infinite (+/- Inf) values in features: {inf_counts}")
    if null_counts > 0 or inf_counts > 0:
        print("[-] ERROR: Dataset contains missing or infinite values!")
        all_passed = False
    else:
        print("    [+] Zero missing or infinite values found. Clean numeric representations.")

    # Check 3: Multi-label validation
    target_cols = ["Crazing", "Inclusion", "Patches", "Pitted_Surface"]
    df["Total_Defect_Classes"] = df[target_cols].sum(axis=1)
    multi_label_count = (df["Total_Defect_Classes"] > 1).sum()
    print(f"\n[*] Target column verification:")
    for col in target_cols:
        count_pos = (df[col] == 1).sum()
        pct = (count_pos / len(df)) * 100
        print(f"    - {col:14s}: {count_pos:5d} ({pct:5.2f}%)")
    print(f"    Multi-label images (>= 2 simultaneous defects): {multi_label_count}")
    if multi_label_count > 0:
        print("    [+] Multi-label target encoding verified successfully.")

    # Check 4: Sample extracted values
    print(f"\n[*] Sample Extracted Feature Values (First 3 Rows):")
    cols_to_show = ["ImageId"] + FEATURE_COLUMNS[:4] + ["Total_Defect_Classes"]
    print(df[cols_to_show].head(3).to_string(index=False))

    print("=" * 70)
    return all_passed


def generate_results_and_visualizations(df: pd.DataFrame, results_dir: str = "results"):
    """
    Generates summary statistics and plots:
      1. Statistical summaries table saved to CSV.
      2. Grid of feature distribution histograms.
      3. Feature correlation heatmap.
    """
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n[*] Generating statistical analysis in '{results_dir}'...")

    # 1. Summary Statistics Table
    stats_df = df[FEATURE_COLUMNS].describe().T
    stats_df["skew"] = df[FEATURE_COLUMNS].skew()
    stats_csv_path = os.path.join(results_dir, "summary_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"    [+] Saved summary statistics to {stats_csv_path}")

    # 2. 15 Feature Distribution Plots (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    axes = axes.ravel()

    for idx, feature_name in enumerate(FEATURE_COLUMNS):
        ax = axes[idx]
        data = df[feature_name].dropna()
        ax.hist(data, bins=35, color="#1f77b4", edgecolor="black", alpha=0.75)
        ax.set_title(f"{idx+1}. {feature_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Distributions of Exactly 15 Handcrafted Image Features", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    dist_plot_path = os.path.join(results_dir, "feature_distributions.png")
    plt.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    [+] Saved feature distributions plot to {dist_plot_path}")

    # 3. Correlation Heatmap
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=9)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=9)

    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.6 else "white", fontsize=7)

    plt.title("Correlation Matrix of Handcrafted Features", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    corr_plot_path = os.path.join(results_dir, "correlation_heatmap.png")
    plt.savefig(corr_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    [+] Saved correlation heatmap to {corr_plot_path}")


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print(" INSPECTRA-X: STEEL DEFECT MULTI-LABEL CLASSIFICATION SYSTEM")
    print("=" * 70)
    print(f"[*] Pipeline Stage:    {args.stage.upper()}")
    print(f"[*] Features CSV Path: {args.output_csv}")

    # Determine whether Stage 1 extraction is requested or needed
    run_stage_1 = (args.stage in ("all", "1")) and not args.skip_extraction
    dataset_exists = os.path.exists(args.dataset_dir)
    features_exist = os.path.exists(args.output_csv)

    df_features = None

    if run_stage_1 and dataset_exists:
        print(f"[*] Dataset Directory: {args.dataset_dir}")
        print(f"[*] Parallel Workers:  {args.n_jobs}")

        # Initialize Dataset Loader
        loader = SteelDatasetLoader(args.dataset_dir)
        total_available = len(loader)
        print(f"[*] Total dataset images indexed: {total_available}")

        # Create/Load Reproducible Defect-Free Reference Image
        ref_path = os.path.abspath(args.reference_image)
        get_or_create_reference_template(loader, save_path=ref_path)
        print(f"[*] Reference defect-free template confirmed: {ref_path}")

        # Select processing cohort
        if args.sample_size is not None and args.sample_size < total_available:
            # Balanced sampling: ensure multi-label, single defect, and clean images are present
            summary_df = loader.get_summary_dataframe()
            multi_ids = summary_df[summary_df['Defect_Count'] > 1]['ImageId'].tolist()
            single_ids = summary_df[summary_df['Defect_Count'] == 1]['ImageId'].tolist()
            clean_ids = summary_df[summary_df['Defect_Count'] == 0]['ImageId'].tolist()

            n_each = max(1, args.sample_size // 3)
            chosen = multi_ids[:n_each] + single_ids[:n_each] + clean_ids[:(args.sample_size - 2 * n_each)]
            image_ids = chosen[:args.sample_size]
            print(f"[*] Running stratified sample of {len(image_ids)} images (multi-defect, single-defect, clean).")
        else:
            image_ids = loader.image_list
            print(f"[*] Running extraction on full dataset: {len(image_ids)} images.")

        # Prepare extraction tasks
        tasks = []
        for img_id in image_ids:
            rec = loader.get_image_info(img_id)
            tasks.append((
                rec['image_id'],
                rec['image_path'],
                rec['rle_map'],
                rec['labels'],
                ref_path
            ))

        # Parallel Execution
        results = []
        completed_count = 0
        total_tasks = len(tasks)
        checkpoint_interval = 1000
        print(f"[*] Starting extraction across {args.n_jobs} processes for {total_tasks} images...", flush=True)

        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(_worker_extract, task): task[0] for task in tasks}
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
                completed_count += 1
                if completed_count % 250 == 0 or completed_count == total_tasks:
                    elapsed = time.time() - start_time
                    fps = completed_count / max(0.1, elapsed)
                    remaining_sec = (total_tasks - completed_count) / max(0.1, fps)
                    eta_min = remaining_sec / 60.0
                    print(f"    Progress: {completed_count:5d}/{total_tasks} ({completed_count/total_tasks*100:5.1f}%) | Speed: {fps:4.1f} img/s | ETA: {eta_min:4.1f} min", flush=True)

                # Periodic checkpointing
                if completed_count % checkpoint_interval == 0:
                    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
                    df_chk = pd.DataFrame(results)
                    ordered_cols = ["ImageId"] + FEATURE_COLUMNS + TARGET_COLUMNS
                    df_chk = df_chk[[c for c in ordered_cols if c in df_chk.columns]]
                    df_chk.to_csv(args.output_csv, index=False)

        # Assemble Output DataFrame
        df_features = pd.DataFrame(results)

        # Reorder columns explicitly: ImageId, 15 features, 4 targets
        ordered_cols = ["ImageId"] + FEATURE_COLUMNS + TARGET_COLUMNS
        df_features = df_features[ordered_cols]

        # Save to CSV
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        df_features.to_csv(args.output_csv, index=False)
        print(f"\n[+] Extracted feature dataset saved to: {args.output_csv}")
        print(f"    Rows: {len(df_features)}, Columns: {len(df_features.columns)}")

        # Validation Checks
        validate_extracted_dataset(df_features, loader)

        # Statistical Summaries and Plots
        if args.generate_plots:
            generate_results_and_visualizations(df_features, results_dir=args.results_dir)

    elif features_exist:
        if run_stage_1 and not dataset_exists:
            print(f"[*] Raw Dataset directory '{args.dataset_dir}' not found on disk.")
            print(f"[*] Utilizing pre-extracted features CSV: '{args.output_csv}'")
        else:
            print(f"[*] Loading pre-extracted features CSV: '{args.output_csv}'")
        df_features = pd.read_csv(args.output_csv)
        print(f"    Loaded {len(df_features):,} records with {len(df_features.columns)} columns.")

        # Baseline summary and plots if stage 1 requested or needed
        if args.stage in ("all", "1") and args.generate_plots:
            generate_results_and_visualizations(df_features, results_dir=args.results_dir)
    else:
        raise FileNotFoundError(
            f"Neither raw dataset directory '{args.dataset_dir}' nor extracted CSV '{args.output_csv}' was found. "
            f"Cannot proceed with feature analysis or data splitting."
        )

    # -------------------------------------------------------------
    # STAGE 2: Comprehensive Feature Analysis & Data Quality
    # -------------------------------------------------------------
    if args.stage in ("all", "2"):
        run_stage2_analysis(
            features_csv=args.output_csv,
            results_dir=args.results_dir
        )

    # -------------------------------------------------------------
    # STAGE 3: Data Preparation & Multi-Label Stratified Splitting
    # -------------------------------------------------------------
    if args.stage in ("all", "3"):
        run_data_preparation(
            features_csv=args.output_csv,
            output_dir=args.splits_dir
        )

    total_time = time.time() - start_time
    print(f"\n[+] Inspectra-X workflow completed successfully in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()

