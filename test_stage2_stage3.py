"""
Comprehensive Test Script for Stage 2 and Stage 3 Deliverables.
Validates all academic and software engineering requirements for Inspectra-X.
"""

import os
import sys
import numpy as np
import pandas as pd

from src.dataset_loader import TARGET_COLUMNS
from src.feature_extraction import FEATURE_COLUMNS
from src.data_quality import verify_dataset_structure, analyze_feature_quality, analyze_target_labels
from src.data_preparation import (
    ID_COLUMN,
    separate_features_and_targets,
    validate_splits,
    load_split_data
)


def run_all_tests():
    print("=" * 80)
    print("        INSPECTRA-X: STAGE 2 & STAGE 3 COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 80)

    features_csv = os.path.join("features", "steel_defect_features.csv")
    splits_dir = os.path.join("data", "splits")
    results_dir = "results"

    # -------------------------------------------------------------
    # TEST GROUP 1: Extracted Features CSV Integrity
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 1: Extracted Features CSV Integrity ---")
    assert os.path.exists(features_csv), f"Missing features CSV: {features_csv}"
    df = pd.read_csv(features_csv)

    assert len(df) == 12568, f"Expected 12,568 rows, found {len(df)}"
    assert len(df.columns) == 20, f"Expected 20 columns, found {len(df.columns)}"
    expected_cols = [ID_COLUMN] + FEATURE_COLUMNS + TARGET_COLUMNS
    assert list(df.columns) == expected_cols, f"Column order or names mismatch: {list(df.columns)}"
    print("[PASS] Row and column counts match expected specifications (12,568 x 20).")

    struct_results = verify_dataset_structure(df)
    assert struct_results["passed"], f"Structural validation failed: {struct_results}"
    print("[PASS] verify_dataset_structure: 0 duplicates, 0 NaNs, 0 Infs, binary targets verified.")

    # -------------------------------------------------------------
    # TEST GROUP 2: Stage 2 Deliverables & Findings
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 2: Stage 2 Analytical Outputs ---")
    expected_stage2_files = [
        os.path.join(results_dir, "summary_statistics.csv"),
        os.path.join(results_dir, "feature_distributions.png"),
        os.path.join(results_dir, "correlation_heatmap.png"),
        os.path.join(results_dir, "target_class_distribution.png"),
        os.path.join(results_dir, "multilabel_distribution.png"),
        os.path.join(results_dir, "feature_outliers_boxplot.png"),
        os.path.join(results_dir, "label_cooccurrence_heatmap.png"),
        os.path.join(results_dir, "data_quality_report.csv"),
        os.path.join(results_dir, "data_quality_report.md"),
        os.path.join(results_dir, "label_cooccurrence_matrix.csv"),
        os.path.join(results_dir, "label_distribution_summary.csv")
    ]

    for fpath in expected_stage2_files:
        assert os.path.exists(fpath), f"Missing Stage 2 output file: {fpath}"
        assert os.path.getsize(fpath) > 0, f"Empty Stage 2 output file: {fpath}"
        print(f"[PASS] File verified: {fpath} ({os.path.getsize(fpath):,} bytes)")

    feat_results = analyze_feature_quality(df, corr_threshold=0.70)
    assert len(feat_results["zero_var_features"]) == 0, "Zero-variance features should be 0"
    assert len(feat_results["high_corr_pairs"]) == 12, f"Expected 12 high corr pairs, got {len(feat_results['high_corr_pairs'])}"
    print("[PASS] Feature quality analysis: 0 zero-variance features, 12 collinear pairs identified.")

    target_results = analyze_target_labels(df)
    assert target_results["clean_images_count"] == 5902, f"Expected 5,902 clean images, got {target_results['clean_images_count']}"
    assert target_results["single_defect_count"] == 6239, f"Expected 6,239 single defect images, got {target_results['single_defect_count']}"
    assert target_results["multi_defect_count"] == 427, f"Expected 427 multi-label images, got {target_results['multi_defect_count']}"
    print("[PASS] Target label distribution verified (5,902 clean, 6,239 single-label, 427 multi-label).")

    # -------------------------------------------------------------
    # TEST GROUP 3: Stage 3 Partitioning & Leakage Prevention
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 3: Stage 3 Partitioning & Leakage Prevention ---")
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "validation.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    summary_csv = os.path.join(splits_dir, "split_summary.csv")

    for p in (train_csv, val_csv, test_csv, summary_csv):
        assert os.path.exists(p), f"Missing split file: {p}"
        assert os.path.getsize(p) > 0, f"Empty split file: {p}"
        print(f"[PASS] Split file verified: {p} ({os.path.getsize(p):,} bytes)")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    summary_df = pd.read_csv(summary_csv)

    assert len(train_df) == 8797, f"Expected 8,797 train rows, got {len(train_df)}"
    assert len(val_df) == 1885, f"Expected 1,885 val rows, got {len(val_df)}"
    assert len(test_df) == 1886, f"Expected 1,886 test rows, got {len(test_df)}"
    assert len(train_df) + len(val_df) + len(test_df) == 12568, "Conservation of records violated!"
    print(f"[PASS] Subset sizes verified: Train={len(train_df)} (70.0%), Val={len(val_df)} (15.0%), Test={len(test_df)} (15.01%). Total=12,568.")

    val_results = validate_splits(df, train_df, val_df, test_df, random_state=42)
    assert val_results["passed"], f"Split validation failed: {val_results}"
    print("[PASS] validate_splits: Zero ImageId overlap, record conservation, column retention, and 100% reproducibility passed.")

    # -------------------------------------------------------------
    # TEST GROUP 4: Reusable Model Loader API
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 4: Reusable Model Loader API ---")
    (X_tr, Y_tr, id_tr), (X_va, Y_va, id_va), (X_te, Y_te, id_te) = load_split_data(splits_dir)

    assert X_tr.shape == (8797, 15), f"Wrong X_tr shape: {X_tr.shape}"
    assert Y_tr.shape == (8797, 4), f"Wrong Y_tr shape: {Y_tr.shape}"
    assert len(id_tr) == 8797, f"Wrong id_tr length: {len(id_tr)}"
    assert ID_COLUMN not in X_tr.columns, f"Leakage: {ID_COLUMN} in X_tr!"
    assert ID_COLUMN not in X_va.columns, f"Leakage: {ID_COLUMN} in X_va!"
    assert ID_COLUMN not in X_te.columns, f"Leakage: {ID_COLUMN} in X_te!"
    assert list(X_tr.columns) == FEATURE_COLUMNS, "Feature columns mismatch"
    assert list(Y_tr.columns) == TARGET_COLUMNS, "Target columns mismatch"
    print("[PASS] load_split_data correctly delivers isolated X (15 features), Y (4 targets), and ID Series.")

    print("\n" + "=" * 80)
    print("           ALL TEST GROUPS PASSED WITH ZERO ERRORS (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
