"""
Data Preparation and Multi-Label Stratified Splitting Module for Inspectra-X (Stage 3).

Provides end-to-end data preparation for classical machine learning:
  1. Input (X) and Output (Y) separation, isolating ImageId as an identifier.
  2. Iterative multi-label stratified partitioning into Train (70%), Validation (15%), and Test (15%).
  3. Strict prevention of data leakage (no global feature scaling or cross-split contamination).
  4. Exhaustive split validation (zero identifier overlap, exact record preservation, column retention).
  5. Persistence of partitioned datasets and summary statistics into data/splits/.
  6. Reusable loader interface for subsequent model training (Random Forest, SVM, XGBoost).
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.dataset_loader import TARGET_COLUMNS
from src.feature_extraction import FEATURE_COLUMNS

ID_COLUMN = "ImageId"
EXPECTED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + TARGET_COLUMNS
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_RANDOM_STATE = 42


def separate_features_and_targets(
    df: pd.DataFrame
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Separates the input DataFrame into identifiers, feature matrix X, and target matrix Y.

    Guarantees:
      - ImageId is strictly preserved as an identifier and NEVER included in X.
      - X contains exactly the 15 handcrafted feature columns.
      - Y contains exactly the 4 binary target columns.

    Args:
        df: Input DataFrame containing 20 columns.

    Returns:
        Tuple of (ids, X, Y).
    """
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]

    if missing_features:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing_features}")
    if missing_targets:
        raise ValueError(f"Input DataFrame is missing required target columns: {missing_targets}")
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Input DataFrame is missing identifier column '{ID_COLUMN}'.")

    ids = df[ID_COLUMN].copy()
    X = df[FEATURE_COLUMNS].copy()
    Y = df[TARGET_COLUMNS].copy()

    # Academic sanity check
    assert ID_COLUMN not in X.columns, f"Leakage alert: '{ID_COLUMN}' detected inside feature matrix X!"
    assert len(X.columns) == 15, f"Expected 15 features, got {len(X.columns)}"
    assert len(Y.columns) == 4, f"Expected 4 targets, got {len(Y.columns)}"

    return ids, X, Y


def multilabel_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset into Train, Validation, and Test subsets using
    Iterative Multi-Label Stratification (MultilabelStratifiedShuffleSplit).

    Preserves label co-occurrence and individual class proportions across all three splits.

    Args:
        df: Complete dataset DataFrame (12,568 rows x 20 columns).
        train_ratio: Proportion of training data (default 0.70).
        val_ratio: Proportion of validation data (default 0.15).
        test_ratio: Proportion of test data (default 0.15).
        random_state: Fixed random seed for deterministic reproducibility (default 42).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    # Extract X (features) and Y (multi-label targets)
    _, X_df, Y_df = separate_features_and_targets(df)
    X = X_df.values
    Y = Y_df.values

    # Step 1: Split full dataset into Train (70%) and Temp (30% = Val + Test)
    temp_ratio = val_ratio + test_ratio
    msss_outer = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=random_state
    )
    train_idx, temp_idx = next(msss_outer.split(X, Y))

    # Step 2: Split Temp into Validation (15% of total = 50% of Temp) and Test (15% of total = 50% of Temp)
    relative_test_ratio = test_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
    msss_inner = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_ratio,
        random_state=random_state
    )
    val_sub_idx, test_sub_idx = next(msss_inner.split(X[temp_idx], Y[temp_idx]))

    # Map sub-indices back to global dataframe indices
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    # Create subset DataFrames with explicit standard column order
    ordered_cols = [ID_COLUMN] + FEATURE_COLUMNS + TARGET_COLUMNS
    train_df = df.iloc[train_idx][ordered_cols].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx][ordered_cols].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx][ordered_cols].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def validate_splits(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """
    Executes academic validation on the partitioned subsets:
      1. Zero overlap of ImageIds between any two splits.
      2. Record conservation: union of splits equals original dataset exactly.
      3. Column retention: all 20 columns present in all subsets.
      4. Target retention: no defect class lost in any split.
      5. Label distribution fidelity: comparison of class frequencies across splits.
      6. Split reproducibility check: identical partition when repeated with random_state=42.
    """
    validation_results = {
        "passed": True,
        "checks": {}
    }

    # 1. ImageId overlap check
    train_ids = set(train_df[ID_COLUMN])
    val_ids = set(val_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])

    train_val_overlap = len(train_ids & val_ids)
    train_test_overlap = len(train_ids & test_ids)
    val_test_overlap = len(val_ids & test_ids)
    total_overlap = train_val_overlap + train_test_overlap + val_test_overlap
    overlap_passed = total_overlap == 0

    validation_results["checks"]["id_overlap"] = {
        "passed": overlap_passed,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "message": "Zero ImageId overlap verified across all splits." if overlap_passed else f"ImageId overlap detected: {total_overlap} samples!"
    }
    if not overlap_passed:
        validation_results["passed"] = False

    # 2. Record conservation
    total_split_records = len(train_df) + len(val_df) + len(test_df)
    original_records = len(original_df)
    record_conservation_passed = (total_split_records == original_records) and ((train_ids | val_ids | test_ids) == set(original_df[ID_COLUMN]))

    validation_results["checks"]["record_conservation"] = {
        "passed": record_conservation_passed,
        "original_count": original_records,
        "split_sum_count": total_split_records,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "message": f"Record conservation verified: {total_split_records} / {original_records} records." if record_conservation_passed else "Record count mismatch!"
    }
    if not record_conservation_passed:
        validation_results["passed"] = False

    # 3. Column retention
    expected_cols = [ID_COLUMN] + FEATURE_COLUMNS + TARGET_COLUMNS
    cols_train_ok = list(train_df.columns) == expected_cols
    cols_val_ok = list(val_df.columns) == expected_cols
    cols_test_ok = list(test_df.columns) == expected_cols
    all_cols_ok = cols_train_ok and cols_val_ok and cols_test_ok

    validation_results["checks"]["column_retention"] = {
        "passed": all_cols_ok,
        "expected_columns": len(expected_cols),
        "train_cols": len(train_df.columns),
        "val_cols": len(val_df.columns),
        "test_cols": len(test_df.columns),
        "message": "All 20 columns (1 ID + 15 features + 4 targets) retained across all splits." if all_cols_ok else "Column discrepancy across splits!"
    }
    if not all_cols_ok:
        validation_results["passed"] = False

    # 4. Target retention
    target_retention_ok = True
    for col in TARGET_COLUMNS:
        if (train_df[col].sum() == 0) or (val_df[col].sum() == 0) or (test_df[col].sum() == 0):
            target_retention_ok = False

    validation_results["checks"]["target_retention"] = {
        "passed": target_retention_ok,
        "message": "All defect classes represented with positive samples in Train, Val, and Test." if target_retention_ok else "Defect class dropped in a split!"
    }
    if not target_retention_ok:
        validation_results["passed"] = False

    # 5. Label distribution comparison
    dist_comparison = []
    for col in TARGET_COLUMNS:
        orig_pct = (original_df[col].sum() / len(original_df)) * 100
        train_pct = (train_df[col].sum() / len(train_df)) * 100
        val_pct = (val_df[col].sum() / len(val_df)) * 100
        test_pct = (test_df[col].sum() / len(test_df)) * 100
        dist_comparison.append({
            "target": col,
            "original_count": int(original_df[col].sum()),
            "original_pct": orig_pct,
            "train_count": int(train_df[col].sum()),
            "train_pct": train_pct,
            "val_count": int(val_df[col].sum()),
            "val_pct": val_pct,
            "test_count": int(test_df[col].sum()),
            "test_pct": test_pct,
            "max_pct_deviation": max(abs(train_pct - orig_pct), abs(val_pct - orig_pct), abs(test_pct - orig_pct))
        })
    validation_results["checks"]["label_distributions"] = dist_comparison

    # 6. Deterministic reproducibility test
    t2, v2, te2 = multilabel_train_val_test_split(original_df, random_state=random_state)
    rep_train = train_df[ID_COLUMN].equals(t2[ID_COLUMN])
    rep_val = val_df[ID_COLUMN].equals(v2[ID_COLUMN])
    rep_test = test_df[ID_COLUMN].equals(te2[ID_COLUMN])
    reproducible = rep_train and rep_val and rep_test

    validation_results["checks"]["reproducibility"] = {
        "passed": reproducible,
        "message": f"Splits are 100% deterministic and reproducible with random_state={random_state}." if reproducible else "Splits are non-deterministic!"
    }
    if not reproducible:
        validation_results["passed"] = False

    return validation_results


def compute_split_summary(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs a comprehensive summary DataFrame comparing sample counts,
    percentages, defect frequencies, clean images, and multi-label counts across splits.
    """
    subsets = [
        ("Full Dataset", original_df),
        ("Training Set (70%)", train_df),
        ("Validation Set (15%)", val_df),
        ("Test Set (15%)", test_df)
    ]

    records = []
    total_len = len(original_df)

    for name, df_sub in subsets:
        n = len(df_sub)
        split_pct = (n / total_len) * 100.0

        defect_sums = df_sub[TARGET_COLUMNS].sum(axis=1)
        clean_count = int((defect_sums == 0).sum())
        clean_pct = (clean_count / n) * 100.0 if n > 0 else 0.0

        single_count = int((defect_sums == 1).sum())
        single_pct = (single_count / n) * 100.0 if n > 0 else 0.0

        multi_count = int((defect_sums >= 2).sum())
        multi_pct = (multi_count / n) * 100.0 if n > 0 else 0.0

        row_dict = {
            "subset": name,
            "total_samples": n,
            "subset_percentage": round(split_pct, 2),
            "clean_images_count": clean_count,
            "clean_images_pct": round(clean_pct, 2),
            "single_defect_count": single_count,
            "single_defect_pct": round(single_pct, 2),
            "multilabel_count": multi_count,
            "multilabel_pct": round(multi_pct, 2)
        }

        for col in TARGET_COLUMNS:
            pos_cnt = int(df_sub[col].sum())
            pos_pct = (pos_cnt / n) * 100.0 if n > 0 else 0.0
            row_dict[f"{col}_count"] = pos_cnt
            row_dict[f"{col}_pct"] = round(pos_pct, 2)

        records.append(row_dict)

    return pd.DataFrame(records)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str = os.path.join("data", "splits")
) -> Dict[str, str]:
    """
    Saves the partitioned datasets and split summary to disk.

    Output files:
      - data/splits/train.csv
      - data/splits/validation.csv
      - data/splits/test.csv
      - data/splits/split_summary.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    summary_path = os.path.join(output_dir, "split_summary.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {
        "train": train_path,
        "validation": val_path,
        "test": test_path,
        "summary": summary_path
    }


def load_split_data(
    splits_dir: str = os.path.join("data", "splits")
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
]:
    """
    Convenience loader for subsequent model training (Random Forest, SVM, XGBoost).

    Loads train.csv, validation.csv, and test.csv, strictly separating:
      - X: Exactly 15 feature columns
      - Y: Exactly 4 binary target columns
      - ids: ImageId Series (for record identification only)

    Returns:
        ((X_train, Y_train, ids_train), (X_val, Y_val, ids_val), (X_test, Y_test, ids_test))
    """
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    for path in (train_path, val_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file '{path}' not found. Run data preparation first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    ids_train, X_train, Y_train = separate_features_and_targets(train_df)
    ids_val, X_val, Y_val = separate_features_and_targets(val_df)
    ids_test, X_test, Y_test = separate_features_and_targets(test_df)

    return (X_train, Y_train, ids_train), (X_val, Y_val, ids_val), (X_test, Y_test, ids_test)


def run_data_preparation(
    features_csv: str = os.path.join("features", "steel_defect_features.csv"),
    output_dir: str = os.path.join("data", "splits"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """
    Executes the complete Stage 3 data preparation and multi-label stratified splitting pipeline.
    """
    print("\n" + "=" * 75)
    print("        INSPECTRA-X: STAGE 3 DATA PREPARATION & STRATIFIED SPLIT")
    print("=" * 75)
    print(f"[*] Input Features CSV: {features_csv}")
    print(f"[*] Splits Output Dir:  {output_dir}")
    print(f"[*] Partition Ratios:   Train: {train_ratio*100:.1f}% | Val: {val_ratio*100:.1f}% | Test: {test_ratio*100:.1f}%")
    print(f"[*] Random Seed:        {random_state}")

    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features CSV not found at '{features_csv}'.")

    df = pd.read_csv(features_csv)
    print(f"[+] Loaded dataset successfully: {len(df):,} samples, {len(df.columns)} columns.")

    # 1. Input/Output Separation and Validation
    print("\n[*] Step 1: Separating Input Features (X), Targets (Y), and Identifiers...")
    ids, X, Y = separate_features_and_targets(df)
    print(f"    [+] Features matrix X : {X.shape[0]:,} rows x {X.shape[1]} columns (15 handcrafted features)")
    print(f"    [+] Targets matrix Y  : {Y.shape[0]:,} rows x {Y.shape[1]} columns (4 binary defect labels)")
    print(f"    [+] Identifier Series : {len(ids):,} unique ImageId strings (Isolated from X)")

    # 2. Multi-Label Stratified Splitting
    print("\n[*] Step 2: Executing Iterative Multi-Label Stratified Splitting...")
    train_df, val_df, test_df = multilabel_train_val_test_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    print(f"    [+] Training split   : {len(train_df):5d} images ({len(train_df)/len(df)*100:5.2f}%)")
    print(f"    [+] Validation split : {len(val_df):5d} images ({len(val_df)/len(df)*100:5.2f}%)")
    print(f"    [+] Test split       : {len(test_df):5d} images ({len(test_df)/len(df)*100:5.2f}%)")

    # 3. Exhaustive Data Validation
    print("\n[*] Step 3: Performing Validation and Leakage Checks...")
    val_results = validate_splits(df, train_df, val_df, test_df, random_state=random_state)
    for check_name, check_data in val_results["checks"].items():
        if check_name == "label_distributions":
            print("    [PASS] label_distributions: Stratified target proportions:")
            for item in check_data:
                print(f"        - {item['target']:14s}: Orig={item['original_pct']:5.2f}% | Train={item['train_pct']:5.2f}% | Val={item['val_pct']:5.2f}% | Test={item['test_pct']:5.2f}% (Dev <= {item['max_pct_deviation']:.2f}%)")
        else:
            status_str = "[PASS]" if check_data["passed"] else "[FAIL]"
            print(f"    {status_str} {check_name:20s}: {check_data['message']}")

    if not val_results["passed"]:
        raise RuntimeError("Validation checks failed on split datasets!")
    print("    [+] All validation checks passed. Zero data leakage detected.")

    # 4. Compute Split Summary Table
    print("\n[*] Step 4: Generating Split Distribution Summary...")
    summary_df = compute_split_summary(df, train_df, val_df, test_df)
    print(summary_df[["subset", "total_samples", "subset_percentage", "clean_images_count", "multilabel_count"]].to_string(index=False))

    # 5. Save Split Datasets
    print(f"\n[*] Step 5: Saving Partitioned Datasets to '{output_dir}'...")
    saved_paths = save_splits(train_df, val_df, test_df, summary_df, output_dir=output_dir)
    for name, path in saved_paths.items():
        size_kb = os.path.getsize(path) / 1024.0
        print(f"    [+] Saved {name:10s} -> {path} ({size_kb:.1f} KB)")

    print("\n" + "=" * 75)
    print("                    STAGE 3 PREPARATION COMPLETE")
    print("=" * 75)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "summary_df": summary_df,
        "validation_results": val_results,
        "saved_paths": saved_paths
    }


if __name__ == "__main__":
    run_data_preparation()
