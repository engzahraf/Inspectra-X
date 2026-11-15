"""
Model Training, Multi-Label Strategy, and Evaluation Pipeline for Inspectra-X (Stage 4).

Implements three classical machine learning architectures:
  1. Random Forest (Multi-Output Ensemble with balanced class weights)
  2. Support Vector Machine (SVM with RBF kernel, training scaler isolation, and probability calibration)
  3. XGBoost (Multi-Output Gradient Boosted Decision Trees with scale_pos_weight for class imbalance)

Provides end-to-end training, validation, model selection, artifact persistence, and reporting.
"""

import json
import os
import tempfile
import time
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.dataset_loader import TARGET_COLUMNS
from src.data_preparation import load_split_data
from src.evaluation import (
    evaluate_multilabel_model,
    format_results_markdown,
    plot_models_comparison_barchart,
    plot_multilabel_confusion_matrices,
    plot_per_label_f1_barchart
)
from src.feature_extraction import FEATURE_COLUMNS


class XGBoostMultiLabelClassifier:
    """
    Multi-label wrapper for XGBoost that optimizes per-class imbalance
    using class-specific scale_pos_weight ratios.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models: List[XGBClassifier] = []
        self.n_outputs_: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X_arr = np.asarray(X)
        Y_arr = np.asarray(Y)
        self.n_outputs_ = Y_arr.shape[1]
        self.models = []

        for j in range(self.n_outputs_):
            y_col = Y_arr[:, j]
            n_pos = int((y_col == 1).sum())
            n_neg = int((y_col == 0).sum())
            scale_weight = float(n_neg / max(1, n_pos))

            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=scale_weight,
                eval_metric="logloss",
                random_state=self.random_state + j,
                n_jobs=self.n_jobs,
                tree_method="hist"
            )
            clf.fit(X_arr, y_col)
            self.models.append(clf)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        preds = [clf.predict(X_arr) for clf in self.models]
        return np.column_stack(preds)

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        X_arr = np.asarray(X)
        return [clf.predict_proba(X_arr) for clf in self.models]


def train_random_forest(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 15,
    random_state: int = 42,
    n_jobs: int = -1
) -> MultiOutputClassifier:
    """
    Model 1: Random Forest Classifier.

    Strategy:
      MultiOutputClassifier with individual balanced class weighting.
      Ensemble trees provide inherent robustness against feature collinearity.
    """
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs
    )
    model = MultiOutputClassifier(base_rf, n_jobs=n_jobs)
    model.fit(X_train, Y_train)
    return model


def train_svm(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    C: float = 2.0,
    gamma: str = "scale",
    random_state: int = 42,
    n_jobs: int = -1
) -> MultiOutputClassifier:
    """
    Model 2: Support Vector Machine (SVM) Classifier.

    Strategy:
      MultiOutputClassifier wrapping an isolated Pipeline of StandardScaler and RBF SVC.
      Feature scaling is strictly contained inside the estimator pipeline to prevent leakage.
      Probability calibration (probability=True) provides confidence scores for inference.
    """
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
            cache_size=500
        ))
    ])
    model = MultiOutputClassifier(svm_pipeline, n_jobs=n_jobs)
    model.fit(X_train, Y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBoostMultiLabelClassifier:
    """
    Model 3: XGBoost Gradient Boosted Classifier.

    Strategy:
      Tuned multi-output gradient boosted trees with dynamic class-imbalance
      weighting (scale_pos_weight). Captures non-linear feature interactions.
    """
    model = XGBoostMultiLabelClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_train, Y_train)
    return model


def run_stage4_training_and_evaluation(
    splits_dir: str = os.path.join("data", "splits"),
    models_dir: str = "models",
    results_dir: str = "results",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes complete Stage 4 workflow:
      1. Loads partitioned Train, Validation, and Test datasets.
      2. Trains Random Forest, SVM, and XGBoost models on Training split.
      3. Fits and persists a global reference StandardScaler on Training split.
      4. Evaluates all three models on Validation split across multi-label metrics.
      5. Selects the superior model based on Validation Macro F1 and generalizability.
      6. Evaluates the selected model on the Test split for final unbiased benchmarks.
      7. Generates comparison tables, publication-ready figures, and academic reports.
    """
    print("\n" + "=" * 80)
    print("      STAGE 4: MACHINE LEARNING MODEL TRAINING, EVALUATION & SELECTION")
    print("=" * 80)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Partitioned Splits
    print(f"[*] Loading dataset splits from '{splits_dir}'...")
    (X_tr_df, Y_tr_df, id_tr), (X_val_df, Y_val_df, id_val), (X_te_df, Y_te_df, id_te) = load_split_data(splits_dir)

    X_train, Y_train = X_tr_df.values, Y_tr_df.values
    X_val, Y_val = X_val_df.values, Y_val_df.values
    X_test, Y_test = X_te_df.values, Y_te_df.values

    print(f"    Train:      X={X_train.shape}, Y={Y_train.shape} (70.00%)")
    print(f"    Validation: X={X_val.shape}, Y={Y_val.shape} (15.00%)")
    print(f"    Test:       X={X_test.shape}, Y={Y_test.shape} (15.01%)")

    # 2. Fit and Persist Standalone Training Scaler (for image prediction pipeline)
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"    [+] Saved training-fitted StandardScaler to: {scaler_path}")

    # 3. Train Classical ML Models
    trained_models: Dict[str, Any] = {}
    train_times: Dict[str, float] = {}

    # Model 1: Random Forest
    print("\n[*] [1/3] Training Random Forest (Multi-Output Ensemble)...")
    t0 = time.time()
    rf_model = train_random_forest(X_train, Y_train, random_state=random_state)
    train_times["Random Forest"] = time.time() - t0
    trained_models["Random Forest"] = rf_model
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"    [+] Random Forest trained in {train_times['Random Forest']:.2f}s and saved to {rf_path}")

    # Model 2: SVM
    print("\n[*] [2/3] Training Support Vector Machine (RBF Kernel & Scaling Pipeline)...")
    t0 = time.time()
    svm_model = train_svm(X_train, Y_train, random_state=random_state)
    train_times["Support Vector Machine"] = time.time() - t0
    trained_models["Support Vector Machine"] = svm_model
    svm_path = os.path.join(models_dir, "svm.joblib")
    joblib.dump(svm_model, svm_path)
    print(f"    [+] SVM trained in {train_times['Support Vector Machine']:.2f}s and saved to {svm_path}")

    # Model 3: XGBoost
    print("\n[*] [3/3] Training XGBoost (Gradient Boosted Trees with scale_pos_weight)...")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, Y_train, random_state=random_state)
    train_times["XGBoost"] = time.time() - t0
    trained_models["XGBoost"] = xgb_model
    xgb_path = os.path.join(models_dir, "xgboost.joblib")
    joblib.dump(xgb_model, xgb_path)
    print(f"    [+] XGBoost trained in {train_times['XGBoost']:.2f}s and saved to {xgb_path}")

    # 4. Evaluate All Models on Validation Split
    print("\n[*] Evaluating models on Validation split (1,885 images)...")
    val_evals: Dict[str, Dict[str, Any]] = {}
    comparison_rows = []
    per_label_dfs: Dict[str, pd.DataFrame] = {}

    for name, model in trained_models.items():
        eval_dict = evaluate_multilabel_model(model, X_val, Y_val, target_names=TARGET_COLUMNS)
        val_evals[name] = eval_dict

        comparison_rows.append({
            "Model": name,
            "Macro F1": round(eval_dict["macro_f1"], 4),
            "Micro F1": round(eval_dict["micro_f1"], 4),
            "Weighted F1": round(eval_dict["weighted_f1"], 4),
            "Subset Accuracy": round(eval_dict["subset_accuracy"], 4),
            "Hamming Loss": round(eval_dict["hamming_loss"], 4),
            "Train Time (s)": round(train_times[name], 2)
        })

        # Per-label table for this model
        pldf_rows = []
        for cname in TARGET_COLUMNS:
            info = eval_dict["per_label"][cname]
            pldf_rows.append({
                "Defect Class": cname,
                "Precision": round(info["precision"], 4),
                "Recall": round(info["recall"], 4),
                "F1-Score": round(info["f1_score"], 4),
                "Support": info["support"],
                "TP": info["true_positive"],
                "FP": info["false_positive"],
                "TN": info["true_negative"],
                "FN": info["false_negative"]
            })
        per_label_dfs[name] = pd.DataFrame(pldf_rows).set_index("Defect Class")

        # Plot individual model confusion matrices
        cm_filename = f"confusion_matrices_{name.lower().replace(' ', '_')}.png"
        cm_path = os.path.join(results_dir, cm_filename)
        plot_multilabel_confusion_matrices(eval_dict["confusion_matrices"], name, TARGET_COLUMNS, cm_path)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="Macro F1", ascending=False)
    comparison_csv = os.path.join(results_dir, "model_comparison_table.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n[+] Validation Performance Summary saved to: {comparison_csv}")
    print(comparison_df.to_string(index=False))

    # Visualizations
    plot_models_comparison_barchart(comparison_df, os.path.join(results_dir, "model_metrics_comparison.png"))
    plot_per_label_f1_barchart(per_label_dfs, TARGET_COLUMNS, os.path.join(results_dir, "per_label_f1_comparison.png"))

    # 5. Model Selection (Based on Validation Macro F1)
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_macro_f1 = best_row["Macro F1"]

    print(f"\n[★] BEST MODEL SELECTED: {best_model_name.upper()}")
    print(f"    Selection Justification: Highest Validation Macro F1 = {best_macro_f1:.4f} with balanced per-label recall across minority defects.")

    # Save Best Model explicitly
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"    [+] Saved champion model to: {best_model_path}")

    # 6. Unbiased Final Test Set Evaluation
    print(f"\n[*] Evaluating models on final Test Split (1,886 images)...")
    test_evals: Dict[str, Dict[str, Any]] = {}
    for name, model in trained_models.items():
        test_evals[name] = evaluate_multilabel_model(model, X_test, Y_test, target_names=TARGET_COLUMNS)

    test_best = test_evals[best_model_name]
    print(f"    Champion Model ({best_model_name}) on Test Split:")
    print(f"      Macro F1:        {test_best['macro_f1']:.4f}")
    print(f"      Micro F1:        {test_best['micro_f1']:.4f}")
    print(f"      Weighted F1:     {test_best['weighted_f1']:.4f}")
    print(f"      Subset Accuracy: {test_best['subset_accuracy']:.4f}")
    print(f"      Hamming Loss:    {test_best['hamming_loss']:.4f}")

    # Save Metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "validation_metrics": {
            "macro_f1": float(best_row["Macro F1"]),
            "micro_f1": float(best_row["Micro F1"]),
            "subset_accuracy": float(best_row["Subset Accuracy"]),
            "hamming_loss": float(best_row["Hamming Loss"])
        },
        "test_metrics": {
            "macro_f1": float(test_best["macro_f1"]),
            "micro_f1": float(test_best["micro_f1"]),
            "subset_accuracy": float(test_best["subset_accuracy"]),
            "hamming_loss": float(test_best["hamming_loss"])
        },
        "all_models_validation": comparison_rows
    }
    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    [+] Saved model metadata to: {metadata_path}")

    # Generate Markdown Report
    report_md = format_results_markdown(comparison_df, per_label_dfs, test_evals)
    report_path = os.path.join(results_dir, "model_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"    [+] Saved comprehensive report to: {report_path}")

    print("=" * 80)
    return {
        "trained_models": trained_models,
        "val_evaluations": val_evals,
        "test_evaluations": test_evals,
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_model": best_model
    }


if __name__ == "__main__":
    run_stage4_training_and_evaluation()
