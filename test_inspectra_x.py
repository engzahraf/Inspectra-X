"""
Comprehensive Test Suite for Inspectra-X End-to-End ML Pipeline.

Tests:
  1. Feature extraction across all 15 handcrafted descriptors.
  2. Candidate defect segmentation and shape features without ground-truth masks.
  3. Preprocessing, dimensions, CLAHE, and template loading.
  4. Dataset structure, CSV integrity, and multi-label stratification.
  5. Machine learning models (Random Forest, SVM, XGBoost) training and persistence.
  6. Multi-label evaluation metrics (Precision, Recall, F1, Subset Accuracy, Hamming Loss).
  7. Prediction pipeline and model loading.
  8. Error handling (invalid images, missing files, empty arrays, malformed inputs).
"""

import json
import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import unittest
import cv2
import joblib
import numpy as np
import pandas as pd

from src.dataset_loader import TARGET_COLUMNS
from src.feature_extraction import (
    FEATURE_COLUMNS,
    extract_features_for_image,
    extract_features_from_image
)
from src.preprocessing import (
    preprocess_image,
    load_and_preprocess_image,
    TARGET_IMAGE_SHAPE
)
from src.shape_features import (
    segment_candidate_defect_mask,
    calculate_connected_component_count,
    calculate_largest_component_solidity,
    calculate_defect_aspect_ratio,
    calculate_skeleton_branching_complexity
)
from src.data_preparation import load_split_data, separate_features_and_targets
from src.models import (
    train_random_forest,
    train_svm,
    train_xgboost,
    XGBoostMultiLabelClassifier
)
from src.evaluation import evaluate_multilabel_model
from src.prediction_pipeline import SteelDefectPredictor


class TestInspectraXPipeline(unittest.TestCase):
    """Test suite covering data, features, modeling, prediction, and edge cases."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic test images and load reference template."""
        # 1. Defect-free synthetic surface
        cls.clean_img = np.full((256, 1600), 128, dtype=np.uint8)

        # 2. Synthetic steel image with simulated scratches/crazing
        cls.defective_img = np.full((256, 1600), 128, dtype=np.uint8)
        # Add crack lines
        cv2.line(cls.defective_img, (100, 50), (300, 200), 20, 3)
        cv2.line(cls.defective_img, (300, 200), (450, 150), 20, 3)
        # Add pitting craters
        for x in range(600, 700, 15):
            cv2.circle(cls.defective_img, (x, 120), 4, 10, -1)
        # Add large patch
        cv2.rectangle(cls.defective_img, (900, 40), (1200, 180), 220, -1)

        # Reference template
        cls.ref_path = os.path.join("features", "reference_defect_free.png")
        if os.path.exists(cls.ref_path):
            cls.ref_img = cv2.imread(cls.ref_path, cv2.IMREAD_GRAYSCALE)
        else:
            cls.ref_img = cls.clean_img

        cls.splits_dir = os.path.join("data", "splits")
        cls.models_dir = "models"

    # -----------------------------------------------------------------
    # Test 1: Preprocessing & Dimensions
    # -----------------------------------------------------------------
    def test_preprocessing_dimensions_and_enhancement(self):
        """Verify image preprocessing enforces (256, 1600) and returns valid ranges."""
        arbitrary_img = np.random.randint(0, 256, (300, 500, 3), dtype=np.uint8)
        u8_out, f_out = preprocess_image(arbitrary_img, target_shape=TARGET_IMAGE_SHAPE)

        self.assertEqual(u8_out.shape, (256, 1600))
        self.assertEqual(f_out.shape, (256, 1600))
        self.assertEqual(u8_out.dtype, np.uint8)
        self.assertEqual(f_out.dtype, np.float64)
        self.assertTrue(0.0 <= f_out.min() and f_out.max() <= 1.0)

    # -----------------------------------------------------------------
    # Test 2: Shape Features & Leakage-Free Segmentation
    # -----------------------------------------------------------------
    def test_candidate_defect_segmentation_and_shape_features(self):
        """Verify candidate defect segmentation generates valid masks without ground truth."""
        candidate_mask = segment_candidate_defect_mask(self.defective_img, reference_image=self.ref_img)
        self.assertEqual(candidate_mask.shape, (256, 1600))
        self.assertEqual(candidate_mask.dtype, np.uint8)
        # Defective image should generate candidate components
        self.assertTrue(candidate_mask.sum() > 0)

        # Test shape features extracted directly from image (leakage-free)
        cc_count = calculate_connected_component_count(image=self.defective_img)
        solidity = calculate_largest_component_solidity(image=self.defective_img)
        aspect_ratio = calculate_defect_aspect_ratio(image=self.defective_img)
        skeleton = calculate_skeleton_branching_complexity(image=self.defective_img)

        self.assertGreaterEqual(cc_count, 0.0)
        self.assertTrue(0.0 <= solidity <= 1.0)
        self.assertGreaterEqual(aspect_ratio, 0.0)
        self.assertTrue(0.0 <= skeleton <= 1.0)

    # -----------------------------------------------------------------
    # Test 3: Feature Extraction Integrity (All 15 Features)
    # -----------------------------------------------------------------
    def test_feature_extraction_completeness_and_finite_values(self):
        """Verify exactly 15 features are returned, all non-null, finite floats."""
        feats = extract_features_from_image(self.defective_img, reference_image=self.ref_img)

        self.assertEqual(len(feats), 15)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, feats)
            val = feats[col]
            self.assertIsInstance(val, float)
            self.assertFalse(np.isnan(val), f"Feature '{col}' returned NaN")
            self.assertFalse(np.isinf(val), f"Feature '{col}' returned Inf")

    # -----------------------------------------------------------------
    # Test 4: Data Splits & Stratification
    # -----------------------------------------------------------------
    def test_data_splits_integrity(self):
        """Verify train/validation/test splits have exact sizes and zero leakage."""
        (X_tr, Y_tr, id_tr), (X_va, Y_va, id_va), (X_te, Y_te, id_te) = load_split_data(self.splits_dir)

        self.assertEqual(X_tr.shape, (8797, 15))
        self.assertEqual(Y_tr.shape, (8797, 4))
        self.assertEqual(X_va.shape, (1885, 15))
        self.assertEqual(Y_va.shape, (1885, 4))
        self.assertEqual(X_te.shape, (1886, 15))
        self.assertEqual(Y_te.shape, (1886, 4))

        # Check total record conservation
        self.assertEqual(len(X_tr) + len(X_va) + len(X_te), 12568)

        # Check zero identifier leakage across partitions
        set_tr, set_va, set_te = set(id_tr), set(id_va), set(id_te)
        self.assertEqual(len(set_tr.intersection(set_va)), 0)
        self.assertEqual(len(set_tr.intersection(set_te)), 0)
        self.assertEqual(len(set_va.intersection(set_te)), 0)

    # -----------------------------------------------------------------
    # Test 5: Model Training, Persistence & Loading
    # -----------------------------------------------------------------
    def test_models_training_and_loading(self):
        """Verify Random Forest, SVM, and XGBoost train, predict, and save correctly."""
        np.random.seed(42)
        X_toy = np.random.randn(60, 15)
        Y_toy = np.random.randint(0, 2, (60, 4))

        # Train mini models
        rf = train_random_forest(X_toy, Y_toy, n_estimators=10, max_depth=3, random_state=42)
        svm = train_svm(X_toy, Y_toy, C=1.0, random_state=42)
        xgb = train_xgboost(X_toy, Y_toy, n_estimators=10, max_depth=3, random_state=42)

        # Check predict output shapes
        pred_rf = rf.predict(X_toy)
        pred_svm = svm.predict(X_toy)
        pred_xgb = xgb.predict(X_toy)

        self.assertEqual(pred_rf.shape, (60, 4))
        self.assertEqual(pred_svm.shape, (60, 4))
        self.assertEqual(pred_xgb.shape, (60, 4))

        # Test joblib persistence & reloading
        with tempfile.TemporaryDirectory() as tmp_dir:
            rf_save_path = os.path.join(tmp_dir, "test_rf.joblib")
            joblib.dump(rf, rf_save_path)
            loaded_rf = joblib.load(rf_save_path)
            loaded_pred = loaded_rf.predict(X_toy)
            np.testing.assert_array_equal(pred_rf, loaded_pred)

    # -----------------------------------------------------------------
    # Test 6: Multi-Label Metrics Evaluation
    # -----------------------------------------------------------------
    def test_multilabel_evaluation_metrics(self):
        """Verify metric evaluation computes expected keys and range bounds."""
        Y_true = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0]
        ])
        Y_pred = np.array([
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0]
        ])

        class MockModel:
            def predict(self, X):
                return Y_pred

        mock = MockModel()
        eval_res = evaluate_multilabel_model(mock, np.zeros((4, 15)), Y_true, TARGET_COLUMNS)

        self.assertIn("micro_f1", eval_res)
        self.assertIn("macro_f1", eval_res)
        self.assertIn("subset_accuracy", eval_res)
        self.assertIn("hamming_loss", eval_res)
        self.assertIn("per_label", eval_res)
        self.assertTrue(0.0 <= eval_res["macro_f1"] <= 1.0)
        self.assertTrue(0.0 <= eval_res["hamming_loss"] <= 1.0)

    # -----------------------------------------------------------------
    # Test 7: Prediction Pipeline End-to-End
    # -----------------------------------------------------------------
    def test_prediction_pipeline_execution(self):
        """Verify SteelDefectPredictor runs end-to-end on image input."""
        predictor = SteelDefectPredictor()
        result = predictor.predict(self.clean_img)

        self.assertIn("status", result)
        self.assertIn("is_defective", result)
        self.assertIn("detected_defects", result)
        self.assertIn("defect_count", result)
        self.assertIn("class_predictions", result)
        self.assertIn("extracted_features", result)
        self.assertEqual(len(result["extracted_features"]), 15)

        # Check each defect class is present in detailed predictions
        for col in TARGET_COLUMNS:
            self.assertIn(col, result["class_predictions"])
            self.assertIn("detected", result["class_predictions"][col])
            self.assertIn("confidence_score", result["class_predictions"][col])

    # -----------------------------------------------------------------
    # Test 8: Error Handling and Edge Cases
    # -----------------------------------------------------------------
    def test_prediction_error_handling(self):
        """Verify proper exceptions are raised for invalid inputs."""
        predictor = SteelDefectPredictor()

        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            predictor.predict("non_existent_file_path_xyz.jpg")

        # None input
        with self.assertRaises(ValueError):
            predictor.predict(None)

        # Empty array
        with self.assertRaises(ValueError):
            predictor.predict(np.array([]))

        # Unsupported type
        with self.assertRaises(TypeError):
            predictor.predict(12345)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInspectraXPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
