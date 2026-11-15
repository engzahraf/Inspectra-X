"""
Production-Ready Reusable Prediction Pipeline for Inspectra-X (Stage 4).

Accepts new steel surface images, validates integrity, extracts 15 handcrafted
numerical features (using leakage-free candidate defect morphology), applies saved
scaler transformations, and outputs multi-label defect classifications with confidence scores.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import cv2
import joblib
import numpy as np
import pandas as pd

from src.dataset_loader import TARGET_COLUMNS
from src.feature_extraction import FEATURE_COLUMNS, extract_features_from_image
from src.preprocessing import preprocess_image, TARGET_IMAGE_SHAPE


class SteelDefectPredictor:
    """
    End-to-End Prediction Engine for Steel Surface Quality Assessment.
    """

    def __init__(
        self,
        model_path: str = os.path.join("models", "best_model.joblib"),
        scaler_path: str = os.path.join("models", "scaler.joblib"),
        metadata_path: str = os.path.join("models", "metadata.json"),
        reference_image_path: str = os.path.join("features", "reference_defect_free.png")
    ):
        """
        Initializes the predictor with trained model artifacts.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.reference_image_path = reference_image_path

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads and verifies required model artifacts and reference templates."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                f"Execute 'python -m src.models' to train and save models first."
            )
        self.model = joblib.load(self.model_path)

        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler = None

        if os.path.exists(self.reference_image_path):
            self.ref_image = cv2.imread(self.reference_image_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.ref_image = None

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.model_name = self.metadata.get("best_model_name", type(self.model).__name__)

    def validate_image_input(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Validates image file existence, format, readable dimensions, and channels.

        Args:
            image_input: File path (str) or loaded numpy image array.

        Returns:
            np.ndarray: Validated 2D or 3D numpy image array.
        """
        if image_input is None:
            raise ValueError("Image input cannot be None.")

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Specified image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {image_input}. File may be corrupted or unsupported.")
            return img

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty (size 0).")
            if image_input.ndim not in (2, 3):
                raise ValueError(f"Expected 2D or 3D image array, got {image_input.ndim}D array.")
            return image_input

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected filepath or numpy array.")

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        decision_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Performs full multi-label inference on a single steel surface image.

        Pipeline:
          1. Validates image input.
          2. Applies standard preprocessing (Grayscale + CLAHE + Resizing to 256x1600).
          3. Extracts the 15 handcrafted numerical features (leakage-free).
          4. Evaluates trained multi-label classifier.
          5. Computes probability / confidence scores when supported.
          6. Formulates human-readable diagnostic report.

        Args:
            image_input: Image path or numpy array.
            decision_threshold: Threshold probability for positive defect detection (default: 0.5).

        Returns:
            Dict[str, Any]: Structured prediction output.
        """
        start_time = time.time()

        # 1. Validation
        raw_image = self.validate_image_input(image_input)

        # 2. Preprocessing
        preprocessed_u8, _ = preprocess_image(raw_image, target_shape=TARGET_IMAGE_SHAPE)

        # 3. Handcrafted Feature Extraction (Exactly 15 features in canonical order)
        features_dict = extract_features_from_image(
            preprocessed_u8,
            reference_image=self.ref_image,
            target_shape=TARGET_IMAGE_SHAPE
        )

        # Build feature vector
        feat_vector = np.array([[features_dict[col] for col in FEATURE_COLUMNS]], dtype=np.float64)

        # 4. Model Prediction
        y_prob = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_vector)
            if isinstance(probs, list):
                y_prob = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probs])[0]
            elif isinstance(probs, np.ndarray) and probs.ndim == 2:
                y_prob = probs[0]
            else:
                y_prob = None

        if y_prob is not None:
            y_pred = (y_prob >= decision_threshold).astype(int)
        else:
            raw_pred = self.model.predict(feat_vector)
            if isinstance(raw_pred, list):
                y_pred = np.column_stack(raw_pred)[0]
            else:
                y_pred = np.asarray(raw_pred)[0]

        # 5. Assemble Results
        predicted_defect_names = []
        detailed_classes = {}

        for idx, col_name in enumerate(TARGET_COLUMNS):
            detected = bool(y_pred[idx] == 1)
            prob_val = float(y_prob[idx]) if y_prob is not None else None

            if detected:
                predicted_defect_names.append(col_name)

            detailed_classes[col_name] = {
                "class_id": idx + 1,
                "detected": detected,
                "confidence_score": round(prob_val, 4) if prob_val is not None else None,
                "threshold_applied": decision_threshold
            }

        is_defective = len(predicted_defect_names) > 0
        overall_status = "DEFECT DETECTED" if is_defective else "CLEAN / DEFECT-FREE"
        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "status": overall_status,
            "is_defective": is_defective,
            "detected_defects": predicted_defect_names if is_defective else ["None (Clean Steel)"],
            "defect_count": len(predicted_defect_names),
            "class_predictions": detailed_classes,
            "extracted_features": features_dict,
            "model_metadata": {
                "model_name": self.model_name,
                "inference_latency_ms": round(elapsed_ms, 2)
            }
        }


def print_prediction_report(result: Dict[str, Any]) -> None:
    """Formats and prints the prediction report to console."""
    print("\n" + "=" * 70)
    print("        INSPECTRA-X: STEEL DEFECT QUALITY ASSESSMENT REPORT")
    print("=" * 70)
    print(f"[*] Overall Status:      {result['status']}")
    print(f"[*] Defect Types Found:  {', '.join(result['detected_defects'])}")
    print(f"[*] Defect Count:        {result['defect_count']} defect class(es)")
    print(f"[*] Champion Model:      {result['model_metadata']['model_name']}")
    print(f"[*] Inference Latency:   {result['model_metadata']['inference_latency_ms']} ms")

    print("\n[*] Per-Category Diagnostics:")
    print("    ------------------------------------------------------------------")
    print("    Class ID | Defect Category | Detected | Confidence / Probability")
    print("    ------------------------------------------------------------------")
    for name, info in result["class_predictions"].items():
        det_str = "YES" if info["detected"] else "NO"
        conf_str = f"{info['confidence_score'] * 100:5.1f}%" if info["confidence_score"] is not None else "N/A"
        print(f"       {info['class_id']:1d}     | {name:15s} |   {det_str:3s}    |   {conf_str}")
    print("    ------------------------------------------------------------------")

    print("\n[*] Extracted 15 Handcrafted Image Features:")
    for idx, (fname, fval) in enumerate(result["extracted_features"].items(), 1):
        print(f"    {idx:2d}. {fname:30s} = {fval:12.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspectra-X Steel Surface Image Prediction Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to steel surface image file")
    parser.add_argument("--model", type=str, default=os.path.join("models", "best_model.joblib"), help="Path to saved model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision probability threshold (default 0.5)")
    parser.add_argument("--json", action="store_true", help="Output results strictly in JSON format")

    args = parser.parse_args()

    predictor = SteelDefectPredictor(model_path=args.model)
    result = predictor.predict(args.image, decision_threshold=args.threshold)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_prediction_report(result)


if __name__ == "__main__":
    main()
