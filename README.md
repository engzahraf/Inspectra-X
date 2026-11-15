# INSPECTRA-X: ML-Based Automated Steel Surface Defect Classification and Quality Assessment System

An academic image-processing, feature-engineering, and classical machine learning system developed for the **Severstal Steel Defect Detection** benchmark. The pipeline extracts **exactly 15 handcrafted numerical features** across texture, edge, morphological, and transform domains without deep neural networks, performs comprehensive data quality and collinearity analysis, executes leakage-free iterative multi-label stratified dataset splitting, trains and benchmarks classical multi-label classifiers (**Random Forest**, **Support Vector Machine**, and **XGBoost**), and provides an end-to-end real-time prediction pipeline.

---

## 1. Project Directory Structure

```text
Inspectra-X/
│
├── Dataset/                                # Raw Severstal dataset (optional if features CSV exists)
│   ├── train.csv                           # ImageId, ClassId, EncodedPixels
│   └── train_images/                       # 12,568 steel surface images (256x1600)
│
├── src/                                    # Modular source packages
│   ├── __init__.py
│   ├── dataset_loader.py                   # Dataset indexing & multi-label aggregation
│   ├── mask_decoder.py                     # Fortran-order Severstal RLE decoder & union mask builder
│   ├── preprocessing.py                    # Grayscale CLAHE enhancement & reference template
│   ├── texture_features.py                 # Features 1–5: GLCM Contrast, Corr, Energy, LBP, Gabor
│   ├── edge_features.py                    # Features 6–8: Canny Density, Sobel Magnitude, Laplacian Var
│   ├── shape_features.py                   # Features 9–12: Component Count, Solidity, Aspect Ratio, Skeleton
│   ├── statistical_features.py             # Features 13–15: Entropy, Reference SSIM, FFT Peak Freq
│   ├── feature_extraction.py               # Unified extraction pipeline (ground-truth & inference)
│   ├── data_quality.py                     # Stage 2: Structural validation, outlier, collinearity & targets
│   ├── data_preparation.py                # Stage 3: Leakage-free iterative multi-label stratified splitting
│   ├── models.py                           # Stage 4: Random Forest, SVM, and XGBoost training & tuning
│   ├── evaluation.py                       # Multi-label metrics, confusion matrices, and reporting
│   └── prediction_pipeline.py              # End-to-end single-image inference and diagnostic pipeline
│
├── features/
│   ├── reference_defect_free.png           # Reproducible defect-free reference template
│   └── steel_defect_features.csv           # Extracted 15 features + 4 multi-label target columns (12,568 rows)
│
├── data/
│   └── splits/                             # Stage 3 Partitioned Datasets (70 / 15 / 15)
│       ├── train.csv                       # Training split (8,797 records, 70.00%)
│       ├── validation.csv                  # Validation split (1,885 records, 15.00%)
│       ├── test.csv                        # Test split (1,886 records, 15.01%)
│       └── split_summary.csv               # Distribution comparison and defect counts per split
│
├── models/                                 # Trained Model Artifacts & Champion Classifier
│   ├── random_forest.joblib                # Trained Multi-Output Random Forest Classifier
│   ├── svm.joblib                          # Trained Multi-Output Scaled SVM Classifier
│   ├── xgboost.joblib                      # Trained Multi-Output XGBoost Classifier
│   ├── scaler.joblib                       # Standalone StandardScaler fit strictly on Training split
│   ├── best_model.joblib                   # Selected Champion Classifier (Random Forest)
│   └── metadata.json                       # Model hyperparameters, feature orders, and benchmark scores
│
├── results/                                # Analytical Outputs, Visualizations & Academic Reports
│   ├── summary_statistics.csv              # Baseline descriptive statistics per feature
│   ├── feature_distributions.png           # 3x5 grid histogram visualization
│   ├── correlation_heatmap.png             # Pairwise Pearson correlation matrix
│   ├── target_class_distribution.png       # Positive vs Negative counts per defect class
│   ├── multilabel_distribution.png         # Defect counts per image (clean vs single vs multi-label)
│   ├── feature_outliers_boxplot.png        # 3x5 grid of IQR boxplots with outlier percentages
│   ├── label_cooccurrence_heatmap.png      # 4x4 defect co-occurrence frequency heatmap
│   ├── label_cooccurrence_matrix.csv       # Tabular 4x4 defect co-occurrence matrix
│   ├── label_distribution_summary.csv      # Target distribution and imbalance metrics
│   ├── data_quality_report.csv             # Detailed tabular feature quality & IQR outlier report
│   ├── data_quality_report.md              # Comprehensive academic data quality report
│   ├── model_comparison_table.csv          # Tabular model performance metrics comparison
│   ├── model_comparison_report.md          # Comprehensive academic model evaluation report
│   ├── model_metrics_comparison.png        # Bar chart comparing Macro/Micro F1, Accuracy, Hamming Loss
│   ├── per_label_f1_comparison.png         # Grouped bar chart of per-class F1-scores across models
│   ├── confusion_matrices_random_forest.png# 2x2 grid confusion matrices for Random Forest
│   ├── confusion_matrices_support_vector_machine.png # 2x2 grid confusion matrices for SVM
│   └── confusion_matrices_xgboost.png      # 2x2 grid confusion matrices for XGBoost
│
├── requirements.txt                        # Project dependencies
├── main.py                                 # Command-line entry point coordinating Stages 1–4 and Prediction
├── test_stage2_stage3.py                   # Verification suite for Stages 2 & 3
├── test_inspectra_x.py                     # Comprehensive end-to-end verification and regression suite
└── README.md
```

---

## 2. Problem Formulation: Multi-Label Defect Classification

In continuous industrial rolling mills, a single flat-rolled steel sheet can present zero defects (clean steel), an isolated defect, or multiple defect types simultaneously. Converting this to single-class multiclass classification drops critical co-occurrence patterns and introduces label distortion.

Inspectra-X strictly models steel quality inspection as a **Multi-Label Binary Classification Problem**:
$$Y = [y_1, y_2, y_3, y_4] \in \{0, 1\}^4$$

### Defect Taxonomy:
1. **Class 1 – Crazing:** High-frequency, network-like stress micro-cracks (897 positive images, 7.14%).
2. **Class 2 – Inclusion:** Subsurface non-metallic foreign particles (247 positive images, 1.97% – severe 20.85:1 class imbalance relative to Patches [5,150 / 247 ≈ 20.85:1], and 49.88:1 binary negative-to-positive ratio).
3. **Class 3 – Patches:** Broad plate-like surface deformations (5,150 positive images, 40.98%).
4. **Class 4 – Pitted Surface:** Localized cratering and pinhole indentations (801 positive images, 6.37%).

**Dataset Composition (12,568 Total Images):**
- **Clean Steel (0 defects):** 5,902 images (46.96%)
- **Single-Defect Images:** 6,239 images (49.64%)
- **Multi-Label Images ($\ge 2$ defects):** 427 images (3.40%)

---

## 3. Exactly 15 Handcrafted Numerical Features

To comply with academic interpretability standards and avoid deep neural networks or black-box convolutional backbones, the system extracts exactly 15 numerical descriptors:

### Group 1: Texture Features (Preprocessed Grayscale Image)
1. **GLCM Contrast:** Second-order spatial intensity variance across neighboring pixels at distance $d=1$ across angles $(0^\circ, 45^\circ, 90^\circ, 135^\circ)$.
2. **GLCM Correlation:** Linear dependency of gray levels along principal rolling directions.
3. **GLCM Energy:** Angular second moment / textural uniformity ($ASM$). High for pristine, uniform steel sheets.
4. **LBP Histogram Entropy:** Shannon entropy of rotation-invariant uniform Local Binary Pattern distribution ($P=8, R=1$).
5. **Gabor Filter Response Energy:** Directional multi-scale filter bank response energy ($\lambda=4.0, \theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$).

### Group 2: Edge Features (Preprocessed Grayscale Image)
6. **Canny Edge Density:** Fraction of edge pixels under adaptive Otsu-derived hysteresis thresholds.
7. **Sobel Edge Magnitude (Mean):** Average Euclidean gradient magnitude $\sqrt{G_x^2 + G_y^2}$.
8. **Laplacian Edge Variance:** Variance of second spatial derivative $\text{Var}(\nabla^2 I)$, measuring surface micro-roughness.

### Group 3: Shape & Morphological Features
9. **Connected Component Count:** Number of 8-connected foreground defect islands ($0.0$ for clean steel, high for pitting).
10. **Largest Component Solidity:** Ratio of area to convex hull area ($0.0$ for clean steel, low for dendritic crazing).
11. **Defect Aspect Ratio:** Major-to-minor axis ratio of the largest defect bounding box (differentiates elongated longitudinal rolling cracks from isotropic pits).
12. **Skeleton Branching Complexity:** Ratio of junction branch points to total medial-axis skeleton length (quantifies branched crazing crack networks).

### Group 4: Statistical & Frequency Features
13. **Image Entropy:** Shannon entropy of the 256-bin global intensity histogram.
14. **SSIM to Reference:** Structural Similarity Index against the canonical defect-free template `reference_defect_free.png`.
15. **Fourier Spectrum Peak Frequency:** Dominant non-DC radial spatial frequency from centered 2D Fast Fourier Transform ($|F(u, v)|$).

### Data Leakage Prevention in Shape Features
During training, shape features were extracted using ground-truth masks. For real-time prediction and inference on new unlabeled images where ground-truth masks are unavailable, Inspectra-X employs **Candidate Defect Region Segmentation** (`src/shape_features.py`):
1. Applies morphological Top-Hat (bright anomalies) and Black-Hat (dark cavities, pits, cracks) transformations.
2. Blends morphological anomaly energy with absolute reference template difference.
3. Segments candidate foreground defect islands via dynamic Otsu thresholding without ground truth.
4. Computes identical geometric and topological descriptors (Connected Components, Solidity, Aspect Ratio, Skeleton Complexity).

---

## 4. Stage 3: Data Preparation and Stratified Splitting

To ensure strict experimental integrity and eliminate data leakage:
- **Input/Output Isolation:** Isolates `ImageId` strictly as an identifier. Feature matrix $X$ contains exactly the 15 handcrafted numerical features, and $Y$ contains the 4 target columns.
- **Iterative Multi-Label Stratification:** Utilizes `MultilabelStratifiedShuffleSplit` with fixed `random_state=42` to partition the dataset:
  - **Training Set:** 8,797 samples (70.00%)
  - **Validation Set:** 1,885 samples (15.00%)
  - **Test Set:** 1,886 samples (15.01%)
- **Distribution Fidelity:** Preserves class proportions with maximum deviation $\le 0.03\%$ across all subsets.
- **Data Leakage Prevention:** Data is split prior to any normalization or scaling. `StandardScaler` is fit strictly on the training set.

---

## 5. Machine Learning Models & Multi-Label Strategies

Inspectra-X implements and compares three classical machine learning architectures:

### 1. Random Forest Classifier
- **Multi-Label Strategy:** `MultiOutputClassifier` wrapping `RandomForestClassifier`.
- **Hyperparameters:** `n_estimators=150`, `max_depth=15`, `min_samples_split=4`, `class_weight='balanced'`, `random_state=42`.
- **Properties:** Inherent robustness against feature collinearity; balanced class weights mitigate severe class imbalance.

### 2. Support Vector Machine (SVM)
- **Multi-Label Strategy:** `MultiOutputClassifier` wrapping an isolated `Pipeline([('scaler', StandardScaler()), ('svm', SVC())])`.
- **Hyperparameters:** `kernel='rbf'`, `C=2.0`, `gamma='scale'`, `probability=True`, `class_weight='balanced'`, `random_state=42`.
- **Properties:** Non-linear radial basis function; strict scaler containment inside the pipeline eliminates cross-fold leakage; calibrated probability output supports confidence scoring.

### 3. XGBoost (Extreme Gradient Boosting)
- **Multi-Label Strategy:** Independent gradient boosted decision trees wrapped in `XGBoostMultiLabelClassifier`.
- **Hyperparameters:** `n_estimators=150`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `scale_pos_weight` dynamically set to $N_{\text{neg}} / N_{\text{pos}}$ for each class, `random_state=42`.
- **Properties:** Captures non-linear feature interactions; dynamic positive weighting dramatically enhances recall on minority defect classes.

---

## 6. Actual Experimental Results & Model Comparison

All results are obtained from execution on the stratified splits (`random_state=42`):

### Overall Performance Comparison (Validation Split – 1,885 Images)

| Model | Macro F1 | Micro F1 | Weighted F1 | Subset Accuracy | Hamming Loss | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.7478** | **0.8519** | **0.8558** | **0.8562** | **0.0452** | **6.53 s** |
| **XGBoost** | 0.7475 | 0.8468 | 0.8557 | 0.8446 | 0.0477 | 2.90 s |
| **Support Vector Machine** | 0.6948 | 0.8470 | 0.8381 | 0.8785 | 0.0424 | 10.00 s |

### Detailed Per-Label Metrics on Validation Split

#### Random Forest (Selected Champion)
| Defect Class | Precision | Recall | F1-Score | Support | True Positive | False Positive | True Negative | False Negative |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crazing** | 0.6347 | 0.7910 | **0.7043** | 134 | 106 | 61 | 1,690 | 28 |
| **Inclusion** | 0.5918 | 0.7838 | **0.6744** | 37 | 29 | 20 | 1,828 | 8 |
| **Patches** | 0.8686 | 0.9676 | **0.9154** | 772 | 747 | 113 | 1,000 | 25 |
| **Pitted Surface** | 0.6037 | 0.8250 | **0.6972** | 120 | 99 | 65 | 1,700 | 21 |

#### XGBoost
| Defect Class | Precision | Recall | F1-Score | Support | True Positive | False Positive | True Negative | False Negative |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crazing** | 0.5808 | 0.8582 | 0.6928 | 134 | 115 | 83 | 1,668 | 19 |
| **Inclusion** | 0.6078 | 0.8378 | **0.7045** | 37 | 31 | 20 | 1,828 | 6 |
| **Patches** | 0.8794 | 0.9637 | **0.9197** | 772 | 744 | 102 | 1,011 | 28 |
| **Pitted Surface** | 0.5469 | 0.8750 | 0.6731 | 120 | 105 | 87 | 1,678 | 15 |

#### Support Vector Machine (SVM)
| Defect Class | Precision | Recall | F1-Score | Support | True Positive | False Positive | True Negative | False Negative |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crazing** | 0.7400 | 0.5522 | 0.6325 | 134 | 74 | 26 | 1,725 | 60 |
| **Inclusion** | 0.8500 | 0.4595 | 0.5965 | 37 | 17 | 3 | 1,845 | 20 |
| **Patches** | 0.8952 | 0.9404 | 0.9172 | 772 | 726 | 85 | 1,028 | 46 |
| **Pitted Surface** | 0.7041 | 0.5750 | 0.6330 | 120 | 69 | 29 | 1,736 | 51 |

### Unbiased Final Evaluation on Test Split (1,886 Images)

| Model | Macro F1 | Micro F1 | Weighted F1 | Subset Accuracy | Hamming Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.7668** | **0.8468** | **0.8500** | **0.8542** | **0.0467** |
| **XGBoost** | 0.7635 | 0.8466 | 0.8545 | 0.8499 | 0.0475 |
| **Support Vector Machine** | 0.7359 | 0.8444 | 0.8374 | 0.8749 | 0.0435 |

---

## 7. Model Selection Justification

**Random Forest was selected as the champion model for the following academic reasons:**
1. **Highest Macro F1-Score:** Achieves top Macro F1 on both the Validation Split (**0.7478**) and unseen Test Split (**0.7668**).
2. **Balanced Recall across Severe Class Imbalance:** Correctly captures 78.4% of Inclusion defects (1.97% prevalence) and 82.5% of Pitted Surface defects while maintaining 86.9% precision on Patches.
3. **High Exact-Match Ratio:** Achieves 85.62% Subset Accuracy and low 0.0452 Hamming Loss.
4. **Collinearity Immunity:** Random Forest's random feature subsampling ($m = \sqrt{15}$) natively navigates the 12 collinear pairs identified in Stage 2 without model destabilization.

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.14)
- Git

### Installation
```bash
git clone https://github.com/engzahraf/Inspectra-X.git
cd Inspectra-X
git checkout Imthath
pip install -r requirements.txt
```

### Complete End-to-End Pipeline (Stages 1–4)
```bash
python main.py
```

### Stage-by-Stage Execution
```bash
# Stage 2: Feature Quality & Collinearity Analysis
python main.py --stage 2

# Stage 3: Multi-Label Stratified Dataset Partitioning
python main.py --stage 3

# Stage 4: Model Training, Evaluation & Selection
python main.py --stage 4
# OR
python -m src.models
```

---

## 9. Prediction Pipeline on New Images

Inspectra-X includes a production-ready real-time diagnostic pipeline:

```bash
# Predict defects on an individual steel image
python main.py --predict features/reference_defect_free.png

# OR directly invoke the prediction pipeline module
python -m src.prediction_pipeline --image path/to/steel_image.jpg
```

### Python API Usage
```python
from src.prediction_pipeline import SteelDefectPredictor

predictor = SteelDefectPredictor()
result = predictor.predict("path/to/steel_surface.jpg")

print("Status:", result["status"])
print("Detected Defects:", result["detected_defects"])
print("Class Predictions:", result["class_predictions"])
```

---

## 10. Verification and Testing

Inspectra-X includes an automated verification and regression test suite:

```bash
# Run comprehensive pipeline test suite
python test_inspectra_x.py

# Run Stage 2 & Stage 3 verification suite
python test_stage2_stage3.py
```

---

## 11. Limitations and Future Improvements

### Limitations
1. **Resolution Downsampling:** The original 1600-pixel longitudinal steel strip is processed at $256 \times 1600$. Extremely microscopic pinhole pits (< 3 pixels) may be smoothed during CLAHE.
2. **Extreme Class Imbalance:** Class 2 (Inclusion) comprises only 247 images across 12,568 samples (1.97%), yielding a smaller absolute support during evaluation.

### Future Improvements
1. **Adaptive Region-of-Interest (ROI) Cropping:** Extracting high-resolution sliding window patches for localized defect identification.
2. **Threshold Tuning (Precision-Recall Optimization):** Tuning individual decision thresholds per class on validation curves to boost precision on rare defect classes.
3. **Ensemble Blending:** Stacking Random Forest and XGBoost probability estimates via soft voting to capitalize on complementary tree structures.
