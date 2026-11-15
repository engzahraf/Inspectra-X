# Inspectra-X: ML-Based Automated Steel Surface Defect Classification

An academic image-processing, feature-engineering, and classical machine learning system developed for the **Severstal Steel Defect Detection** benchmark. The pipeline extracts **exactly 15 handcrafted numerical features** across texture, edge, morphological, and transform domains without deep neural networks, performs comprehensive data quality and collinearity analysis, and executes leakage-free iterative multi-label stratified dataset splitting.

---

## 1. Project Directory Structure

```text
Inspectra-X/
│
├── Dataset/                            # Raw dataset (optional if features CSV exists)
│   ├── train.csv                       # ImageId, ClassId, EncodedPixels
│   └── train_images/                   # 12,568 steel surface images (256x1600)
│
├── src/                                # Modular source packages
│   ├── __init__.py
│   ├── dataset_loader.py               # Dataset indexing & multi-label aggregation
│   ├── mask_decoder.py                 # Fortran-order Severstal RLE decoder & union mask builder
│   ├── preprocessing.py                # Grayscale CLAHE enhancement & reference template
│   ├── texture_features.py             # Features 1–5: GLCM Contrast, Corr, Energy, LBP, Gabor
│   ├── edge_features.py                # Features 6–8: Canny Density, Sobel Magnitude, Laplacian Var
│   ├── shape_features.py               # Features 9–12: Component Count, Solidity, Aspect Ratio, Skeleton
│   ├── statistical_features.py         # Features 13–15: Entropy, Reference SSIM, FFT Peak Freq
│   ├── feature_extraction.py           # Unified extraction pipeline & parallel workers
│   ├── data_quality.py                 # Stage 2: Structural validation, outlier, collinearity & target analysis
│   └── data_preparation.py            # Stage 3: Leakage-free iterative multi-label stratified splitting
│
├── features/
│   ├── reference_defect_free.png       # Reproducible defect-free reference template
│   └── steel_defect_features.csv       # Extracted 15 features + 4 multi-label target columns (12,568 rows)
│
├── data/
│   └── splits/                         # Stage 3 Partitioned Datasets (70 / 15 / 15)
│       ├── train.csv                   # Training split (8,797 records, 70.00%)
│       ├── validation.csv              # Validation split (1,885 records, 15.00%)
│       ├── test.csv                    # Test split (1,886 records, 15.01%)
│       └── split_summary.csv           # Distribution comparison and defect counts per split
│
├── results/                            # Stage 1 & Stage 2 Analytical Outputs
│   ├── summary_statistics.csv          # Baseline descriptive statistics per feature
│   ├── feature_distributions.png       # 3x5 grid histogram visualization
│   ├── correlation_heatmap.png         # Pairwise Pearson correlation matrix
│   ├── target_class_distribution.png   # Positive vs Negative counts per defect class
│   ├── multilabel_distribution.png     # Defect counts per image (clean vs single vs multi-label)
│   ├── feature_outliers_boxplot.png    # 3x5 grid of IQR boxplots with outlier percentages
│   ├── label_cooccurrence_heatmap.png  # 4x4 defect co-occurrence frequency heatmap
│   ├── label_cooccurrence_matrix.csv   # Tabular 4x4 defect co-occurrence matrix
│   ├── label_distribution_summary.csv  # Target distribution and imbalance metrics
│   ├── data_quality_report.csv         # Detailed tabular feature quality & IQR outlier report
│   └── data_quality_report.md          # Comprehensive academic data quality report
│
├── requirements.txt                    # Project dependencies
├── main.py                             # Command-line entry point coordinating Stages 1, 2, and 3
└── README.md
```

---

## 2. Problem Formulation: Multi-Label Defect Classification

This system frames steel surface quality inspection as a **Multi-Label Classification** problem. A single steel sheet may exhibit multiple defects simultaneously:

1. **Class 1 – Crazing:** High-frequency, network-like stress micro-cracks (897 positive images, 7.14%).
2. **Class 2 – Inclusion:** Subsurface non-metallic foreign particles (247 positive images, 1.97% – highly imbalanced).
3. **Class 3 – Patches:** Broad plate-like surface deformations (5,150 positive images, 40.98%).
4. **Class 4 – Pitted Surface:** Localized cratering and small indentations (801 positive images, 6.37%).

**Dataset Composition (12,568 Total Images):**
- **Clean Steel (0 defects):** 5,902 images (46.96%)
- **Single-Defect Images:** 6,239 images (49.64%)
- **Multi-Label Images ($\ge 2$ defects):** 427 images (3.40%)

---

## 3. Exactly 15 Handcrafted Features

### Group 1: Texture Features (Preprocessed Grayscale Image)
1. **GLCM Contrast:** Second-order spatial intensity variance across neighboring pixels at distance $d=1$ across angles $(0^\circ, 45^\circ, 90^\circ, 135^\circ)$.
2. **GLCM Correlation:** Linear dependency of pixel values along rolling directions.
3. **GLCM Energy:** Angular second moment / textural uniformity ($ASM$). High for pristine, uniform steel sheets.
4. **LBP Histogram Entropy:** Shannon entropy of rotation-invariant uniform Local Binary Pattern distribution ($P=8, R=1$).
5. **Gabor Filter Response Energy:** Directional filter bank response energy ($\lambda=4.0, \theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$).

### Group 2: Edge Features (Preprocessed Grayscale Image)
6. **Canny Edge Density:** Fraction of edge pixels under adaptive Otsu-derived hysteresis thresholds.
7. **Sobel Edge Magnitude (Mean):** Average Euclidean gradient magnitude $\sqrt{G_x^2 + G_y^2}$.
8. **Laplacian Edge Variance:** Variance of second spatial derivative $\text{Var}(\nabla^2 I)$, measuring surface roughness.

### Group 3: Shape & Morphological Features (Decoded Union Mask)
9. **Connected Component Count:** Number of 8-connected foreground defect islands ($0.0$ for clean steel, high for pitting).
10. **Largest Component Solidity:** Ratio of area to convex hull area ($0.0$ for clean steel, low for dendritic crazing).
11. **Defect Aspect Ratio:** Major-to-minor axis ratio of the largest defect bounding box (differentiates elongated longitudinal rolling cracks from isotropic pits).
12. **Skeleton Branching Complexity:** Ratio of junction branch points to total medial-axis skeleton length (quantifies branched crazing crack networks).

### Group 4: Statistical & Frequency Features
13. **Image Entropy:** Shannon entropy of the 256-bin global intensity histogram.
14. **SSIM to Reference:** Structural Similarity Index against the canonical defect-free template `reference_defect_free.png`.
15. **Fourier Spectrum Peak Frequency:** Dominant non-DC radial spatial frequency from centered 2D Fast Fourier Transform ($|F(u, v)|$).

---

## 4. Stage 2: Feature Analysis and Data Quality

The Stage 2 module (`src/data_quality.py`) provides rigorous academic validation of the extracted tabular dataset:

- **Structural Integrity:** Verifies 12,568 rows, 20 expected columns (`ImageId` + 15 features + 4 targets), zero missing values, zero infinite values, and strict binary target encoding.
- **Statistical Characterization:** Evaluates mean, median, standard deviation, quartiles, and skewness for all 15 features.
- **Outlier Detection via IQR:** Flags values exceeding $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$. Outliers are academically justified as genuine defect signals (e.g. high crack branching or pit density) and preserved.
- **Collinearity & Redundancy:** Discovers 12 highly correlated feature pairs ($|r| \ge 0.70$), notably between `glcm_contrast` and `laplacian_variance` ($r = 0.9941$), identifying the Surface Roughness and Entropy/Uniformity feature clusters.
- **Multi-Label Co-Occurrence:** Computes the $4 \times 4$ co-occurrence matrix $Y^T Y$, discovering that Patches and Pitted Surface co-occur most frequently (284 images), while Crazing and Pitted Surface never co-occur.

---

## 5. Stage 3: Data Preparation and Stratified Splitting

The Stage 3 module (`src/data_preparation.py`) partitions the dataset for machine learning:

- **Input/Output Isolation:** Isolates `ImageId` strictly as an identifier. Feature matrix $X$ contains exactly the 15 handcrafted numerical features, and $Y$ contains the 4 target columns.
- **Iterative Multi-Label Stratification:** Utilizes `MultilabelStratifiedShuffleSplit` with fixed `random_state=42` to partition the dataset into:
  - **Training Set:** 8,797 samples (70.00%)
  - **Validation Set:** 1,885 samples (15.00%)
  - **Test Set:** 1,886 samples (15.01%)
- **Distribution Fidelity:** Preserves class proportions with maximum deviation $\le 0.03\%$ across all subsets.
- **Data Leakage Prevention:** Data is split prior to any normalization or scaling. `StandardScaler` must be fit strictly on the training set during Stage 4 model pipelines.
- **Validation:** Verifies zero `ImageId` overlap across splits and complete record conservation ($8,797 + 1,885 + 1,886 = 12,568$).

---

## 6. Execution Instructions

### Complete Workflow (Stage 1, Stage 2, and Stage 3)
```bash
python main.py
```
*(Automatically uses existing `features/steel_defect_features.csv` when `Dataset/` images are not present).*

### Run Stage 2 Only (Feature Analysis & Data Quality)
```bash
python main.py --stage 2
# OR
python -m src.data_quality
```

### Run Stage 3 Only (Data Preparation & Stratified Splitting)
```bash
python main.py --stage 3
# OR
python -m src.data_preparation
```

### Re-Extract Features from Raw Images (Stage 1)
```bash
python main.py --stage 1 --dataset-dir Dataset --n-jobs 4
```

---

## 7. Model Training Integration (For Subsequent Stages)

Subsequent model development (Random Forest, SVM, XGBoost) can directly import the prepared splits:

```python
from src.data_preparation import load_split_data

# Loads X (15 features), Y (4 binary targets), and ImageId identifiers
(X_train, Y_train, ids_train), (X_val, Y_val, ids_val), (X_test, Y_test, ids_test) = load_split_data("data/splits")

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
# X_train: (8797, 15), Y_train: (8797, 4)
```
