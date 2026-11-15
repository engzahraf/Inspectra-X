# Inspectra-X: Stage 2 Feature Analysis and Data Quality Report

**Project:** ML-Based Automated Steel Surface Defect Classification and Quality Assessment System  
**Dataset:** Severstal Steel Defect Detection (Classical Machine Learning Tabular Features)  
**Total Records:** 12,568 images | **Total Columns:** 20  

---

## 1. Executive Summary

This report provides an academic data quality assessment and statistical characterization of the 15 handcrafted numerical features extracted across the Severstal Steel Defect Detection dataset. The dataset comprises exactly 12,568 steel surface images represented as a multi-label classification problem across four target classes: **Crazing (Class 1)**, **Inclusion (Class 2)**, **Patches (Class 3)**, and **Pitted Surface (Class 4)**.

All structural integrity checks passed with zero missing, null, or infinite values. Zero-variance features were verified absent. The target classes exhibit severe class imbalance and multi-label co-occurrence patterns, which strictly mandates iterative multi-label stratification for dataset partitioning.

---

## 2. Dataset Structural Integrity Verification

| Verification Check | Target / Specification | Actual Observed | Validation Status |
| :--- | :--- | :--- | :--- |
| **Record Count** | 12,568 images | 12,568 images | PASS |
| **Column Count** | 20 columns (1 ID + 15 features + 4 targets) | 20 columns | PASS |
| **Unique Identifiers** | 0 duplicate ImageIds | 0 duplicates | PASS |
| **Missing Values** | 0 NaN / Null cells | 0 nulls | PASS |
| **Infinite Values** | 0 (+/- Inf) values | 0 inf values | PASS |
| **Binary Target Labels** | Strictly (0, 1) | Binary discrete verified | PASS |
| **Numeric Features** | 15 continuous / discrete features | All float64 | PASS |

---

## 3. Statistical Distribution and Outlier Analysis (IQR Method)

Outlier identification was performed using the standard interquartile range (IQR) rule:
$$\text{IQR} = Q_3 - Q_1, \quad \text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

| # | Feature Name | Mean | Std | Median | IQR | Skewness | Outlier Count | Outlier % | Zero Var |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `glcm_contrast` | 32.7455 | 21.2954 | 29.0959 | 23.8991 | +1.414 | 515 | 4.10% | NO |
| 2 | `glcm_correlation` | 0.8219 | 0.1181 | 0.8333 | 0.1764 | -0.719 | 103 | 0.82% | NO |
| 3 | `glcm_energy` | 0.1215 | 0.1410 | 0.0577 | 0.0812 | +2.243 | 1,871 | 14.89% | NO |
| 4 | `lbp_entropy` | 3.0845 | 0.3865 | 3.2310 | 0.1345 | -3.139 | 1,741 | 13.85% | NO |
| 5 | `gabor_energy` | 0.1543 | 0.0981 | 0.1366 | 0.1076 | +1.683 | 476 | 3.79% | NO |
| 6 | `canny_edge_density` | 0.1374 | 0.0717 | 0.1356 | 0.1054 | +0.116 | 1 | 0.01% | NO |
| 7 | `sobel_magnitude_mean` | 85.1213 | 34.5974 | 85.1119 | 41.5715 | +0.206 | 138 | 1.10% | NO |
| 8 | `laplacian_variance` | 21814.3188 | 14255.3888 | 19438.8417 | 16062.9456 | +1.459 | 516 | 4.11% | NO |
| 9 | `connected_component_count` | 1.5844 | 2.3018 | 1.0000 | 2.0000 | +2.255 | 873 | 6.95% | NO |
| 10 | `largest_component_solidity` | 0.4734 | 0.4536 | 0.6483 | 0.9519 | -0.033 | 0 | 0.00% | NO |
| 11 | `defect_aspect_ratio` | 2.7753 | 4.1798 | 1.1250 | 3.8791 | +2.316 | 1,076 | 8.56% | NO |
| 12 | `skeleton_branching_complexity` | 0.0083 | 0.0183 | 0.0000 | 0.0081 | +3.502 | 1,781 | 14.17% | NO |
| 13 | `image_entropy` | 6.6883 | 1.0132 | 6.9601 | 0.6011 | -2.716 | 1,291 | 10.27% | NO |
| 14 | `ssim_to_reference` | 0.0573 | 0.0164 | 0.0590 | 0.0220 | +0.866 | 26 | 0.21% | NO |
| 15 | `fourier_peak_frequency` | 0.0095 | 0.0104 | 0.0063 | 0.0038 | +6.199 | 1,311 | 10.43% | NO |

> [!IMPORTANT]
> **Academic Justification on Outlier Preservation:**
> Outliers are **NOT** measurement errors or sensor noise. In steel surface inspection, pristine clean steel sheets generate baseline zero or low responses in morphological features (e.g. `skeleton_branching_complexity`, `connected_component_count`, `defect_aspect_ratio`). Images containing severe defect clusters (such as dense pitting networks or dendritic crazing cracks) naturally register as statistical outliers under the 1.5*IQR rule. Removing these values would discard genuine defect signatures and cripple the classification models. Therefore, all detected outliers are preserved for model training.

---

## 4. Collinearity and Feature Redundancy Analysis

Pairwise Pearson correlation analysis ($|r| \ge 0.70$) reveals significant collinearity among specific feature pairs:

| Feature 1 | Feature 2 | Pearson $r$ | Dependency Classification | Domain Interpretation |
| :--- | :--- | :---: | :--- | :--- |
| `glcm_contrast` | `laplacian_variance` | +0.9941 | **Severe Collinearity** | Both measure second-order spatial intensity variation and surface roughness. |
| `glcm_contrast` | `gabor_energy` | +0.9573 | **Severe Collinearity** | Both capture high-frequency textural variance across directional sub-bands. |
| `glcm_energy` | `lbp_entropy` | -0.9552 | **Severe Collinearity** | Inverse relationship: high textural uniformity (energy) yields low LBP entropy. |
| `lbp_entropy` | `image_entropy` | +0.9469 | **Severe Collinearity** | Micro-texture entropy and global intensity entropy scale synchronously. |
| `gabor_energy` | `laplacian_variance` | +0.9440 | **Severe Collinearity** | Frequency domain response energy correlates with spatial Laplacian variance. |
| `gabor_energy` | `sobel_magnitude_mean` | +0.9372 | **Severe Collinearity** | Gabor multi-scale filter responses mirror first-order Sobel gradient magnitudes. |
| `glcm_contrast` | `sobel_magnitude_mean` | +0.9357 | **Severe Collinearity** | Gradient edge strength directly translates into spatial GLCM contrast. |
| `glcm_energy` | `image_entropy` | -0.9185 | **Severe Collinearity** | Inverse relationship: uniform pristine sheets possess low Shannon entropy. |
| `sobel_magnitude_mean` | `laplacian_variance` | +0.9177 | **Severe Collinearity** | First-order Sobel and second-order Laplacian gradients describe surface edge density. |
| `sobel_magnitude_mean` | `image_entropy` | +0.7454 | Strong Correlation | Intensity dispersion increases with overall edge and gradient activity. |
| `glcm_contrast` | `canny_edge_density` | +0.7280 | Strong Correlation | Binary edge density scales monotonically with continuous gradient magnitudes. |
| `canny_edge_density` | `laplacian_variance` | +0.7200 | Strong Correlation | Binary edge density scales monotonically with continuous gradient magnitudes. |

### Redundancy Clusters Identified:
1. **Surface Gradient & Roughness Cluster:** `{glcm_contrast, laplacian_variance, sobel_magnitude_mean, gabor_energy}` ($r > 0.91$). These features all quantify surface roughness and edge sharpness.
2. **Information Richness vs. Uniformity Cluster:** `{glcm_energy, lbp_entropy, image_entropy}` ($|r| > 0.91$). `glcm_energy` measures pristine steel uniformity, which negatively correlates with Shannon and LBP entropies.

> [!NOTE]
> **Preservation Decision:** In accordance with university project guidelines, no features are eliminated at Stage 2. Tree-based ensembles (Random Forest, XGBoost) are inherently robust to multicollinearity via feature subsampling, while regularized models (e.g. SVM with L2 penalty) handle correlated inputs gracefully.

---

## 5. Multi-Label Target Distribution and Co-Occurrence Analysis

### Target Class Imbalance Summary

| Defect Class | Severstal Code | Positive Count | Positive % | Negative Count | Negative % | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crazing** | Class 1 | 897 | 7.14% | 11,671 | 92.86% | 13.0:1 |
| **Inclusion** | Class 2 | 247 | 1.97% | 12,321 | 98.03% | 49.9:1 |
| **Patches** | Class 3 | 5,150 | 40.98% | 7,418 | 59.02% | 1.4:1 |
| **Pitted_Surface** | Class 4 | 801 | 6.37% | 11,767 | 93.63% | 14.7:1 |

### Simultaneous Defect Distribution per Image

| Defect Count per Image | Category | Image Count | Percentage | Cumulative % |
| :---: | :--- | :---: | :---: | :---: |
| 0 | Clean Steel (Defect-Free) | 5,902 | 46.96% | 46.96% |
| 1 | Single-Label Image | 6,239 | 49.64% | 96.60% |
| 2 | Multi-Label Image | 425 | 3.38% | 99.98% |
| 3 | Multi-Label Image | 2 | 0.02% | 100.00% |
| 4 | Multi-Label Image | 0 | 0.00% | 100.00% |

- **Defect-Free Images (Clean Sheets):** 5,902 (46.96%)
- **Single Defect Images:** 6,239 (49.64%)
- **Multi-Label Images ($\ge 2$ simultaneous defects):** 427 (3.40%)

### Defect Co-Occurrence Matrix ($Y^T Y$)

The co-occurrence matrix quantifies joint defect occurrences across all 12,568 images (diagonal represents total single-class occurrences):

| Defect Class | Crazing | Inclusion | Patches | Pitted_Surface |
| :--- | :---: | :---: | :---: | :---: |
| **Crazing** | 897 | 37 | 93 | 0 |
| **Inclusion** | 37 | 247 | 16 | 1 |
| **Patches** | 93 | 16 | 5,150 | 284 |
| **Pitted_Surface** | 0 | 1 | 284 | 801 |

**Key Co-Occurrence Observations:**
- **Patches + Pitted Surface** exhibits the highest co-occurrence frequency (284 images), indicating localized pitting frequently manifests within larger patched surface anomalies.
- **Crazing + Patches** co-occur in 93 images.
- **Crazing + Inclusion** co-occur in 37 images.
- **Crazing + Pitted Surface** never co-occur (0 images), indicating distinct physical metallurgical formation mechanisms.

---

## 6. Stage 3 & Stage 4 Methodological Recommendations

1. **Iterative Multi-Label Stratification:** Standard single-label stratified splitting (`StratifiedKFold`) is strictly invalid because it ignores multi-label combinations. Splitting must employ `MultilabelStratifiedShuffleSplit` or `MultilabelStratifiedKFold` to maintain identical class ratios across Train (70%), Validation (15%), and Test (15%) subsets.
2. **Strict Leakage Prevention:** Feature transformations (e.g. `StandardScaler`) must be fitted exclusively on the 70% training split. Applying global scaling prior to partitioning would introduce data leakage from validation and test distributions.
3. **Identifier Isolation:** `ImageId` must be retained as a tracking key in output CSVs, but excluded from predictor matrix $X$.
4. **Model Architecture Compatibility:** High feature collinearity must be addressed during Stage 4 model exploration: tree ensembles (Random Forest, XGBoost) will naturally partition feature spaces, while SVM will require kernel tuning and appropriate regularization.
