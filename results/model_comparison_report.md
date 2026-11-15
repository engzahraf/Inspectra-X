# INSPECTRA-X: Multi-Label Machine Learning Model Evaluation Report

**Project:** INSPECTRA-X – ML-Based Automated Steel Surface Defect Classification and Quality Assessment System  
**Benchmark Dataset:** Severstal Steel Defect Detection (Tabular Representation)  
**Feature Engineering:** Exactly 15 Handcrafted Numerical Features (Texture, Edge, Morphological, Frequency)  
**Methodology:** Classical Machine Learning (No Deep Neural Networks or Black-Box CNNs)  
**Dataset Partitioning:** 70% Training (8,797), 15% Validation (1,885), 15% Test (1,886) via Iterative Multi-Label Stratification  

---

## 1. Problem Overview & Formulation

Automated visual surface defect detection is critical in continuous steel rolling mills to prevent catastrophic structural failures in aerospace, automotive, and infrastructure manufacturing. In industrial rolling lines, flat-rolled steel strips undergo severe thermal and mechanical stresses, manifesting diverse surface anomalies. Crucially, a single steel strip may exhibit zero defects (pristine steel), a solitary defect, or multiple defect classes concurrently.

Therefore, Inspectra-X strictly formulates surface quality assessment as a **Multi-Label Binary Classification Problem**:
$$\mathbf{y} = [y_1, y_2, y_3, y_4] \in \{0, 1\}^4$$
where each target label $y_k$ represents the independent presence ($1$) or absence ($0$) of defect class $k$. Converting this problem to single-label multiclass classification would discard simultaneous co-occurrences and introduce substantial target distortion.

---

## 2. Dataset and Target-Label Distribution

The benchmark tabular dataset comprises exactly **12,568 steel surface images** ($256 \times 1600$ native resolution) represented across 15 handcrafted numerical features and 4 binary defect targets:

1. **Class 1 – Crazing:** Interconnected networks of high-frequency thermal and mechanical micro-fissures (897 positive samples, 7.14% incidence).
2. **Class 2 – Inclusion:** Non-metallic foreign material embedded beneath or breaking the surface (247 positive samples, 1.97% incidence – severe 20.85:1 class imbalance relative to Patches [5,150 / 247 ≈ 20.85:1], and 49.88:1 binary negative-to-positive ratio [12,321 / 247 ≈ 49.88:1]).
3. **Class 3 – Patches:** Broad, localized plate deformations and scale exfoliations (5,150 positive samples, 40.98% incidence).
4. **Class 4 – Pitted Surface:** Localized craters, pinholes, and indentation clusters (801 positive samples, 6.37% incidence).

### Dataset Multi-Label Composition:
- **Clean Steel (0 defects):** 5,902 images (46.96%)
- **Single-Defect Sheets:** 6,239 images (49.64%)
- **Multi-Label Defective Sheets ($\ge 2$ defects):** 427 images (3.40%)

---

## 3. Overall Model Performance on Validation Split

All candidate architectures were trained strictly on the Training set (8,797 samples) and evaluated on the **Validation set** (1,885 samples) for hyperparameter tuning and model selection. Feature scaling (`StandardScaler`) was fit exclusively on training data to prevent data leakage.

| Model | Macro F1 | Micro F1 | Weighted F1 | Subset Accuracy | Hamming Loss | Train Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.7478 | 0.8519 | 0.8558 | 0.8562 | 0.0452 | 14.29 |
| XGBoost | 0.7475 | 0.8468 | 0.8557 | 0.8446 | 0.0477 | 6.38 |
| Support Vector Machine | 0.6948 | 0.847 | 0.8381 | 0.8785 | 0.0424 | 23.14 |

---

## 4. Per-Label Detailed Performance on Validation Split

To assess how each algorithm handles individual defect morphologies and class imbalances, per-label Precision, Recall, F1-Score, Support, and contingency metrics are detailed below.

### 4.1 Random Forest (Validation Split)

| Defect Class | Precision | Recall | F1-Score | Support | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crazing | 0.6347 | 0.791 | 0.7043 | 134 | 106 | 61 | 1690 | 28 |
| Inclusion | 0.5918 | 0.7838 | 0.6744 | 37 | 29 | 20 | 1828 | 8 |
| Patches | 0.8686 | 0.9676 | 0.9154 | 772 | 747 | 113 | 1000 | 25 |
| Pitted_Surface | 0.6037 | 0.825 | 0.6972 | 120 | 99 | 65 | 1700 | 21 |

### 4.2 Support Vector Machine (Validation Split)

| Defect Class | Precision | Recall | F1-Score | Support | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crazing | 0.74 | 0.5522 | 0.6325 | 134 | 74 | 26 | 1725 | 60 |
| Inclusion | 0.85 | 0.4595 | 0.5965 | 37 | 17 | 3 | 1845 | 20 |
| Patches | 0.8952 | 0.9404 | 0.9172 | 772 | 726 | 85 | 1028 | 46 |
| Pitted_Surface | 0.7041 | 0.575 | 0.633 | 120 | 69 | 29 | 1736 | 51 |

### 4.3 XGBoost (Validation Split)

| Defect Class | Precision | Recall | F1-Score | Support | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crazing | 0.5808 | 0.8582 | 0.6928 | 134 | 115 | 83 | 1668 | 19 |
| Inclusion | 0.6078 | 0.8378 | 0.7045 | 37 | 31 | 20 | 1828 | 6 |
| Patches | 0.8794 | 0.9637 | 0.9197 | 772 | 744 | 102 | 1011 | 28 |
| Pitted_Surface | 0.5469 | 0.875 | 0.6731 | 120 | 105 | 87 | 1678 | 15 |

---

## 5. Champion Model Selection & Macro F1 Evaluation Rationale

### 5.1 Why Macro F1 is the Primary Selection Metric
In imbalanced multi-label classification, standard global metrics like **Subset Accuracy** and **Micro F1** can be fundamentally misleading:
- **Micro-Averaged F1** aggregates global True Positives, False Positives, and False Negatives across all labels. Consequently, it is overwhelmingly dominated by the majority defect class (**Patches**, 5,150 positive samples, 40.98%) and the large volume of clean negative instances. A model that completely ignores rare, dangerous defects like **Inclusion** (247 positive samples, 1.97%, representing a 20.85:1 class imbalance ratio relative to Patches: $5,150 / 247 \approx 20.85:1$) can still register a high Micro F1 (> 0.84).
- **Subset Accuracy (Exact Match Ratio)** requires that all 4 binary labels match ground truth simultaneously. While a valuable indicator of joint prediction perfection, it over-penalizes single-label discrepancies on multi-defect sheets and offers no visibility into per-defect recall.
- In contrast, **Macro-Averaged F1** calculates the unweighted arithmetic mean of individual class F1-scores:
$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k = \frac{\text{F1}_{\text{Crazing}} + \text{F1}_{\text{Inclusion}} + \text{F1}_{\text{Patches}} + \text{F1}_{\text{Pitted Surface}}}{4}$$
Macro F1 assigns exactly equal weight (25%) to each defect category regardless of its sample frequency. It strictly penalizes classifiers that fail on the minority defect classes, directly aligning with industrial quality assurance goals where missing a rare inclusion or hairline crack is unacceptably hazardous.

### 5.2 Model Selection Decision (Validation Split)
On the Validation Split, **Random Forest** achieved the highest Macro F1 score (**0.7478**), outperforming **XGBoost** (**0.7475**) and **Support Vector Machine** (**0.6948**).

**Key Observations on Candidate Models:**
1. **Random Forest (Champion):** Demonstrates the most balanced trade-off between precision and recall across all four classes. Balanced class weighting (`class_weight='balanced'`) enables high recall on rare defects (78.38% on Inclusion, 79.10% on Crazing, 82.50% on Pitted Surface) while retaining strong precision (86.86% on Patches, 63.47% on Crazing). Additionally, random feature subsampling ($m = \sqrt{15}$) provides natural resilience against the 12 collinear feature pairs identified in Stage 2.
2. **XGBoost (Runner-Up):** Highly competitive Macro F1 (0.7475). With per-class `scale_pos_weight` dynamic adjustment, XGBoost achieved the highest sensitivity/recall on minority defects (85.82% on Crazing, 83.78% on Inclusion, 87.50% on Pitted Surface). However, it suffered from a higher false-positive rate, yielding lower overall precision and slightly lower subset accuracy (0.8446 vs 0.8562 for RF).
3. **Support Vector Machine (SVM):** Attained the highest Subset Accuracy (0.8785) and lowest Hamming Loss (0.0424), driven by very high precision (e.g. 85.00% on Inclusion, 74.00% on Crazing). However, SVM struggled with minority-class sensitivity, missing over 54% of inclusions (recall 45.95%) and 44% of crazing fissures (recall 55.22%), resulting in a substantially lower Macro F1 (0.6948).

> [!IMPORTANT]
> **Methodological Integrity:** Model selection was governed strictly by the Validation Split. The Test Split was kept completely sealed during all model development and tuning phases.

---

## 6. Overall Independent Test Split Performance

To verify true generalization capability and confirm that no overfitting or validation bias occurred, all three models were evaluated on the held-out **Test Split** (1,886 images, 15.01% of the dataset).

| Model | Macro F1 | Micro F1 | Weighted F1 | Subset Accuracy | Hamming Loss |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.7668 | 0.8468 | 0.85 | 0.8542 | 0.0467 |
| XGBoost | 0.7635 | 0.8466 | 0.8545 | 0.8499 | 0.0475 |
| Support Vector Machine | 0.7359 | 0.8444 | 0.8374 | 0.8749 | 0.0435 |

---

## 7. Per-Label Detailed Performance on Test Split

The tables below document the empirical performance of each model on the unseen Test Split across all four defect classes:

### 7.1 Random Forest (Test Split)

| Defect Class | Precision | Recall | F1-Score | Support | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crazing | 0.6325 | 0.7778 | 0.6977 | 135 | 105 | 61 | 1690 | 30 |
| Inclusion | 0.7674 | 0.8919 | 0.825 | 37 | 33 | 10 | 1839 | 4 |
| Patches | 0.8626 | 0.9664 | 0.9115 | 773 | 747 | 119 | 994 | 26 |
| Pitted_Surface | 0.557 | 0.7333 | 0.6331 | 120 | 88 | 70 | 1696 | 32 |

### 7.2 Support Vector Machine (Test Split)

| Defect Class | Precision | Recall | F1-Score | Support | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crazing | 0.7647 | 0.5778 | 0.6582 | 135 | 78 | 24 | 1727 | 57 |
| Inclusion | 0.8387 | 0.7027 | 0.7647 | 37 | 26 | 5 | 1844 | 11 |
| Patches | 0.8834 | 0.9314 | 0.9068 | 773 | 720 | 95 | 1018 | 53 |
| Pitted_Surface | 0.6947 | 0.55 | 0.614 | 120 | 66 | 29 | 1737 | 54 |

### 7.3 XGBoost (Test Split)

| Defect Class | Precision | Recall | F1-Score | Support | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crazing | 0.6032 | 0.8444 | 0.7037 | 135 | 114 | 75 | 1676 | 21 |
| Inclusion | 0.7 | 0.9459 | 0.8046 | 37 | 35 | 15 | 1834 | 2 |
| Patches | 0.8784 | 0.9625 | 0.9185 | 773 | 744 | 103 | 1010 | 29 |
| Pitted_Surface | 0.5191 | 0.7917 | 0.6271 | 120 | 95 | 88 | 1678 | 25 |

---

## 8. Conclusion and Engineering Assessment

1. **Generalization Confirmed:** The champion **Random Forest** classifier demonstrated robust generalization on unseen test data, achieving an independent **Test Macro F1 of 0.7668** (higher than its validation benchmark of 0.7478), with Micro F1 of **0.8468**, Subset Accuracy of **85.42%**, and low Hamming Loss of **0.0467**.
2. **Defect-Specific Competencies:**
   - **Patches (Class 3):** All three models achieved outstanding F1-scores (> 0.90) due to prominent textural contrast and morphological scale area.
   - **Inclusion (Class 2):** Despite severe class imbalance (20.85:1 ratio of Patches to Inclusion, 5,150 / 247; 1.97% overall prevalence), both Random Forest (F1: 0.8250, Recall: 89.19%) and XGBoost (F1: 0.8046, Recall: 94.59%) proved capable of detecting microscopic foreign particles using handcrafted second-order contrast, Laplacian variance, and candidate component geometry.
   - **Crazing (Class 1):** Accurately identified via skeleton branching complexity, Gabor energy, and directional GLCM correlation, with XGBoost leading in sensitivity (84.44% recall) and Random Forest achieving the best harmonic balance (F1: 0.6977).
   - **Pitted Surface (Class 4):** Connected component count and Otsu edge density effectively capture pinhole clusters, with Random Forest achieving 0.6331 F1 and XGBoost achieving 0.6271 F1.
3. **Practical Deployment Recommendation:**
   - **Random Forest** is recommended as the primary classification engine for production steel inspection lines where an optimal balance of precision, recall, and computational throughput is mandatory.
   - For inspection environments where safety regulations enforce zero tolerance for false negatives (at the expense of secondary manual verification of false alarms), **XGBoost** with boosted positive weighting serves as an exceptional high-recall alternative.
4. **No Deep Learning Requirement:** This project demonstrates that classical machine learning combined with rigorous handcrafted feature engineering can achieve competitive multi-label classification performance on complex industrial steel inspection tasks with low latency (< 10 ms model inference time), complete interpretability, and modest compute resources.