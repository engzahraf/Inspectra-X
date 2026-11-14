# Steel Defect Classification Using 15 Handcrafted Image Features

An academic image-processing and feature-engineering system developed for the **Severstal Steel Defect Detection** dataset. This pipeline extracts **exactly 15 handcrafted numerical features** across texture, edge, morphological, and transform domains without utilizing deep neural networks.

---

## 1. Project Directory Structure

```text
SteelDefectML/
│
├── Dataset/                            # Raw dataset
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
│   └── feature_extraction.py           # Unified extraction pipeline & parallel workers
│
├── features/
│   ├── reference_defect_free.png       # Reproducible defect-free reference template
│   └── steel_defect_features.csv       # Extracted 15 features + 4 multi-label target columns
│
├── results/
│   ├── summary_statistics.csv          # Mean, std, quartiles, skewness per feature
│   ├── feature_distributions.png       # 3x5 grid histogram visualization
│   └── correlation_heatmap.png         # Pairwise Pearson correlation matrix
│
├── requirements.txt                    # Project dependencies
├── main.py                             # Command-line entry point and validator
└── README.md
```

---

## 2. Exactly 15 Handcrafted Features

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

## 3. Usage Instructions

### Quick Test / Sample Extraction
```bash
python main.py --dataset-dir Dataset --sample-size 100 --n-jobs 4
```

### Full Dataset Extraction (All 12,568 images)
```bash
python main.py --dataset-dir Dataset --n-jobs 4
```
*(Extraction runs in parallel with real-time throughput metrics).*

