"""
Group 1: Texture Feature Extraction Module (Features 1 - 5).

Implements 5 classical texture descriptors:
  1. GLCM Contrast
  2. GLCM Correlation
  3. GLCM Energy
  4. LBP Histogram Entropy
  5. Gabor Filter Response Energy

All features are extracted from the preprocessed grayscale steel surface image.
"""

from typing import List, Optional
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def calculate_glcm_contrast(
    image: np.ndarray,
    distances: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
    levels: int = 64
) -> float:
    """
    Feature 1: Gray-Level Co-occurrence Matrix (GLCM) Contrast.

    What it measures:
        Measures the intensity contrast between a pixel and its neighbor over the whole image.
        Mathematically, it computes the second moment of difference between gray levels:
            Contrast = sum_{i,j} |i - j|^2 * P(i, j)

    Calculation Method:
        1. Quantizes the 8-bit input image to `levels` (default 64 bins, i.e., image // 4)
           to ensure statistically robust co-occurrence accumulation.
        2. Computes the spatial co-occurrence matrix P(i, j) using distances=[1] and
           four cardinal/diagonal orientations [0, pi/4, pi/2, 3*pi/4].
        3. Computes the contrast property for each direction via skimage.feature.graycoprops.
        4. Returns the mean contrast averaged across all 4 directions.

    Relevance to Steel Surface Defect Analysis:
        Defect-free steel surfaces have relatively smooth transitions between adjacent pixels.
        Surface defects such as crazing (sharp cracks), inclusion pits, and abrasions introduce
        abrupt local intensity gradients and deep shadows, resulting in substantially elevated
        GLCM contrast values.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        distances: List of pixel separation distances. Defaults to [1].
        angles: List of angles in radians. Defaults to [0, pi/4, pi/2, 3*pi/4].
        levels: Number of quantized gray levels. Defaults to 64.

    Returns:
        float: Mean GLCM contrast across evaluated orientations.
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    try:
        # Quantize image to reduce GLCM dimensionality from 256x256 to 64x64
        quantized = (image // (256 // levels)).astype(np.uint8)
        glcm = graycomatrix(
            quantized,
            distances=distances,
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )
        contrast = graycoprops(glcm, prop='contrast')
        return float(np.mean(contrast))
    except Exception:
        return 0.0


def calculate_glcm_correlation(
    image: np.ndarray,
    distances: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
    levels: int = 64
) -> float:
    """
    Feature 2: Gray-Level Co-occurrence Matrix (GLCM) Correlation.

    What it measures:
        Measures how correlated a pixel is to its neighbor across the whole image:
            Correlation = sum_{i,j} [(i - mu_i) * (j - mu_j) * P(i, j)] / (sigma_i * sigma_j)
        Values range between -1.0 and 1.0 (with 1.0 indicating perfect linear dependency).

    Calculation Method:
        1. Quantizes the 8-bit input image to `levels` (default 64 bins).
        2. Computes the normalized co-occurrence matrix across [0, pi/4, pi/2, 3*pi/4] at distance 1.
        3. Computes the correlation property for each direction via graycoprops.
        4. Handles edge cases (such as zero-variance uniform patches where denominator is 0)
           by falling back safely to 1.0.
        5. Returns the mean correlation across orientations.

    Relevance to Steel Surface Defect Analysis:
        Cold-rolled and hot-rolled steel plates possess strong linear texture patterns along the
        rolling direction. Pristine steel sheets display high directional correlation. When
        scattered inclusions, pitting, or localized patch defects disrupt the manufacturing
        grain, spatial correlation diminishes.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        distances: List of pixel separation distances. Defaults to [1].
        angles: List of angles in radians. Defaults to [0, pi/4, pi/2, 3*pi/4].
        levels: Number of quantized gray levels. Defaults to 64.

    Returns:
        float: Mean GLCM correlation across evaluated orientations.
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    try:
        quantized = (image // (256 // levels)).astype(np.uint8)
        glcm = graycomatrix(
            quantized,
            distances=distances,
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )
        corr = graycoprops(glcm, prop='correlation')
        val = float(np.mean(corr))
        if np.isnan(val) or np.isinf(val):
            return 1.0
        return val
    except Exception:
        return 1.0


def calculate_glcm_energy(
    image: np.ndarray,
    distances: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
    levels: int = 64
) -> float:
    """
    Feature 3: Gray-Level Co-occurrence Matrix (GLCM) Energy.

    What it measures:
        Measures the textural uniformity or Angular Second Moment (ASM):
            Energy = sqrt(sum_{i,j} P(i, j)^2)
        Reaches its maximum of 1.0 when the image is constant/completely homogeneous.

    Calculation Method:
        1. Quantizes the 8-bit input image to `levels` (default 64 bins).
        2. Computes normalized GLCM across [0, pi/4, pi/2, 3*pi/4] at distance 1.
        3. Computes the energy property for each direction via graycoprops.
        4. Returns the mean energy across evaluated directions.

    Relevance to Steel Surface Defect Analysis:
        Clean, defect-free industrial steel surfaces exhibit high spatial uniformity and minimal
        gray-level transitions, yielding elevated energy values. Surface defects (scabs, crazing,
        patches) disperse probability mass across disparate GLCM bins, sharply reducing energy.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        distances: List of pixel separation distances. Defaults to [1].
        angles: List of angles in radians. Defaults to [0, pi/4, pi/2, 3*pi/4].
        levels: Number of quantized gray levels. Defaults to 64.

    Returns:
        float: Mean GLCM energy across evaluated orientations.
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    try:
        quantized = (image // (256 // levels)).astype(np.uint8)
        glcm = graycomatrix(
            quantized,
            distances=distances,
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )
        energy = graycoprops(glcm, prop='energy')
        val = float(np.mean(energy))
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    except Exception:
        return 0.0


def calculate_lbp_entropy(
    image: np.ndarray,
    P: int = 8,
    R: int = 1
) -> float:
    """
    Feature 4: Local Binary Pattern (LBP) Histogram Shannon Entropy.

    What it measures:
        Measures the micro-textural complexity and unpredictability of local binary pattern transitions.
            LBP_Entropy = - sum_{k} p_k * log2(p_k + eps)
        where p_k is the normalized probability of the k-th uniform LBP pattern code.

    Calculation Method:
        1. Evaluates rotation-invariant uniform Local Binary Patterns on the image with radius R=1
           and P=8 sampling points using skimage.feature.local_binary_pattern(method='uniform').
        2. In uniform LBP with P=8, there are P + 2 = 10 distinct pattern bins.
        3. Computes the normalized histogram (probability distribution) across the 10 bins.
        4. Calculates the Shannon entropy of this distribution in bits.

    Relevance to Steel Surface Defect Analysis:
        Uniform steel rolling patterns produce highly predictable micro-structures dominated by flat
        and single-edge LBP patterns (low entropy). Complex defect topologies, particularly crazing
        (fine network of chaotic micro-fissures) and pitting, generate a wide spectrum of complex
        micro-textures, driving LBP entropy significantly higher.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        P: Number of circularly symmetric neighbor set points. Default: 8.
        R: Radius of circle. Default: 1.

    Returns:
        float: Shannon entropy of the LBP pattern histogram.
    """
    try:
        lbp = local_binary_pattern(image, P=P, R=R, method='uniform')
        n_bins = int(P + 2)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        # Shannon entropy
        prob = hist[hist > 0]
        if len(prob) == 0:
            return 0.0
        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)
    except Exception:
        return 0.0


def calculate_gabor_energy(
    image: np.ndarray,
    frequencies: Optional[List[float]] = None,
    thetas: Optional[List[float]] = None
) -> float:
    """
    Feature 5: Gabor Filter Response Energy.

    What it measures:
        Measures the mean squared energy response of multi-directional Gabor wavelets tuned
        to specific spatial frequencies:
            Gabor_Energy = (1 / |Theta|) * sum_{theta} [ (1 / HW) * sum_{x,y} (I * g_{lambda, theta})^2 ]

    Calculation Method:
        1. Constructs a Gabor filter bank covering 4 orientations [0, pi/4, pi/2, 3*pi/4]
           using cv2.getGaborKernel with wavelength lambda = 4.0, bandwidth sigma = 2.0,
           gamma = 0.5 (aspect ratio), and psi = 0 (phase offset).
        2. Filters the normalized float image with each directional kernel.
        3. Computes the mean squared response (energy) per filter.
        4. Averages the energy across all orientations to provide an omnidirectional measure.

    Relevance to Steel Surface Defect Analysis:
        Steel rolling machinery imprints subtle, unidirectional grain lines. Gabor wavelets act as
        localized spatial frequency analyzers. Rolling defects such as crazing cracks, deep scratches,
        and foreign inclusions create sharp directional energy spikes that stand out strongly against
        the background steel substrate.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        frequencies: Optional spatial frequencies. Default uses wavelength lambda=4.0.
        thetas: List of filter orientations in radians. Default: [0, pi/4, pi/2, 3*pi/4].

    Returns:
        float: Mean Gabor filter response energy.
    """
    if thetas is None:
        thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    try:
        # Convert to float64 in [0.0, 1.0] for linear filtering
        img_float = image.astype(np.float64) / 255.0
        ksize = 15
        sigma = 2.0
        lambd = 4.0
        gamma = 0.5
        psi = 0.0

        energies = []
        for theta in thetas:
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=sigma,
                theta=theta,
                lambd=lambd,
                gamma=gamma,
                psi=psi,
                ktype=cv2.CV_64F
            )
            filtered = cv2.filter2D(img_float, cv2.CV_64F, kernel)
            energy = np.mean(filtered ** 2)
            energies.append(energy)

        return float(np.mean(energies))
    except Exception:
        return 0.0

