"""
Group 4: Statistical and Frequency Feature Extraction Module (Features 13 - 15).

Implements 3 global statistical and transform-domain descriptors:
  13. Image Entropy
  14. SSIM to Reference
  15. Fourier Spectrum Peak Frequency

Features 13 and 15 are extracted from the preprocessed grayscale image.
Feature 14 is computed against a reproducible, defect-free reference template.
"""

from typing import Optional
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_image_entropy(image: np.ndarray) -> float:
    """
    Feature 13: Image Shannon Entropy.

    What it measures:
        Measures the overall information content, disorder, and grayscale unpredictability:
            Entropy = - sum_{i=0}^{255} p_i * log2(p_i + eps)
        where p_i is the empirical probability of occurrence of intensity level i.

    Calculation Method:
        1. Computes the 256-bin histogram of the preprocessed 8-bit image.
        2. Normalizes the histogram by total pixel count (H * W) to obtain the probability
           density function p_i.
        3. Identifies non-zero probabilities (p_i > 0) to avoid log(0).
        4. Calculates the Shannon entropy in bits per pixel.

    Relevance to Steel Surface Defect Analysis:
        Defect-free steel possesses a narrow, unimodal intensity histogram centered around the
        nominal reflectance of the alloy, yielding low entropy. The presence of oxidized patches,
        scabs, or inclusions broadens the gray-level distribution, introducing multimodal peaks
        and significantly higher global entropy.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.

    Returns:
        float: Shannon entropy in bits (typically in [0.0, 8.0]).
    """
    try:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        total_pixels = image.size
        p = hist / total_pixels
        non_zero_p = p[p > 0]

        if len(non_zero_p) == 0:
            return 0.0

        entropy = -np.sum(non_zero_p * np.log2(non_zero_p))
        return float(entropy)
    except Exception:
        return 0.0


def calculate_ssim_to_reference(
    image: np.ndarray,
    reference_image: np.ndarray
) -> float:
    """
    Feature 14: Structural Similarity Index (SSIM) to Reference Template.

    What it measures:
        Measures the structural, luminance, and contrast similarity between the candidate
        steel surface image and a canonical, reproducible defect-free steel template:
            SSIM(x, y) = [ (2*mu_x*mu_y + C1)*(2*sigma_xy + C2) ] / [ (mu_x^2 + mu_y^2 + C1)*(sigma_x^2 + sigma_y^2 + C2) ]
        Values range in [-1.0, 1.0], with 1.0 indicating identical surface appearance.

    Calculation Method:
        1. Verifies that candidate image and reference template have identical dimensions (256x1600).
        2. Uses skimage.metrics.structural_similarity with data_range=255.
        3. Returns the mean SSIM index over the full surface area.

    Relevance to Steel Surface Defect Analysis:
        Rather than comparing raw pixel differences (which are sensitive to lighting shifts),
        SSIM specifically isolates structural deviations from the ideal manufacturing baseline.
        Images with large patches, rolling tears, or pitting exhibit marked drops in SSIM,
        providing a direct indicator of degradation relative to pristine steel.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        reference_image: Preprocessed 2D grayscale uint8 reference template of identical shape.

    Returns:
        float: Mean SSIM index in [-1.0, 1.0].
    """
    try:
        if reference_image is None or image is None:
            return 0.0

        if image.shape != reference_image.shape:
            # Resize safely if dimensions mismatch
            ref_aligned = cv2.resize(
                reference_image,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        else:
            ref_aligned = reference_image

        val, _ = ssim(
            image,
            ref_aligned,
            data_range=255,
            full=True
        )
        return float(val)
    except Exception:
        return 0.0


def calculate_fourier_peak_frequency(
    image: np.ndarray,
    dc_radius: int = 3
) -> float:
    """
    Feature 15: 2D Fourier Spectrum Peak Spatial Frequency.

    What it measures:
        Measures the dominant non-DC spatial frequency in the image's 2D Fourier power spectrum:
            f_peak = sqrt( ((u* - u0) / (H/2))^2 + ((v* - v0) / (W/2))^2 )
        where (u*, v*) is the coordinate of the maximum spectral energy outside the central DC region.

    Calculation Method:
        1. Converts image to float64 and computes the 2D Fast Fourier Transform via np.fft.fft2.
        2. Centers the zero-frequency DC component at (u0, v0) = (H/2, W/2) via np.fft.fftshift.
        3. Computes the magnitude spectrum: |F(u, v)|.
        4. Suppresses the DC component and immediate low-frequency illumination bias within a
           circular mask of radius `dc_radius` (default 3 pixels) around the center.
        5. Identifies the peak spectral coordinate (u*, v*) having maximum residual magnitude.
        6. Computes the normalized radial spatial frequency in [0.0, 1.0] relative to the Nyquist limit.

    Relevance to Steel Surface Defect Analysis:
        Cold rolling operations generate regular, periodic mechanical striations along the sheet.
        Defects introduce high-energy anomalous frequency components: repetitive crazing crackles
        and pitting clusters shift dominant energy toward specific mid-to-high spatial frequency bands,
        differentiating them from broad, low-frequency background rolling marks.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        dc_radius: Radius in pixels around center DC frequency to mask out. Default: 3.

    Returns:
        float: Normalized dominant non-DC radial spatial frequency in [0.0, 1.0].
    """
    try:
        img_f = image.astype(np.float64)
        h, w = img_f.shape

        # 2D Fast Fourier Transform
        f = np.fft.fft2(img_f)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # Center coordinates
        cy, cx = h // 2, w // 2

        # Mask out central DC component
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        magnitude_spectrum[dist_from_center <= dc_radius] = 0.0

        # Find maximum non-DC peak
        peak_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        peak_y, peak_x = peak_idx

        # Calculate normalized radial frequency relative to Nyquist limit
        norm_y = (peak_y - cy) / (h / 2.0)
        norm_x = (peak_x - cx) / (w / 2.0)
        radial_freq = float(np.sqrt(norm_y ** 2 + norm_x ** 2))

        return min(1.0, max(0.0, radial_freq))
    except Exception:
        return 0.0

