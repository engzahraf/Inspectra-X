"""
Group 2: Edge Feature Extraction Module (Features 6 - 8).

Implements 3 classical edge and boundary descriptors:
  6. Canny Edge Density
  7. Sobel Edge Magnitude (Mean)
  8. Laplacian Edge Variance

All features are extracted from the preprocessed grayscale steel surface image.
"""

import cv2
import numpy as np


def calculate_canny_edge_density(
    image: np.ndarray,
    sigma: float = 1.0,
    low_ratio: float = 0.5
) -> float:
    """
    Feature 6: Canny Edge Density.

    What it measures:
        Measures the spatial density of sharp structural boundaries in the image:
            Edge_Density = (Total Canny Edge Pixels) / (Total Image Pixels)
        The returned value represents the fraction of the steel sheet containing prominent edges.

    Calculation Method:
        1. Slightly smooths the preprocessed image using a Gaussian blur (sigma=1.0) to eliminate
           fine sensor noise.
        2. Computes Otsu's optimal global threshold T_otsu on the smoothed image.
        3. Defines dynamic hysteresis thresholds: high_threshold = T_otsu,
           low_threshold = low_ratio * T_otsu (default 0.5 * T_otsu).
        4. Evaluates cv2.Canny to obtain the single-pixel-wide binary edge map.
        5. Computes the ratio of non-zero edge pixels to the total pixel count (H * W).

    Relevance to Steel Surface Defect Analysis:
        Cold-formed steel strip substrates are largely devoid of abrupt contours. Defect classes
        such as crazing (dense networks of fractured fissures) and pitted surfaces (numerous sharp
        crater peripheries) produce thousands of localized edge segments, resulting in an order-of-magnitude
        higher edge density than smooth defect-free steel sheets.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        sigma: Standard deviation for Gaussian smoothing prior to edge detection. Default: 1.0.
        low_ratio: Multiplier for low hysteresis threshold relative to Otsu threshold. Default: 0.5.

    Returns:
        float: Edge pixel density in the range [0.0, 1.0].
    """
    try:
        # Gaussian blur to suppress camera thermal noise
        ksize = int(2 * round(3 * sigma) + 1)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        # Compute dynamic threshold via Otsu's method
        high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = high_thresh * low_ratio

        # Clamp thresholds safely
        high_thresh = max(10.0, min(250.0, high_thresh))
        low_thresh = max(5.0, min(high_thresh - 1.0, low_thresh))

        edges = cv2.Canny(blurred, int(low_thresh), int(high_thresh))
        edge_density = float(np.count_nonzero(edges) / edges.size)
        return edge_density
    except Exception:
        return 0.0


def calculate_sobel_magnitude_mean(
    image: np.ndarray,
    ksize: int = 3
) -> float:
    """
    Feature 7: Sobel Edge Magnitude (Mean).

    What it measures:
        Measures the average gradient sharpness and transition magnitude across both spatial dimensions:
            Magnitude(x, y) = sqrt( G_x(x, y)^2 + G_y(x, y)^2 )
            Sobel_Magnitude_Mean = (1 / HW) * sum_{x,y} Magnitude(x, y)

    Calculation Method:
        1. Applies 3x3 Sobel convolution operators in horizontal (cv2.Sobel dx=1, dy=0)
           and vertical (cv2.Sobel dx=0, dy=1) directions using 64-bit floating point precision.
        2. Computes the Euclidean gradient magnitude at every pixel: sqrt(G_x^2 + G_y^2).
        3. Computes the arithmetic mean over all image pixels.

    Relevance to Steel Surface Defect Analysis:
        The gradient magnitude directly quantifies the optical contrast between surface anomalies
        and the surrounding steel matrix. Deep slag inclusions, oxidized patches, and scratch gouges
        create steep intensity cliffs, elevating the mean Sobel magnitude, whereas clean polished steel
        maintains a low, baseline gradient response.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        ksize: Size of the extended Sobel kernel (must be 1, 3, 5, or 7). Default: 3.

    Returns:
        float: Mean Sobel gradient magnitude across the entire surface.
    """
    try:
        # Convert to float64 to prevent truncation during squaring
        img_f = image.astype(np.float64)
        gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.mean(magnitude))
    except Exception:
        return 0.0


def calculate_laplacian_edge_variance(
    image: np.ndarray,
    ksize: int = 3
) -> float:
    """
    Feature 8: Laplacian Edge Variance.

    What it measures:
        Measures the variance of the second spatial derivative (Laplacian operator):
            Laplacian_Variance = Var( Delta I ) = (1 / HW) * sum_{x,y} ( Delta I(x, y) - mu_{Delta I} )^2
        This is a standard computer vision metric for structural frequency disruption and focus sharpness.

    Calculation Method:
        1. Evaluates the 2D Laplacian operator Delta I = d^2 I / dx^2 + d^2 I / dy^2 using
           cv2.Laplacian with a 3x3 aperture in CV_64F precision.
        2. Computes the statistical variance of the resulting Laplacian response map.

    Relevance to Steel Surface Defect Analysis:
        Smooth rolled steel manifests near-zero second-derivative fluctuations because intensity
        changes smoothly across the flat plate. Defects such as pitting, jagged inclusions, and
        crazing fissures introduce rapid second-order intensity oscillations, creating sharp peaks
        and valleys that drastically increase the variance of the Laplacian response.

    Args:
        image: Preprocessed 2D grayscale uint8 image array.
        ksize: Aperture size for computing the second-derivative Laplacian. Default: 3.

    Returns:
        float: Variance of the Laplacian response map.
    """
    try:
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        variance = float(np.var(laplacian))
        return variance
    except Exception:
        return 0.0

