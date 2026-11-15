"""
Group 3: Shape and Morphological Feature Extraction Module (Features 9 - 12).

Implements 4 geometric and topological descriptors extracted from binary defect masks:
  9. Connected Component Count
  10. Largest Component Solidity
  11. Defect Aspect Ratio
  12. Skeleton Branching Complexity

Features can be computed from:
  a) Ground-truth binary segmentation mask (used during offline training & ground-truth validation).
  b) Image-only candidate defect segmentation mask (used during leakage-free inference on new images
     where ground-truth masks are not available).
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def segment_candidate_defect_mask(
    image: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
    kernel_size: int = 15
) -> np.ndarray:
    """
    Leakage-Free Candidate Defect Region Segmentation.

    Extracts a binary candidate defect mask directly from the preprocessed grayscale
    image without relying on ground-truth RLE annotations. This prevents data leakage
    during real-time prediction and inference.

    Methodology:
      1. Morphological Top-Hat filter isolates bright surface anomalies against local background.
      2. Morphological Black-Hat filter isolates dark cavities, cracks, and pitting indentations.
      3. Morphological gradient/sum captures structural anomaly energy.
      4. If a defect-free reference template is supplied, blends the absolute structural difference.
      5. Applies Otsu's dynamic threshold to segment prominent surface defects.
      6. Applies morphological area filtering to suppress camera thermal noise.

    Args:
        image: Preprocessed 2D grayscale uint8 image array (256x1600).
        reference_image: Optional preprocessed defect-free reference template.
        kernel_size: Structuring element dimension. Default is 15.

    Returns:
        np.ndarray: 2D binary candidate defect mask (uint8, values 0 and 255).
    """
    try:
        if image is None:
            return np.zeros((256, 1600), dtype=np.uint8)

        # 1. Structural morphology filters (Top-Hat + Black-Hat)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        anomaly_map = cv2.add(top_hat, black_hat)

        # 2. Blend with reference image difference if available
        if reference_image is not None and reference_image.shape == image.shape:
            ref_diff = cv2.absdiff(image, reference_image)
            anomaly_map = cv2.addWeighted(anomaly_map, 0.7, ref_diff, 0.3, 0)

        # 3. Dynamic Otsu Thresholding
        blurred = cv2.GaussianBlur(anomaly_map, (5, 5), 0)
        otsu_thresh, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological opening to suppress trivial sub-pixel salt noise
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        filtered_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, clean_kernel)

        return filtered_mask
    except Exception:
        return np.zeros_like(image, dtype=np.uint8) if image is not None else np.zeros((256, 1600), dtype=np.uint8)


def _resolve_mask(mask: Optional[np.ndarray], image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Helper to resolve mask: uses explicit mask if provided; otherwise segments from image."""
    if mask is not None:
        return mask
    if image is not None:
        return segment_candidate_defect_mask(image)
    return None


def calculate_connected_component_count(
    mask: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None
) -> float:
    """
    Feature 9: Connected Component Count.

    What it measures:
        Counts the total number of distinct, isolated foreground defect regions
        present in the binary segmentation mask using 8-connectivity.

    Args:
        mask: Optional 2D binary numpy array (uint8).
        image: Optional preprocessed grayscale image (used when mask is None).

    Returns:
        float: Number of foreground connected components (0.0 for defect-free images).
    """
    try:
        active_mask = _resolve_mask(mask, image)
        if active_mask is None or not np.any(active_mask):
            return 0.0

        binary = (active_mask > 0).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
        return float(max(0, num_labels - 1))
    except Exception:
        return 0.0


def calculate_largest_component_solidity(
    mask: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None
) -> float:
    """
    Feature 10: Largest Component Solidity.

    What it measures:
        Measures the structural compactness and convexity of the most significant defect component:
            Solidity = Area(C_max) / ConvexHullArea(C_max)

    Args:
        mask: Optional 2D binary numpy array (uint8).
        image: Optional preprocessed grayscale image (used when mask is None).

    Returns:
        float: Solidity of the largest defect component in [0.0, 1.0] (0.0 for defect-free).
    """
    try:
        active_mask = _resolve_mask(mask, image)
        if active_mask is None or not np.any(active_mask):
            return 0.0

        binary = (active_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area <= 0.0:
            return 0.0

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0.0:
            return 0.0

        solidity = float(area / hull_area)
        return min(1.0, max(0.0, solidity))
    except Exception:
        return 0.0


def calculate_defect_aspect_ratio(
    mask: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None
) -> float:
    """
    Feature 11: Defect Aspect Ratio.

    What it measures:
        Measures the elongation and directional bias of the largest defect component:
            Aspect_Ratio = max(width, height) / min(width, height)

    Args:
        mask: Optional 2D binary numpy array (uint8).
        image: Optional preprocessed grayscale image (used when mask is None).

    Returns:
        float: Elongation aspect ratio >= 1.0 for defects (0.0 for defect-free images).
    """
    try:
        active_mask = _resolve_mask(mask, image)
        if active_mask is None or not np.any(active_mask):
            return 0.0

        binary = (active_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest_contour)

        if w <= 0 or h <= 0:
            return 0.0

        aspect_ratio = float(max(w, h) / max(min(w, h), 1))
        return aspect_ratio
    except Exception:
        return 0.0


def calculate_skeleton_branching_complexity(
    mask: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None
) -> float:
    """
    Feature 12: Skeleton Branching Complexity.

    What it measures:
        Measures the topological connectivity and branching network density of the defect structure:
            Branching_Complexity = (Number of Junction Points) / (Total Skeleton Pixels + eps)

    Args:
        mask: Optional 2D binary numpy array (uint8).
        image: Optional preprocessed grayscale image (used when mask is None).

    Returns:
        float: Branching ratio in [0.0, 1.0] (0.0 for defect-free images).
    """
    try:
        active_mask = _resolve_mask(mask, image)
        if active_mask is None or not np.any(active_mask):
            return 0.0

        binary = active_mask > 0
        skeleton = skeletonize(binary)
        skeleton_pixels = int(skeleton.sum())

        if skeleton_pixels == 0:
            return 0.0

        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.int32)

        neighbor_count = ndimage.convolve(skeleton.astype(np.int32), kernel, mode='constant', cval=0)
        junctions = (neighbor_count >= 3) & skeleton
        junction_count = int(junctions.sum())

        complexity = float(junction_count / skeleton_pixels)
        return min(1.0, max(0.0, complexity))
    except Exception:
        return 0.0


