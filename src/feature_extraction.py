"""
Master Feature Extraction Pipeline.

Coordinates the extraction of exactly 15 handcrafted features across:
  - Texture (Features 1 - 5)
  - Edge (Features 6 - 8)
  - Shape / Morphology (Features 9 - 12)
  - Statistical & Frequency (Features 13 - 15)

Combines multi-label target indicators and outputs tabular records.
"""

from typing import Any, Dict, List, Optional
import os
import cv2
import numpy as np
import pandas as pd

from src.dataset_loader import SteelDatasetLoader, TARGET_COLUMNS
from src.mask_decoder import rle_to_mask, create_union_mask
from src.preprocessing import load_and_preprocess_image
from src.texture_features import (
    calculate_glcm_contrast,
    calculate_glcm_correlation,
    calculate_glcm_energy,
    calculate_lbp_entropy,
    calculate_gabor_energy
)
from src.edge_features import (
    calculate_canny_edge_density,
    calculate_sobel_magnitude_mean,
    calculate_laplacian_edge_variance
)
from src.shape_features import (
    calculate_connected_component_count,
    calculate_largest_component_solidity,
    calculate_defect_aspect_ratio,
    calculate_skeleton_branching_complexity
)
from src.statistical_features import (
    calculate_image_entropy,
    calculate_ssim_to_reference,
    calculate_fourier_peak_frequency
)


# Exact 15 handcrafted feature names
FEATURE_COLUMNS = [
    # Group 1: Texture (5)
    "glcm_contrast",
    "glcm_correlation",
    "glcm_energy",
    "lbp_entropy",
    "gabor_energy",
    # Group 2: Edge (3)
    "canny_edge_density",
    "sobel_magnitude_mean",
    "laplacian_variance",
    # Group 3: Shape & Morphology (4)
    "connected_component_count",
    "largest_component_solidity",
    "defect_aspect_ratio",
    "skeleton_branching_complexity",
    # Group 4: Statistical & Frequency (3)
    "image_entropy",
    "ssim_to_reference",
    "fourier_peak_frequency"
]


def extract_features_for_image(
    image_path: str,
    rle_map: Dict[int, str],
    reference_image: np.ndarray,
    target_shape: tuple = (256, 1600)
) -> Dict[str, float]:
    """
    Extracts all 15 handcrafted image features for a single steel surface image.

    Args:
        image_path: Absolute or relative path to the steel image file.
        rle_map: Mapping of defect ClassId (1-4) to Severstal RLE strings.
        reference_image: Preprocessed defect-free reference image for SSIM.
        target_shape: Expected image and mask dimensions.

    Returns:
        Dict[str, float]: Dictionary mapping feature names to numerical values.
    """
    # 1. Load and Preprocess Image
    img_u8, _ = load_and_preprocess_image(image_path, target_shape=target_shape)

    # 2. Decode and Combine Binary Defect Masks (Union Mask)
    if rle_map:
        masks = [rle_to_mask(rle, shape=target_shape) for rle in rle_map.values()]
        union_mask = create_union_mask(masks, shape=target_shape)
    else:
        union_mask = np.zeros(target_shape, dtype=np.uint8)

    # Group 1: Texture Features (5)
    f_glcm_contrast = calculate_glcm_contrast(img_u8)
    f_glcm_correlation = calculate_glcm_correlation(img_u8)
    f_glcm_energy = calculate_glcm_energy(img_u8)
    f_lbp_entropy = calculate_lbp_entropy(img_u8)
    f_gabor_energy = calculate_gabor_energy(img_u8)

    # Group 2: Edge Features (3)
    f_canny_density = calculate_canny_edge_density(img_u8)
    f_sobel_magnitude = calculate_sobel_magnitude_mean(img_u8)
    f_laplacian_var = calculate_laplacian_edge_variance(img_u8)

    # Group 3: Shape & Morphological Features (4) - From Binary Mask
    f_cc_count = calculate_connected_component_count(union_mask)
    f_solidity = calculate_largest_component_solidity(union_mask)
    f_aspect_ratio = calculate_defect_aspect_ratio(union_mask)
    f_skeleton_complexity = calculate_skeleton_branching_complexity(union_mask)

    # Group 4: Statistical & Frequency Features (3)
    f_image_entropy = calculate_image_entropy(img_u8)
    f_ssim = calculate_ssim_to_reference(img_u8, reference_image)
    f_fourier_peak = calculate_fourier_peak_frequency(img_u8)

    return {
        "glcm_contrast": float(f_glcm_contrast),
        "glcm_correlation": float(f_glcm_correlation),
        "glcm_energy": float(f_glcm_energy),
        "lbp_entropy": float(f_lbp_entropy),
        "gabor_energy": float(f_gabor_energy),
        "canny_edge_density": float(f_canny_density),
        "sobel_magnitude_mean": float(f_sobel_magnitude),
        "laplacian_variance": float(f_laplacian_var),
        "connected_component_count": float(f_cc_count),
        "largest_component_solidity": float(f_solidity),
        "defect_aspect_ratio": float(f_aspect_ratio),
        "skeleton_branching_complexity": float(f_skeleton_complexity),
        "image_entropy": float(f_image_entropy),
        "ssim_to_reference": float(f_ssim),
        "fourier_peak_frequency": float(f_fourier_peak)
    }


def _worker_extract(args: tuple) -> Optional[Dict[str, Any]]:
    """Worker function for concurrent/parallel extraction."""
    img_id, img_path, rle_map, labels, ref_path = args
    try:
        # Load reference image inside worker
        ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        features = extract_features_for_image(img_path, rle_map, ref_image)

        row = {
            "ImageId": img_id,
            **features,
            "Crazing": int(labels["Crazing"]),
            "Inclusion": int(labels["Inclusion"]),
            "Patches": int(labels["Patches"]),
            "Pitted_Surface": int(labels["Pitted_Surface"])
        }
        return row
    except Exception as e:
        print(f"Error extracting features for {img_id}: {e}")
        return None

