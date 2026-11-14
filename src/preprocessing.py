"""
Image Preprocessing and Reference Template Module.

Implements consistent preprocessing pipeline:
  1. Grayscale loading and verification.
  2. Contrast Limited Adaptive Histogram Equalization (CLAHE).
  3. Dynamic range normalization.
  4. Generation/loading of reproducible defect-free reference template for SSIM.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np


TARGET_IMAGE_SHAPE: Tuple[int, int] = (256, 1600)  # (Height, Width)


def preprocess_image(
    image_input: np.ndarray,
    target_shape: Tuple[int, int] = TARGET_IMAGE_SHAPE,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies consistent preprocessing to a steel surface image.

    Pipeline:
      1. Converts to single-channel 8-bit grayscale if color.
      2. Validates / aligns dimensions to target_shape without distortion.
      3. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to balance
         illumination variations across the 1600px longitudinal rolling strip.
      4. Generates both uint8 ([0, 255]) and normalized float64 ([0.0, 1.0]) representations.

    Args:
        image_input: Grayscale or BGR image as numpy array.
        target_shape: (height, width) tuple. Default is (256, 1600).
        clip_limit: Threshold for contrast limiting in CLAHE. Default is 2.0.
        tile_grid_size: Grid size for histogram equalization. Default is (8, 8).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
          - preprocessed_uint8: uint8 array in range [0, 255] (for GLCM, LBP, Canny, SSIM).
          - normalized_float: float64 array in range [0.0, 1.0] (for gradient/frequency filters).
    """
    if image_input is None:
        raise ValueError("Input image is None")

    # Step 1: Grayscale conversion
    if len(image_input.shape) == 3:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    elif len(image_input.shape) == 2:
        gray = image_input.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image_input.shape}")

    # Step 2: Dimension alignment
    target_h, target_w = target_shape
    if gray.shape != (target_h, target_w):
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Step 3: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    preprocessed_uint8 = clahe.apply(gray)

    # Step 4: Floating-point normalization [0.0, 1.0]
    normalized_float = preprocessed_uint8.astype(np.float64) / 255.0

    return preprocessed_uint8, normalized_float


def load_and_preprocess_image(
    image_path: str,
    target_shape: Tuple[int, int] = TARGET_IMAGE_SHAPE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image from disk and executes standard preprocessing.

    Args:
        image_path: Path to the image file.
        target_shape: Expected (height, width).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (preprocessed_uint8, normalized_float)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read image at {image_path}")

    return preprocess_image(img, target_shape=target_shape)


def get_or_create_reference_template(
    dataset_loader,
    save_path: str = "features/reference_defect_free.png",
    num_samples: int = 50
) -> np.ndarray:
    """
    Retrieves or creates a deterministic, reproducible defect-free reference template.

    Methodology:
      To avoid picking an arbitrary reference image, this function takes a deterministic
      subset of certified defect-free steel surface images (sorted alphabetically by ImageId),
      preprocesses each image, and computes their pixel-wise median.
      This produces a noise-reduced 'Golden Template' representing standard defect-free
      surface morphology.

    Args:
        dataset_loader: SteelDatasetLoader instance.
        save_path: Filepath where the reference image is cached.
        num_samples: Number of defect-free images to aggregate (default: 50).

    Returns:
        np.ndarray: Preprocessed uint8 reference image of shape (256, 1600).
    """
    if os.path.exists(save_path):
        cached_ref = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        if cached_ref is not None and cached_ref.shape == TARGET_IMAGE_SHAPE:
            return cached_ref

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    clean_ids = sorted(dataset_loader.get_defect_free_image_ids())
    if not clean_ids:
        # Fallback to zero/mid-gray baseline if no defect-free images
        ref_template = np.full(TARGET_IMAGE_SHAPE, 128, dtype=np.uint8)
        cv2.imwrite(save_path, ref_template)
        return ref_template

    selected_ids = clean_ids[:min(num_samples, len(clean_ids))]
    accumulated_images = []

    for img_id in selected_ids:
        info = dataset_loader.get_image_info(img_id)
        if info and os.path.exists(info['image_path']):
            p_uint8, _ = load_and_preprocess_image(info['image_path'])
            accumulated_images.append(p_uint8)

    if accumulated_images:
        stack = np.stack(accumulated_images, axis=0)
        # Compute median across defect-free sample images to remove localized artifacts
        median_ref = np.median(stack, axis=0).astype(np.uint8)
        # Apply gentle CLAHE to ensure uniform reference contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ref_template = clahe.apply(median_ref)
    else:
        ref_template = np.full(TARGET_IMAGE_SHAPE, 128, dtype=np.uint8)

    cv2.imwrite(save_path, ref_template)
    return ref_template

