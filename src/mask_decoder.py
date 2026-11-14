"""
Mask Decoder Module for Severstal Steel Defect Detection Dataset.

Implements Run-Length Encoding (RLE) decoding with column-major (Fortran order)
convention and multi-mask union combination for multi-label defect analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def rle_to_mask(
    rle_string: Optional[Union[str, float]],
    shape: Tuple[int, int] = (256, 1600)
) -> np.ndarray:
    """
    Decodes Severstal Run-Length Encoded (RLE) string into a 2D binary segmentation mask.

    Severstal RLE Encoding Convention:
      - 1-based pixel indexing.
      - Column-major (Fortran) ordering: pixels are indexed down columns from top
        to bottom, then left to right.
      - Encoded format: Pairs of integers [start_index, run_length, ...].
      - For example, '1 3 10 5' indicates pixels 1..3 and 10..14 in 1-based indexing.

    Args:
        rle_string: RLE encoded string of pairs, or NaN/None/empty for defect-free images.
        shape: Output mask dimensions (height, width). Default is (256, 1600).

    Returns:
        np.ndarray: Binary mask of shape `shape` with dtype uint8 (1 = defect, 0 = background).
    """
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)

    # Handle empty, NaN, or non-string inputs safely
    if rle_string is None or (isinstance(rle_string, float) and np.isnan(rle_string)):
        return mask.reshape((height, width), order='F')

    rle_str = str(rle_string).strip()
    if not rle_str or rle_str.lower() == 'nan':
        return mask.reshape((height, width), order='F')

    try:
        tokens = rle_str.split()
        if len(tokens) % 2 != 0:
            raise ValueError(f"Invalid RLE string length (not an even number of tokens): {len(tokens)}")

        starts = np.asarray(tokens[0::2], dtype=int) - 1  # Convert 1-based to 0-based indexing
        lengths = np.asarray(tokens[1::2], dtype=int)
        ends = starts + lengths

        total_pixels = height * width
        for start, end in zip(starts, ends):
            if start < 0 or end > total_pixels:
                # Clamp indices safely to avoid memory faults if malformed RLE
                start = max(0, start)
                end = min(total_pixels, end)
            if start < end:
                mask[start:end] = 1

        return mask.reshape((height, width), order='F')

    except Exception as e:
        # Graceful fallback: return empty mask on corrupted RLE
        return np.zeros((height, width), dtype=np.uint8)


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes a 2D binary mask into a Severstal-compliant RLE string (column-major).

    Args:
        mask: 2D binary numpy array.

    Returns:
        str: Encoded RLE string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_union_mask(
    masks: Union[List[np.ndarray], Dict[int, np.ndarray]],
    shape: Tuple[int, int] = (256, 1600)
) -> np.ndarray:
    """
    Combines multiple class defect masks into a single unified binary defect mask
    using logical bitwise OR.

    This ensures that when an image exhibits multiple defect classes, the topological
    and geometric feature extractors evaluate the complete defect structure present
    on the steel sheet.

    Args:
        masks: List or dictionary of binary mask arrays.
        shape: Target mask dimensions (height, width).

    Returns:
        np.ndarray: Combined binary mask with dtype uint8 (1 = defect present, 0 = background).
    """
    union_mask = np.zeros(shape, dtype=np.uint8)

    if isinstance(masks, dict):
        mask_list = list(masks.values())
    elif isinstance(masks, (list, tuple)):
        mask_list = masks
    else:
        return union_mask

    for m in mask_list:
        if m is not None and m.shape == shape:
            union_mask = np.logical_or(union_mask, m > 0).astype(np.uint8)

    return union_mask

