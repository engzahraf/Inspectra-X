"""
Group 3: Shape and Morphological Feature Extraction Module (Features 9 - 12).

Implements 4 geometric and topological descriptors extracted from binary defect masks:
  9. Connected Component Count
  10. Largest Component Solidity
  11. Defect Aspect Ratio
  12. Skeleton Branching Complexity

All features are calculated on the 2D binary defect segmentation mask (or union of masks
for multi-defect images). For defect-free images, safe documented defaults (0.0) are returned.
"""

from typing import Tuple
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def calculate_connected_component_count(mask: np.ndarray) -> float:
    """
    Feature 9: Connected Component Count.

    What it measures:
        Counts the total number of distinct, isolated foreground defect regions
        present in the binary segmentation mask using 8-connectivity.

    Calculation Method:
        1. Ensures the mask is in binary 8-bit format (values 0 and 255 or 1).
        2. Evaluates cv2.connectedComponentsWithStats using 8-way pixel adjacency.
        3. Subtracts 1 from the total label count to exclude background (label 0).
        4. If the mask is entirely zero (defect-free), returns 0.0.

    Relevance to Steel Surface Defect Analysis:
        The spatial distribution of defect clusters is a primary discriminator between classes.
        Pitted surface defects typically manifest as numerous distributed pinhole clusters
        (yielding high component counts, often > 20), whereas patch defects usually form a single
        cohesive defect area (count = 1 or 2). Defect-free steel produces 0 components.

    Args:
        mask: 2D binary numpy array (uint8).

    Returns:
        float: Number of foreground connected components (0.0 for defect-free images).
    """
    try:
        if mask is None or not np.any(mask):
            return 0.0

        binary = (mask > 0).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
        # Background is label 0; foreground count is num_labels - 1
        return float(max(0, num_labels - 1))
    except Exception:
        return 0.0


def calculate_largest_component_solidity(mask: np.ndarray) -> float:
    """
    Feature 10: Largest Component Solidity.

    What it measures:
        Measures the structural compactness and convexity of the most significant defect component:
            Solidity = Area(C_max) / ConvexHullArea(C_max)
        Values range in [0.0, 1.0], where 1.0 indicates a perfectly convex defect geometry.

    Calculation Method:
        1. Identifies external contours of all foreground defect regions using cv2.findContours.
        2. If no valid contours exist (defect-free mask), returns the safe documented default 0.0.
        3. Identifies the contour with the largest area C_max.
        4. Computes the 2D convex hull of C_max via cv2.convexHull.
        5. Calculates the area of the contour and the area of its convex hull.
        6. Computes the ratio Area / ConvexHullArea. If the hull area is zero, returns 0.0.

    Relevance to Steel Surface Defect Analysis:
        Solidity captures the boundary irregularity and concavity of defect geometries.
        Circular slag inclusions and isolated pitting craters have high solidity (close to 1.0)
        due to convex shapes. In contrast, crazing fractures and jagged surface tears feature
        interconnected fissures with large concavities, resulting in significantly lower solidity.

    Args:
        mask: 2D binary numpy array (uint8).

    Returns:
        float: Solidity of the largest defect component in [0.0, 1.0] (0.0 for defect-free).
    """
    try:
        if mask is None or not np.any(mask):
            return 0.0

        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Find largest component by area
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


def calculate_defect_aspect_ratio(mask: np.ndarray) -> float:
    """
    Feature 11: Defect Aspect Ratio.

    What it measures:
        Measures the elongation and directional bias of the largest defect component:
            Aspect_Ratio = max(width, height) / min(width, height)
        An aspect ratio near 1.0 signifies an isotropic shape (circular or square), while values
        substantially greater than 1.0 indicate prolonged linear elongation.

    Calculation Method:
        1. Extracts external contours from the binary mask.
        2. If no defect contour is found (defect-free image), returns safe default 0.0.
        3. Identifies the largest defect contour C_max.
        4. Obtains its bounding rectangle (x, y, w, h) via cv2.boundingRect.
        5. Computes the ratio of major dimension to minor dimension: max(w, h) / max(min(w, h), 1).

    Relevance to Steel Surface Defect Analysis:
        Continuous rolling mills subject steel strips to severe unidirectional tensile forces.
        Longitudinal rolling cracks and slivers stretch along the rolling axis, creating very high
        aspect ratios. Conversely, pitted surface indentations and small inclusions are relatively
        equiaxed, exhibiting low aspect ratios close to 1.0.

    Args:
        mask: 2D binary numpy array (uint8).

    Returns:
        float: Elongation aspect ratio >= 1.0 for defects (0.0 for defect-free images).
    """
    try:
        if mask is None or not np.any(mask):
            return 0.0

        binary = (mask > 0).astype(np.uint8) * 255
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


def calculate_skeleton_branching_complexity(mask: np.ndarray) -> float:
    """
    Feature 12: Skeleton Branching Complexity.

    What it measures:
        Measures the topological connectivity and branching network density of the defect structure:
            Branching_Complexity = (Number of Junction Points) / (Total Skeleton Pixels + eps)
        A high ratio indicates a heavily bifurcated, dendritic network of fissures.

    Calculation Method:
        1. If the mask is empty (defect-free), returns the safe documented default 0.0.
        2. Applies morphological medial-axis thinning using skimage.morphology.skeletonize to
           reduce the 2D defect regions into 1-pixel-thick central skeletons.
        3. Counts the total number of active skeleton pixels (skeleton length).
        4. Uses 2D spatial convolution with an 8-neighbor structuring element:
               [1, 1, 1]
               [1, 0, 1]
               [1, 1, 1]
           to count the number of skeleton neighbors surrounding each skeleton pixel.
        5. Pixels with 3 or more neighbors along the skeleton are topological junction/branch points.
        6. Computes the ratio of junction points to total skeleton pixels.

    Relevance to Steel Surface Defect Analysis:
        Class 1 defects (Crazing) consist of interconnected networks of fine cooling cracks that
        branch repeatedly in a web-like formation. This morphological branching structure produces
        high junction density. In contrast, simple patches, solid inclusions, or pits have isolated
        skeletons with few to zero branching intersections.

    Args:
        mask: 2D binary numpy array (uint8).

    Returns:
        float: Branching ratio in [0.0, 1.0] (0.0 for defect-free images).
    """
    try:
        if mask is None or not np.any(mask):
            return 0.0

        binary = mask > 0
        skeleton = skeletonize(binary)
        skeleton_pixels = int(skeleton.sum())

        if skeleton_pixels == 0:
            return 0.0

        # Structuring kernel to count 8-neighbors
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.int32)

        neighbor_count = ndimage.convolve(skeleton.astype(np.int32), kernel, mode='constant', cval=0)
        # Only evaluate points that belong to the skeleton
        junctions = (neighbor_count >= 3) & skeleton
        junction_count = int(junctions.sum())

        complexity = float(junction_count / skeleton_pixels)
        return min(1.0, max(0.0, complexity))
    except Exception:
        return 0.0

