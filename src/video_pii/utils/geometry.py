"""Geometry utilities for bounding box operations."""

from typing import Sequence

import numpy as np


BBox = tuple[float, float, float, float] | Sequence[float] | np.ndarray


def iou(box_a: BBox, box_b: BBox) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box_a: First bounding box as (x1, y1, x2, y2).
        box_b: Second bounding box as (x1, y1, x2, y2).
        
    Returns:
        IoU value in [0, 1].
    """
    # Extract coordinates
    a_x1, a_y1, a_x2, a_y2 = box_a[0], box_a[1], box_a[2], box_a[3]
    b_x1, b_y1, b_x2, b_y2 = box_b[0], box_b[1], box_b[2], box_b[3]
    
    # Compute intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area_a = max(0.0, a_x2 - a_x1) * max(0.0, a_y2 - a_y1)
    area_b = max(0.0, b_x2 - b_x1) * max(0.0, b_y2 - b_y1)
    
    # Compute union
    union_area = area_a + area_b - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def smooth_bbox(
    prev_box: BBox, 
    new_box: BBox, 
    alpha: float
) -> tuple[float, float, float, float]:
    """Smooth bounding box using exponential moving average.
    
    smoothed = alpha * new_box + (1 - alpha) * prev_box
    
    Args:
        prev_box: Previous (smoothed) bounding box (x1, y1, x2, y2).
        new_box: New detection bounding box (x1, y1, x2, y2).
        alpha: Smoothing factor in [0, 1]. 
            Higher alpha = more weight to new detection.
            
    Returns:
        Smoothed bounding box as (x1, y1, x2, y2).
    """
    if alpha <= 0.0:
        return (float(prev_box[0]), float(prev_box[1]), float(prev_box[2]), float(prev_box[3]))
    if alpha >= 1.0:
        return (float(new_box[0]), float(new_box[1]), float(new_box[2]), float(new_box[3]))
    
    x1 = alpha * new_box[0] + (1 - alpha) * prev_box[0]
    y1 = alpha * new_box[1] + (1 - alpha) * prev_box[1]
    x2 = alpha * new_box[2] + (1 - alpha) * prev_box[2]
    y2 = alpha * new_box[3] + (1 - alpha) * prev_box[3]
    
    return (float(x1), float(y1), float(x2), float(y2))


def expand_bbox(
    bbox: BBox,
    padding: float,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by a padding ratio and clip to image bounds.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2).
        padding: Padding ratio (e.g., 0.3 = 30% expansion on each side).
        img_width: Image width for clipping.
        img_height: Image height for clipping.
        
    Returns:
        Expanded and clipped bounding box as (x1, y1, x2, y2) integers.
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = x2 - x1
    h = y2 - y1
    
    # Expand
    pad_w = w * padding
    pad_h = h * padding
    
    x1_exp = x1 - pad_w
    y1_exp = y1 - pad_h
    x2_exp = x2 + pad_w
    y2_exp = y2 + pad_h
    
    # Clip to image bounds
    x1_clip = int(max(0, x1_exp))
    y1_clip = int(max(0, y1_exp))
    x2_clip = int(min(img_width, x2_exp))
    y2_clip = int(min(img_height, y2_exp))
    
    return (x1_clip, y1_clip, x2_clip, y2_clip)

