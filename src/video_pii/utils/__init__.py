"""Utility modules.

Note: Imports are done lazily to avoid circular imports with track module.
"""

from .device import (
    DeviceInfo,
    get_best_device,
    is_cuda_available,
    is_mps_available,
    list_available_devices,
    validate_device,
)
from .geometry import expand_bbox, iou, smooth_bbox

# visual_debug imports TrackState from track module, so we don't import it here
# to avoid circular imports. Import directly: from video_pii.utils.visual_debug import ...

__all__ = [
    "DeviceInfo",
    "expand_bbox",
    "get_best_device",
    "iou",
    "is_cuda_available",
    "is_mps_available",
    "list_available_devices",
    "smooth_bbox",
    "validate_device",
]
