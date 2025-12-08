"""Device detection and GPU utilities.

Provides functions for detecting available compute devices (CUDA, MPS, CPU)
and selecting the best available device for inference.
"""

from dataclasses import dataclass
from typing import Literal

DeviceType = Literal["cuda", "mps", "cpu"]


@dataclass
class DeviceInfo:
    """Information about an available compute device.
    
    Attributes:
        name: Device name (e.g., 'cuda:0', 'mps', 'cpu').
        type: Device type ('cuda', 'mps', 'cpu').
        memory_gb: Available memory in GB (None if unknown).
        is_available: Whether the device is available.
    """
    name: str
    type: DeviceType
    memory_gb: float | None = None
    is_available: bool = True


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps_available() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) is available."""
    try:
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        return False


def get_cuda_device_count() -> int:
    """Get the number of available CUDA devices."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def get_cuda_device_info(device_idx: int = 0) -> DeviceInfo | None:
    """Get information about a CUDA device.
    
    Args:
        device_idx: CUDA device index.
        
    Returns:
        DeviceInfo or None if device not available.
    """
    try:
        import torch
        if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_idx)
            memory_gb = props.total_memory / (1024 ** 3)
            return DeviceInfo(
                name=f"cuda:{device_idx}",
                type="cuda",
                memory_gb=round(memory_gb, 2),
                is_available=True,
            )
    except ImportError:
        pass
    return None


def get_best_device() -> str:
    """Get the best available compute device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if is_cuda_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    return "cpu"


def list_available_devices() -> list[DeviceInfo]:
    """List all available compute devices.
    
    Returns:
        List of DeviceInfo for all available devices.
    """
    devices: list[DeviceInfo] = []
    
    # Check CUDA devices
    cuda_count = get_cuda_device_count()
    for i in range(cuda_count):
        info = get_cuda_device_info(i)
        if info:
            devices.append(info)
    
    # Check MPS
    if is_mps_available():
        devices.append(DeviceInfo(name="mps", type="mps"))
    
    # CPU is always available
    devices.append(DeviceInfo(name="cpu", type="cpu"))
    
    return devices


def validate_device(device: str) -> str:
    """Validate and normalize device string.
    
    Args:
        device: Device string to validate.
        
    Returns:
        Validated device string.
        
    Raises:
        ValueError: If device is not valid or available.
    """
    device = device.lower().strip()
    
    if device == "auto":
        return get_best_device()
    
    if device.startswith("cuda"):
        if not is_cuda_available():
            raise ValueError("CUDA requested but not available. Use 'cpu' or 'auto'.")
        return device
    
    if device == "mps":
        if not is_mps_available():
            raise ValueError("MPS requested but not available. Use 'cpu' or 'auto'.")
        return device
    
    if device == "cpu":
        return device
    
    raise ValueError(f"Unknown device: {device}. Use 'cuda', 'mps', 'cpu', or 'auto'.")

