"""
Padding calculation utilities for video crop operations
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any


def find_mask_bounds(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of the masked area.
    
    Args:
        mask: Binary mask tensor (H, W) or (1, H, W)
    
    Returns:
        Tuple of (min_y, min_x, max_y, max_x) coordinates
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    # Find non-zero pixels
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    
    if len(coords) == 0:
        # No mask found, return center area
        h, w = mask.shape
        return h // 4, w // 4, 3 * h // 4, 3 * w // 4
    
    min_y = coords[:, 0].min().item()
    max_y = coords[:, 0].max().item()
    min_x = coords[:, 1].min().item()
    max_x = coords[:, 1].max().item()
    
    return min_y, min_x, max_y, max_x


def calculate_dynamic_padding(
    image_shape: Tuple[int, int], 
    mask: torch.Tensor, 
    target_width: int, 
    target_height: int
) -> Dict[str, Any]:
    """
    Calculate dynamic padding to achieve exact target dimensions with mask centered in output.
    
    The mask will always be centered in the final output. If the desired crop area extends
    beyond the image boundaries, padding will be applied to maintain centering.
    
    Args:
        image_shape: (height, width) of the original image
        mask: Binary mask tensor
        target_width: Desired output width
        target_height: Desired output height
    
    Returns:
        Dictionary containing crop coordinates and metadata
    """
    orig_height, orig_width = image_shape
    
    # Find mask bounds
    min_y, min_x, max_y, max_x = find_mask_bounds(mask)
    
    # Calculate mask center (this will be the center of our output)
    mask_center_y = (min_y + max_y) / 2.0
    mask_center_x = (min_x + max_x) / 2.0
    
    # Calculate ideal crop bounds centered on mask (integer coordinates)
    # Use floor for consistent behavior - we'll compensate with padding
    ideal_crop_top = int(mask_center_y - target_height / 2.0)
    ideal_crop_left = int(mask_center_x - target_width / 2.0)
    ideal_crop_bottom = ideal_crop_top + target_height
    ideal_crop_right = ideal_crop_left + target_width
    
    # Calculate how much we need to pad to maintain perfect centering
    # This accounts for the fractional part lost in integer conversion
    center_offset_y = mask_center_y - (ideal_crop_top + target_height / 2.0)
    center_offset_x = mask_center_x - (ideal_crop_left + target_width / 2.0)
    
    # Calculate padding needed when crop extends beyond image boundaries
    boundary_pad_top = max(0, -ideal_crop_top)
    boundary_pad_left = max(0, -ideal_crop_left)
    boundary_pad_bottom = max(0, ideal_crop_bottom - orig_height)
    boundary_pad_right = max(0, ideal_crop_right - orig_width)
    
    # Add centering compensation to padding
    # Distribute the centering offset as padding to maintain perfect centering
    pad_top = boundary_pad_top + max(0, round(center_offset_y))
    pad_left = boundary_pad_left + max(0, round(center_offset_x))
    pad_bottom = boundary_pad_bottom + max(0, -round(center_offset_y))
    pad_right = boundary_pad_right + max(0, -round(center_offset_x))
    
    # Adjust crop coordinates to stay within image boundaries
    actual_crop_top = max(0, ideal_crop_top)
    actual_crop_left = max(0, ideal_crop_left)
    actual_crop_bottom = min(orig_height, ideal_crop_bottom)
    actual_crop_right = min(orig_width, ideal_crop_right)
    
    # Calculate actual crop dimensions
    actual_crop_width = actual_crop_right - actual_crop_left
    actual_crop_height = actual_crop_bottom - actual_crop_top
    
    return {
        'crop_coords': (actual_crop_top, actual_crop_left, actual_crop_bottom, actual_crop_right),
        'padding': (pad_top, pad_bottom, pad_left, pad_right),
        'original_size': (orig_height, orig_width),
        'target_size': (target_height, target_width),
        'mask_bounds': (min_y, min_x, max_y, max_x),
        'mask_center': (mask_center_y, mask_center_x),
        'actual_crop_size': (actual_crop_height, actual_crop_width)
    }