"""
Metadata handling for crop and stitch operations
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch


@dataclass
class CropMetadata:
    """
    Metadata container for crop and stitch operations
    """
    crop_coords: Tuple[int, int, int, int]  # (top, left, bottom, right)
    padding: Tuple[int, int, int, int]      # (top, bottom, left, right)
    original_size: Tuple[int, int]          # (height, width)
    target_size: Tuple[int, int]            # (height, width)
    mask_bounds: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    mask_center: Tuple[float, float]        # (center_y, center_x)
    actual_crop_size: Tuple[int, int]       # (height, width)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'crop_coords': self.crop_coords,
            'padding': self.padding,
            'original_size': self.original_size,
            'target_size': self.target_size,
            'mask_bounds': self.mask_bounds,
            'mask_center': self.mask_center,
            'actual_crop_size': self.actual_crop_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CropMetadata':
        """Create from dictionary"""
        return cls(**data)
    
    def get_crop_slice(self) -> Tuple[slice, slice]:
        """Get slice objects for cropping"""
        top, left, bottom, right = self.crop_coords
        return slice(top, bottom), slice(left, right)
    
    def get_stitch_coords(self) -> Tuple[int, int, int, int]:
        """Get coordinates for stitching back into original image"""
        return self.crop_coords