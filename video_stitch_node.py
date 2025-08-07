"""
Video Stitch Node

Stitches processed video frames back into their original positions and dimensions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Any, Dict
from .utils.metadata_handler import CropMetadata


class VideoStitchNode:
    """
    ComfyUI node for stitching processed video frames back into original dimensions.
    
    This node takes processed frames and crop metadata to reassemble the video
    back to its original size and position.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_images": ("IMAGE",),      # Processed cropped frames
                "crop_metadata": ("CROP_METADATA",), # Metadata from crop node
                "original_images": ("IMAGE",),       # Original full-size frames for background
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stitched_images",)
    FUNCTION = "stitch_video_frames"
    CATEGORY = "video/stitch"
    
    def stitch_video_frames(
        self, 
        processed_images: torch.Tensor, 
        crop_metadata: Dict[str, Any], 
        original_images: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Stitch processed frames back into original video dimensions with complete replacement.
        
        Args:
            processed_images: Processed cropped frames (B, H, W, C)
            crop_metadata: Metadata from the crop operation
            original_images: Original full-size frames (B, H, W, C)
            
        Returns:
            Tuple containing the stitched video frames
        """
        
        if processed_images.dim() != 4 or original_images.dim() != 4:
            raise ValueError("Both processed_images and original_images must be 4D tensors (B, H, W, C)")
        
        batch_size = processed_images.shape[0]
        if original_images.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: processed={batch_size}, original={original_images.shape[0]}")
        
        # Create metadata object from dictionary
        metadata = CropMetadata.from_dict(crop_metadata)
        
        # Get original dimensions
        orig_height, orig_width = metadata.original_size
        target_height, target_width = metadata.target_size
        
        # Start with original images as base
        result_images = original_images.clone()
        
        # Resize processed images back to the crop area size if needed
        crop_height = metadata.crop_coords[2] - metadata.crop_coords[0]
        crop_width = metadata.crop_coords[3] - metadata.crop_coords[1]
        
        if processed_images.shape[1] != crop_height or processed_images.shape[2] != crop_width:
            # Convert to (B, C, H, W) for interpolation
            processed_resized = processed_images.permute(0, 3, 1, 2)
            processed_resized = F.interpolate(
                processed_resized,
                size=(crop_height, crop_width),
                mode='bilinear',
                align_corners=False
            )
            # Convert back to (B, H, W, C)
            processed_resized = processed_resized.permute(0, 2, 3, 1)
        else:
            processed_resized = processed_images
        
        # Get crop coordinates
        top, left, bottom, right = metadata.crop_coords
        
        # Complete replacement: processed area overwrites original area
        result_images[:, top:bottom, left:right, :] = processed_resized
        
        return (result_images,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always process since we're dealing with video frames
        return float("inf")