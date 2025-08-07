"""
Video Crop Node with Dynamic Padding

Crops video frames around a static mask area with dynamic padding to achieve exact target dimensions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Any, Dict
from .utils.padding_calculator import calculate_dynamic_padding
from .utils.metadata_handler import CropMetadata


class VideoCropNode:
    """
    ComfyUI node for cropping video frames with dynamic padding based on target dimensions.
    
    This node is designed for video processing where:
    - The mask position is static across all frames
    - You want exact output dimensions (e.g., 1500x520)
    - Padding is calculated dynamically to include the masked area
    - Outputs both cropped images and corresponding cropped mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of video frames
                "mask": ("MASK",),     # Static mask (same size as images)
                "target_width": ("INT", {
                    "default": 1500,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Width of the cropped output"
                }),
                "target_height": ("INT", {
                    "default": 520,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Height of the cropped output"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "CROP_METADATA")
    RETURN_NAMES = ("cropped_images", "cropped_mask", "crop_metadata")
    FUNCTION = "crop_video_frames"
    CATEGORY = "video/crop"
    
    def crop_video_frames(
        self, 
        images: torch.Tensor, 
        mask: torch.Tensor, 
        target_width: int, 
        target_height: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Crop video frames centered on mask to achieve exact target dimensions.
        
        Args:
            images: Batch of video frames (B, H, W, C)
            mask: Static mask same size as images (H, W) or (1, H, W)
            target_width: Desired output width
            target_height: Desired output height
            
        Returns:
            Tuple of (cropped_images, cropped_mask, crop_metadata)
        """
        
        if images.dim() != 4:
            raise ValueError(f"Expected 4D images tensor (B, H, W, C), got {images.shape}")
        
        batch_size, orig_height, orig_width, channels = images.shape
        
        # Handle mask dimensions - it can be (H, W), (1, H, W), or (B, H, W)
        if mask.dim() == 3:
            # If mask has batch dimension, take the first frame (assuming static mask)
            if mask.shape[0] == batch_size:
                # Mask has same batch size as images - use first frame
                mask = mask[0]
            else:
                # Mask has different batch size - squeeze if possible
                mask = mask.squeeze(0)
        elif mask.dim() == 2:
            # Already 2D, perfect
            pass
        else:
            raise ValueError(f"Expected 2D or 3D mask, got {mask.shape}")
        
        # Verify mask and image dimensions match
        if mask.shape[0] != orig_height or mask.shape[1] != orig_width:
            raise ValueError(f"Mask size {mask.shape} doesn't match image size ({orig_height}, {orig_width})")
        
        # Calculate crop region and padding (once for all frames since mask is static)
        crop_info = calculate_dynamic_padding(
            image_shape=(orig_height, orig_width),
            mask=mask,
            target_width=target_width,
            target_height=target_height
        )
        
        # Create metadata object
        metadata = CropMetadata.from_dict(crop_info)
        
        # Extract crop coordinates
        top, left, bottom, right = crop_info['crop_coords']
        pad_top, pad_bottom, pad_left, pad_right = crop_info['padding']
        
        # Crop all frames
        cropped_images = images[:, top:bottom, left:right, :]
        
        # Crop the mask using the same coordinates
        cropped_mask = mask[top:bottom, left:right]
        
        # Apply padding if needed (use reflect mode for natural-looking extension)
        if any(p > 0 for p in crop_info['padding']):
            # Convert images to (B, C, H, W) for padding
            cropped_images = cropped_images.permute(0, 3, 1, 2)
            
            # Apply padding with reflect mode (mirrors image content at boundaries)
            cropped_images = F.pad(
                cropped_images, 
                (pad_left, pad_right, pad_top, pad_bottom), 
                mode='reflect'
            )
            
            # Convert images back to (B, H, W, C)
            cropped_images = cropped_images.permute(0, 2, 3, 1)
            
            # Apply padding to mask (use constant mode with 0 for mask padding)
            cropped_mask = F.pad(
                cropped_mask.unsqueeze(0),  # Add batch dimension for padding
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0.0
            ).squeeze(0)  # Remove batch dimension
        
        return (cropped_images, cropped_mask, metadata.to_dict())
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always process since we're dealing with video frames
        return float("inf")