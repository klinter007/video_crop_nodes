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
                "blend_radius": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Radius for blending edges (0 = no blending)"
                }),
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
        original_images: torch.Tensor,
        blend_radius: int = 16
    ) -> Tuple[torch.Tensor]:
        """
        Stitch processed frames back into original video dimensions.
        
        Args:
            processed_images: Processed cropped frames (B, H, W, C)
            crop_metadata: Metadata from the crop operation
            original_images: Original full-size frames (B, H, W, C)
            blend_radius: Radius for edge blending (0 = no blending)
            
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
        
        if blend_radius > 0:
            # Create blending mask for smooth edges
            blend_mask = self._create_blend_mask(
                crop_height, crop_width, blend_radius, processed_images.device
            )
            
            # Apply blending to avoid hard edges
            for b in range(batch_size):
                original_crop = result_images[b, top:bottom, left:right, :]
                processed_crop = processed_resized[b]
                
                # Blend the images with the mask
                blended = (
                    blend_mask.unsqueeze(-1) * processed_crop + 
                    (1 - blend_mask.unsqueeze(-1)) * original_crop
                )
                
                result_images[b, top:bottom, left:right, :] = blended
        else:
            # Direct replacement without blending
            result_images[:, top:bottom, left:right, :] = processed_resized
        
        return (result_images,)
    
    def _create_blend_mask(self, height: int, width: int, radius: int, device: torch.device) -> torch.Tensor:
        """
        Create a blending mask with smooth edges.
        
        Args:
            height: Height of the mask
            width: Width of the mask
            radius: Blending radius from edges
            device: Device to create the mask on
            
        Returns:
            Blending mask tensor with smooth falloff at edges
        """
        
        # Create coordinate grids
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        
        # Calculate distance from edges
        dist_top = y
        dist_bottom = height - 1 - y
        dist_left = x.unsqueeze(0).expand(height, -1)
        dist_right = width - 1 - dist_left
        
        # Minimum distance to any edge
        min_dist_vertical = torch.minimum(dist_top.unsqueeze(1).expand(-1, width), 
                                        dist_bottom.unsqueeze(1).expand(-1, width))
        min_dist_horizontal = torch.minimum(dist_left, dist_right)
        min_dist = torch.minimum(min_dist_vertical, min_dist_horizontal)
        
        # Create smooth falloff
        blend_mask = torch.clamp(min_dist / radius, 0.0, 1.0)
        
        # Apply smooth step function for better blending
        blend_mask = 3 * blend_mask**2 - 2 * blend_mask**3
        
        return blend_mask
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always process since we're dealing with video frames
        return float("inf")