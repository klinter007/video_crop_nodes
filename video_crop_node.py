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
            },
            "optional": {
                "size_rounded_to_8": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Round dimensions to nearest multiple of 8 (better for video codecs and AI models)"
                }),
                "wan_numbers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adjust frame count to 4n+1 format (5, 9, 13, 17, 21, etc.) for model compatibility"
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
        target_height: int,
        size_rounded_to_8: bool = False,
        wan_numbers: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Crop video frames centered on mask to achieve exact target dimensions.
        
        Args:
            images: Batch of video frames (B, H, W, C)
            mask: Static mask same size as images (H, W) or (1, H, W)
            target_width: Desired output width
            target_height: Desired output height
            size_rounded_to_8: Round dimensions to nearest multiple of 8
            wan_numbers: Adjust frame count to 4n+1 format (5, 9, 13, 17, 21, etc.)
            
        Returns:
            Tuple of (cropped_images, cropped_mask, crop_metadata)
        """
        
        if images.dim() != 4:
            raise ValueError(f"Expected 4D images tensor (B, H, W, C), got {images.shape}")
        
        batch_size, orig_height, orig_width, channels = images.shape
        
        # Adjust frame count for wan numbers (4n+1 format) if requested
        if wan_numbers:
            # Calculate the closest 4n+1 number
            if batch_size <= 5:
                target_frames = 5
            else:
                # Find closest 4n+1: current frames = 4n+r, we want 4m+1
                remainder = (batch_size - 1) % 4
                if remainder == 0:
                    # Already 4n+1, keep as is
                    target_frames = batch_size
                else:
                    # Round to nearest 4n+1
                    if remainder <= 2:
                        # Round down to previous 4n+1
                        target_frames = batch_size - remainder
                    else:
                        # Round up to next 4n+1
                        target_frames = batch_size + (4 - remainder)
            
            # Adjust the batch by trimming or repeating frames
            if target_frames < batch_size:
                # Trim frames from the end
                images = images[:target_frames]
                batch_size = target_frames
            elif target_frames > batch_size:
                # Repeat last frame to reach target
                last_frame = images[-1:].expand(target_frames - batch_size, -1, -1, -1)
                images = torch.cat([images, last_frame], dim=0)
                batch_size = target_frames
        
        # Handle mask dimensions - it can be (H, W), (1, H, W), or (B, H, W)
        # Always convert to single 2D mask since we use static mask for all frames
        if mask.dim() == 3:
            # Take the first frame as our static mask (regardless of batch size)
            mask = mask[0]
        elif mask.dim() == 2:
            # Already 2D, perfect
            pass
        else:
            raise ValueError(f"Expected 2D or 3D mask, got {mask.shape}")
        
        # Verify mask and image dimensions match
        if mask.shape[0] != orig_height or mask.shape[1] != orig_width:
            raise ValueError(f"Mask size {mask.shape} doesn't match image size ({orig_height}, {orig_width})")
        
        # Round dimensions to nearest multiple of 8 if requested
        if size_rounded_to_8:
            actual_target_width = round(target_width / 8) * 8
            actual_target_height = round(target_height / 8) * 8
        else:
            actual_target_width = target_width
            actual_target_height = target_height
        
        # Calculate crop region and padding (once for all frames since mask is static)
        crop_info = calculate_dynamic_padding(
            image_shape=(orig_height, orig_width),
            mask=mask,
            target_width=actual_target_width,
            target_height=actual_target_height
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
        
        # Expand the single cropped mask to match the batch size of images
        # Shape: (H, W) -> (B, H, W) where B = batch_size
        cropped_mask = cropped_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return (cropped_images, cropped_mask, metadata.to_dict())
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always process since we're dealing with video frames
        return float("inf")