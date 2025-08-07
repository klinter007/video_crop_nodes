"""
ComfyUI Video Crop and Stitch Node Pack

Video-focused crop and stitch nodes for ComfyUI that handle batch frame processing
with dynamic padding based on target output dimensions.
"""

from .video_crop_node import VideoCropNode
from .video_stitch_node import VideoStitchNode

NODE_CLASS_MAPPINGS = {
    "VideoCropNode": VideoCropNode,
    "VideoStitchNode": VideoStitchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCropNode": "Video Crop - static mask",
    "VideoStitchNode": "Video Stitch (Reassemble)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]