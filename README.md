# ComfyUI Video Crop and Stitch

Video-focused crop and stitch nodes for ComfyUI that handle batch frame processing with dynamic padding based on target output dimensions.

## Features

- **Static Mask Processing**: Optimized for videos where the mask position remains constant across all frames
- **Dynamic Padding**: Automatically calculates padding to achieve exact target output dimensions
- **Batch Processing**: Efficiently processes entire video sequences
- **Smart Blending**: Seamless stitching with configurable edge blending
- **Context-Aware Cropping**: Includes configurable context around the masked area

## Nodes

### Video Crop Node (Dynamic Padding)

Crops video frames around a static mask area with dynamic padding to achieve exact target dimensions.

**Inputs:**
- `images`: Batch of video frames (IMAGE)
- `mask`: Static mask that applies to all frames (MASK)
- `target_width`: Desired output width (INT, default: 1500)
- `target_height`: Desired output height (INT, default: 520)
- `padding_mode`: How to pad when extending beyond image bounds ("reflect", "replicate", "constant")
- `padding_value`: Value for constant padding (FLOAT, 0.0-1.0)
- `context_factor`: Factor to expand crop area around mask (FLOAT, optional, default: 1.2)

**Outputs:**
- `cropped_images`: Cropped and padded video frames
- `crop_metadata`: Metadata needed for stitching back

### Video Stitch Node (Reassemble)

Stitches processed video frames back into their original positions and dimensions.

**Inputs:**
- `processed_images`: Processed cropped frames (IMAGE)
- `crop_metadata`: Metadata from the crop node (CROP_METADATA)
- `original_images`: Original full-size frames for background (IMAGE)
- `blend_radius`: Radius for blending edges (INT, optional, default: 16)
- `blend_strength`: Strength of blending (FLOAT, optional, default: 0.5)
- `preserve_original_outside_mask`: Keep original pixels outside processed area (BOOLEAN, optional, default: True)

**Outputs:**
- `stitched_images`: Final reassembled video frames

## Use Cases

Perfect for scenarios such as:
- Fixed camera position with consistent subject area
- Processing specific regions of video content
- Video inpainting with static masks
- Architectural/landscape videos where the area of interest doesn't move
- UI element processing in screen recordings

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory
2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ComfyUI-Video-CropAndStitch.git
   ```
3. Restart ComfyUI

## Example Workflow

1. Load your video frames using a video loader node
2. Create or load a static mask for the area you want to process
3. Connect both to the **Video Crop Node** with your desired target dimensions
4. Process the cropped frames through your desired nodes (inpainting, upscaling, etc.)
5. Connect the processed frames, crop metadata, and original frames to the **Video Stitch Node**
6. Save the final stitched video

## Technical Details

### Dynamic Padding Algorithm

The node calculates padding to achieve exact target dimensions while ensuring the masked area is included:

1. Find the bounding box of the masked area
2. Center the crop region on the mask
3. Expand to target dimensions with smart boundary handling
4. Apply configurable padding modes when extending beyond image bounds
5. Final resize to exact target dimensions if needed

### Memory Efficiency

- Static mask analysis is performed once for the entire video sequence
- Batch processing minimizes overhead
- Efficient tensor operations for padding and blending

## License

GNU GENERAL PUBLIC LICENSE Version 3

## Acknowledgments

Inspired by the excellent work in [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) by lquesada, adapted specifically for video processing workflows.