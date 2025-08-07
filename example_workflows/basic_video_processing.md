# Basic Video Processing Workflow

This example demonstrates how to use the Video Crop and Stitch nodes for basic video processing.

## Workflow Steps

1. **Load Video Frames**
   - Use a video loader node to load your video as a batch of frames
   - Format should be (B, H, W, C) where B is the number of frames

2. **Create/Load Static Mask**
   - Create a mask that defines the area you want to process
   - The mask should be binary (0 for background, 1 for area to process)
   - This mask will apply to all frames in the video

3. **Video Crop Node Setup**
   - Connect video frames to `images` input
   - Connect mask to `mask` input
   - Set `target_width` and `target_height` to your desired output dimensions (e.g., 1500x520)
   - Adjust `context_factor` if you need more context around the masked area (default 1.2)
   - Choose appropriate `padding_mode`:
     - "reflect": Mirror pixels at boundaries (good for most cases)
     - "replicate": Extend edge pixels (good for solid backgrounds)
     - "constant": Fill with a constant value (set `padding_value`)

4. **Process Cropped Frames**
   - Connect the `cropped_images` output to your processing nodes
   - This could be inpainting, upscaling, style transfer, etc.
   - The frames are now at your target dimensions and ready for processing

5. **Video Stitch Node Setup**
   - Connect processed frames to `processed_images` input
   - Connect the `crop_metadata` from the crop node to `crop_metadata` input
   - Connect original frames to `original_images` input for background
   - Adjust blending parameters:
     - `blend_radius`: How many pixels to blend at edges (16 is usually good)
     - `blend_strength`: How much to blend processed vs original (0.5 = 50/50)
     - `preserve_original_outside_mask`: Usually keep as True

6. **Save Final Video**
   - Connect the `stitched_images` output to a video saver node
   - Your video is now processed and reassembled at original dimensions

## Example Parameters

For a typical video inpainting workflow:
- Target dimensions: 1024x1024 (good for most AI models)
- Context factor: 1.3 (30% extra context)
- Padding mode: "reflect"
- Blend radius: 24
- Blend strength: 0.6

## Tips

- **Performance**: The static mask means calculations are done once for the entire video, making it very efficient
- **Quality**: Higher `context_factor` gives more context but increases processing time
- **Blending**: Increase `blend_radius` if you see seams at the edges
- **Memory**: Process videos in smaller batches if you encounter memory issues