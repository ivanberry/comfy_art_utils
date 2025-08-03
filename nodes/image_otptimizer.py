"""
Smart Image Upscaling Nodes for ComfyUI

FIXED ISSUES:
- Removed incorrect common_upscale call that was causing hanging
- Updated logic to work with fixed upscale model factors (2x, 4x)
- Added proper error handling and progress logging
- Improved device management and tensor operations

These nodes provide intelligent upscaling that accounts for upscale model limitations
and provides multiple strategies for reaching target dimensions.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math

class SmartUpscaleCalculator:
    """
    Calculates the optimal upscale factor to reach target dimensions,
    then applies upscale model multiple times if needed
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "target_width": ("INT", {
                    "default": 2000, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Target width for final image"
                }),
                "target_height": ("INT", {
                    "default": 2000, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Target height for final image"
                }),
                "upscale_method": ([
                    "closest_factor",
                    "multiple_passes", 
                    "hybrid"
                ], {
                    "default": "hybrid",
                    "tooltip": "How to handle upscaling: closest_factor=single pass, multiple_passes=apply model multiple times, hybrid=model+resize"
                }),
                "max_upscale_factor": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "Maximum factor for single model application (typical models support 2x or 4x)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("upscaled_image", "actual_scale_factor", "process_info")
    FUNCTION = "smart_upscale"
    CATEGORY = "image/upscaling"
    
    def smart_upscale(self, image, upscale_model, target_width, target_height, upscale_method, max_upscale_factor):
        try:
            print(f"[SmartUpscaleCalculator] Starting upscale process...")
            
            # Get original dimensions
            batch_size, height, width, channels = image.shape
            original_width = width
            original_height = height
            
            print(f"[SmartUpscaleCalculator] Input image: {original_width}x{original_height}")
            
            # Calculate required scale factors
            width_scale = target_width / original_width
            height_scale = target_height / original_height
            
            # Use the larger scale factor to ensure we reach target size
            required_scale = max(width_scale, height_scale)
            
            process_info = f"Original: {original_width}x{original_height}, Target: {target_width}x{target_height}, Required scale: {required_scale:.2f}"
            print(f"[SmartUpscaleCalculator] {process_info}")
            
            # Import the upscale function from ComfyUI
            from comfy.model_management import get_torch_device
            device = get_torch_device()
            print(f"[SmartUpscaleCalculator] Using device: {device}")
        
        if upscale_method == "closest_factor":
            # Single pass - upscale models typically have fixed factors (2x, 4x)
            if required_scale <= max_upscale_factor:
                upscaled_image = self._apply_upscale_model(image, upscale_model, max_upscale_factor)
                actual_factor = max_upscale_factor
                process_info += f" | Applied model: {max_upscale_factor}x"
            else:
                # Model can't reach target, use model then resize
                upscaled_image = self._apply_upscale_model(image, upscale_model, max_upscale_factor)
                remaining_scale = required_scale / max_upscale_factor
                upscaled_image = self._resize_image_tensor(upscaled_image, remaining_scale)
                actual_factor = required_scale
                process_info += f" | Model: {max_upscale_factor}x + Resize: {remaining_scale:.2f}x"
            
        elif upscale_method == "multiple_passes":
            # Apply model multiple times - each pass uses the model's fixed factor
            current_image = image
            total_factor = 1.0
            passes = 0
            
            while total_factor < required_scale and passes < 3:  # Max 3 passes to prevent infinite loops
                remaining_scale = required_scale / total_factor
                
                if remaining_scale >= max_upscale_factor:
                    # Full model pass
                    current_image = self._apply_upscale_model(current_image, upscale_model, max_upscale_factor)
                    total_factor *= max_upscale_factor
                    passes += 1
                    process_info += f" | Pass {passes}: {max_upscale_factor}x"
                else:
                    # Final resize for remaining scale
                    if remaining_scale > 1.1:  # Only resize if significant
                        current_image = self._resize_image_tensor(current_image, remaining_scale)
                        total_factor *= remaining_scale
                        process_info += f" | Final resize: {remaining_scale:.2f}x"
                    break
            
            upscaled_image = current_image
            actual_factor = total_factor
            
        else:  # hybrid method
            # Use model for quality, then resize for exact dimensions
            if required_scale <= max_upscale_factor:
                # Single model pass is enough
                upscaled_image = self._apply_upscale_model(image, upscale_model, max_upscale_factor)
                actual_factor = max_upscale_factor
                process_info += f" | Single model pass: {max_upscale_factor}x"
            else:
                # Model upscale + final resize
                upscaled_image = self._apply_upscale_model(image, upscale_model, max_upscale_factor)
                
                # Calculate remaining scale needed
                remaining_scale = required_scale / max_upscale_factor
                upscaled_image = self._resize_image_tensor(upscaled_image, remaining_scale)
                
                actual_factor = required_scale
                process_info += f" | Model: {max_upscale_factor}x + Resize: {remaining_scale:.2f}x"
        
            # Final resize to exact target dimensions if needed
            print(f"[SmartUpscaleCalculator] Performing final resize to exact dimensions...")
            final_image = self._resize_to_exact_dimensions(upscaled_image, target_width, target_height)
            
            print(f"[SmartUpscaleCalculator] Upscale complete. Final size: {final_image.shape[2]}x{final_image.shape[1]}")
            return (final_image, actual_factor, process_info)
            
        except Exception as e:
            print(f"[SmartUpscaleCalculator] Error during upscaling: {e}")
            # Return original image on error
            error_info = f"Error: {str(e)}"
            return (image, 1.0, error_info)
    
    def _apply_upscale_model(self, image, upscale_model, scale_factor):
        """Apply upscale model - upscale models handle scaling internally"""
        from comfy.model_management import get_torch_device
        
        device = get_torch_device()
        input_shape = image.shape
        print(f"[Upscaler] Applying upscale model to {input_shape[2]}x{input_shape[1]} image")
        
        # Convert to the format expected by upscale models (BCHW)
        samples = image.movedim(-1, 1)  # BHWC to BCHW
        
        # Move to appropriate device
        samples = samples.to(device)
        
        # Apply the upscale model directly - it will handle the scaling
        # Note: scale_factor is ignored here as upscale models have fixed scaling
        try:
            print(f"[Upscaler] Running upscale model...")
            upscaled = upscale_model(samples)
            output_shape = upscaled.shape
            print(f"[Upscaler] Upscale model output: {output_shape[3]}x{output_shape[2]}")
        except Exception as e:
            print(f"[Upscaler] Error applying upscale model: {e}")
            # Fallback to original if model fails
            upscaled = samples
        
        # Convert back to BHWC format
        result = upscaled.movedim(1, -1)
        
        return result
    
    def _resize_image_tensor(self, image_tensor, scale_factor):
        """Resize image tensor by scale factor using high-quality interpolation"""
        batch_size, height, width, channels = image_tensor.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Convert to BCHW for F.interpolate
        tensor_bchw = image_tensor.permute(0, 3, 1, 2)
        
        # Use bicubic interpolation for high quality
        resized = F.interpolate(
            tensor_bchw, 
            size=(new_height, new_width), 
            mode='bicubic', 
            align_corners=False
        )
        
        # Convert back to BHWC
        return resized.permute(0, 2, 3, 1)
    
    def _resize_to_exact_dimensions(self, image_tensor, target_width, target_height):
        """Resize to exact target dimensions"""
        batch_size, height, width, channels = image_tensor.shape
        
        if width == target_width and height == target_height:
            return image_tensor
        
        # Convert to BCHW for F.interpolate
        tensor_bchw = image_tensor.permute(0, 3, 1, 2)
        
        # Resize to exact dimensions
        resized = F.interpolate(
            tensor_bchw, 
            size=(target_height, target_width), 
            mode='bicubic', 
            align_corners=False
        )
        
        # Convert back to BHWC
        return resized.permute(0, 2, 3, 1)


class ModelAwareUpscaler:
    """
    Advanced version that can detect model capabilities and optimize accordingly
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "target_width": ("INT", {
                    "default": 2000, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 1
                }),
                "target_height": ("INT", {
                    "default": 2000, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 1
                }),
                "model_type": ([
                    "RealESRGAN_2x", 
                    "RealESRGAN_4x", 
                    "ESRGAN_4x",
                    "SwinIR_4x",
                    "auto_detect"
                ], {
                    "default": "auto_detect",
                    "tooltip": "Specify model type for optimal factor calculation"
                }),
                "quality_priority": ([
                    "speed", 
                    "balanced", 
                    "quality"
                ], {
                    "default": "balanced",
                    "tooltip": "Trade-off between speed and quality"
                }),
                "allow_downscale": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Allow downscaling if target is smaller than original"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("upscaled_image", "scale_factor", "process_log", "final_width", "final_height")
    FUNCTION = "model_aware_upscale"
    CATEGORY = "image/upscaling"
    
    def model_aware_upscale(self, image, upscale_model, target_width, target_height, 
                           model_type, quality_priority, allow_downscale):
        
        # Get original dimensions
        batch_size, height, width, channels = image.shape
        
        # Calculate required scale
        width_scale = target_width / width
        height_scale = target_height / height
        required_scale = max(width_scale, height_scale)
        
        # Determine model capabilities
        model_factors = {
            "RealESRGAN_2x": 2.0,
            "RealESRGAN_4x": 4.0,
            "ESRGAN_4x": 4.0,
            "SwinIR_4x": 4.0,
            "auto_detect": 4.0  # Default assumption
        }
        
        max_model_factor = model_factors.get(model_type, 4.0)
        
        process_log = f"Input: {width}x{height}, Target: {target_width}x{target_height}\n"
        process_log += f"Required scale: {required_scale:.2f}, Model max: {max_model_factor}x\n"
        
        # Handle downscaling case
        if required_scale < 1.0:
            if allow_downscale == "false":
                process_log += "Downscaling not allowed, returning original size"
                return (image, 1.0, process_log, width, height)
            else:
                # Direct downscale without model
                final_image = self._resize_to_exact_dimensions(image, target_width, target_height)
                process_log += f"Direct downscale to {target_width}x{target_height}"
                return (final_image, required_scale, process_log, target_width, target_height)
        
        # Strategy based on quality priority
        if quality_priority == "speed":
            # Single pass with model's fixed factor
            if required_scale <= max_model_factor:
                upscaled = self._apply_upscale_model(image, upscale_model, max_model_factor)
                process_log += f"Single model pass: {max_model_factor}x"
            else:
                upscaled = self._apply_upscale_model(image, upscale_model, max_model_factor)
                remaining = required_scale / max_model_factor
                upscaled = self._resize_image_tensor(upscaled, remaining)
                process_log += f"Model {max_model_factor}x + resize {remaining:.2f}x"
                
        elif quality_priority == "quality":
            # Multiple model passes for maximum quality
            current_image = image
            total_factor = 1.0
            pass_count = 0
            
            while total_factor < required_scale and pass_count < 3:
                remaining = required_scale / total_factor
                
                if remaining >= max_model_factor:
                    # Full model pass
                    current_image = self._apply_upscale_model(current_image, upscale_model, max_model_factor)
                    total_factor *= max_model_factor
                    pass_count += 1
                    process_log += f"Pass {pass_count}: {max_model_factor}x, "
                else:
                    # Final resize for remaining scale
                    if remaining > 1.1:  # Only resize if significant
                        current_image = self._resize_image_tensor(current_image, remaining)
                        total_factor *= remaining
                        process_log += f"Final resize: {remaining:.2f}x"
                    break
            
            upscaled = current_image
            
        else:  # balanced
            # Optimal single or double pass with fixed model factors
            if required_scale <= max_model_factor:
                upscaled = self._apply_upscale_model(image, upscale_model, max_model_factor)
                process_log += f"Single model pass: {max_model_factor}x"
            elif required_scale <= max_model_factor * max_model_factor:
                # Two model passes
                upscaled = self._apply_upscale_model(image, upscale_model, max_model_factor)
                second_factor = required_scale / max_model_factor
                
                if second_factor <= max_model_factor and second_factor > 1.1:
                    upscaled = self._apply_upscale_model(upscaled, upscale_model, max_model_factor)
                    process_log += f"Two model passes: {max_model_factor}x + {max_model_factor}x"
                else:
                    upscaled = self._resize_image_tensor(upscaled, second_factor)
                    process_log += f"Model {max_model_factor}x + resize {second_factor:.2f}x"
            else:
                # Model + resize
                upscaled = self._apply_upscale_model(image, upscale_model, max_model_factor)
                remaining = required_scale / max_model_factor
                upscaled = self._resize_image_tensor(upscaled, remaining)
                process_log += f"Model {max_model_factor}x + resize {remaining:.2f}x"
        
        # Final resize to exact dimensions
        final_image = self._resize_to_exact_dimensions(upscaled, target_width, target_height)
        
        return (final_image, required_scale, process_log, target_width, target_height)
    
    def _apply_upscale_model(self, image, upscale_model, scale_factor):
        """Apply upscale model - upscale models handle scaling internally"""
        from comfy.model_management import get_torch_device
        
        device = get_torch_device()
        input_shape = image.shape
        print(f"[Upscaler] Applying upscale model to {input_shape[2]}x{input_shape[1]} image")
        
        # Convert to the format expected by upscale models (BCHW)
        samples = image.movedim(-1, 1)  # BHWC to BCHW
        
        # Move to appropriate device
        samples = samples.to(device)
        
        # Apply the upscale model directly - it will handle the scaling
        # Note: scale_factor is ignored here as upscale models have fixed scaling
        try:
            print(f"[Upscaler] Running upscale model...")
            upscaled = upscale_model(samples)
            output_shape = upscaled.shape
            print(f"[Upscaler] Upscale model output: {output_shape[3]}x{output_shape[2]}")
        except Exception as e:
            print(f"[Upscaler] Error applying upscale model: {e}")
            # Fallback to original if model fails
            upscaled = samples
        
        # Convert back to BHWC format
        result = upscaled.movedim(1, -1)
        
        return result
    
    def _resize_image_tensor(self, image_tensor, scale_factor):
        """Same as SmartUpscaleCalculator"""
        batch_size, height, width, channels = image_tensor.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        tensor_bchw = image_tensor.permute(0, 3, 1, 2)
        resized = F.interpolate(tensor_bchw, size=(new_height, new_width), mode='bicubic', align_corners=False)
        return resized.permute(0, 2, 3, 1)
    
    def _resize_to_exact_dimensions(self, image_tensor, target_width, target_height):
        """Same as SmartUpscaleCalculator"""
        batch_size, height, width, channels = image_tensor.shape
        
        if width == target_width and height == target_height:
            return image_tensor
        
        tensor_bchw = image_tensor.permute(0, 3, 1, 2)
        resized = F.interpolate(tensor_bchw, size=(target_height, target_width), mode='bicubic', align_corners=False)
        return resized.permute(0, 2, 3, 1)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SmartUpscaleCalculator": SmartUpscaleCalculator,
    "ModelAwareUpscaler": ModelAwareUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartUpscaleCalculator": "🧮 Smart Upscale Calculator",
    "ModelAwareUpscaler": "🤖 Model-Aware Upscaler",
}