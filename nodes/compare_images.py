import torch
import numpy as np

class CompareImages:
    """
    Simple image comparison node with split view preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "split_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "image/compare"

    def compare(self, image_a, image_b, split_position=0.5):
        try:
            if image_a.shape != image_b.shape:
                raise ValueError(f"Image dimensions must match: {image_a.shape} vs {image_b.shape}")
            
            # Process each image in the batch
            results = []
            
            for i in range(image_a.shape[0]):
                # Get individual images
                img_a = image_a[i].cpu().numpy()
                img_b = image_b[i].cpu().numpy()
                
                h, w, c = img_a.shape
                split_x = int(w * split_position)
                
                # Create comparison: left = image_a, right = image_b
                result = img_a.copy()
                if split_x < w:
                    result[:, split_x:, :] = img_b[:, split_x:, :]
                
                # Add red divider line at split
                if 0 < split_x < w:
                    line_width = max(1, w // 500)  # Scale line width with image
                    start_x = max(0, split_x - line_width//2)
                    end_x = min(w, split_x + line_width//2 + 1)
                    
                    result[:, start_x:end_x, 0] = 1.0  # Red
                    result[:, start_x:end_x, 1] = 0.0  # Green
                    result[:, start_x:end_x, 2] = 0.0  # Blue
                
                # Convert back to tensor
                result_tensor = torch.from_numpy(result.astype(np.float32))
                results.append(result_tensor)
            
            # Stack results back into batch format
            final_result = torch.stack(results, dim=0)
            
            print(f"[CompareImages] Created comparison with split at {split_position:.2f}")
            return (final_result,)
            
        except Exception as e:
            print(f"[CompareImages ERROR] {e}")
            # Return original image_a on error
            return (image_a,)

NODE_CLASS_MAPPINGS = {
    "CompareImages": CompareImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompareImages": "Compare Images",
}