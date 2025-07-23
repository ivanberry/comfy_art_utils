import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import folder_paths


class URLImageLoader:
    """
    A ComfyUI node from Comfy Art Utils that loads images from URLs with custom referer and user agent headers.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "referer": ("STRING", {"default": "", "multiline": False}),
                "user_agent": ("STRING", {"default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "multiline": False}),
            },
            "optional": {
                "timeout": ("INT", {"default": 30, "min": 1, "max": 300}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "image/loaders"
    
    def load_image(self, url, referer="", user_agent="", timeout=30):
        """
        Load an image from a URL with custom headers.
        
        Args:
            url (str): The URL of the image to load
            referer (str): Custom referer header
            user_agent (str): Custom user agent header
            timeout (int): Request timeout in seconds
            
        Returns:
            tuple: A tuple containing the loaded image as a tensor
        """
        try:
            # Prepare headers
            headers = {}
            if user_agent:
                headers['User-Agent'] = user_agent
            if referer:
                headers['Referer'] = referer
            
            # Make the request
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Load image from response content
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL image to tensor format expected by ComfyUI
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]  # Add batch dimension
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to load image from URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "URLImageLoader": URLImageLoader
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "URLImageLoader": "URL Image Loader"
}