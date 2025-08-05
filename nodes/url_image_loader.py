import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import folder_paths
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings for problematic certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
                "verify_ssl": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "image/loaders"
    
    def create_error_image(self, error_message, width=512, height=512):
        """Create a red error image with text"""
        image = Image.new('RGB', (width, height), color='darkred')
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        # Add error text
        text_lines = [
            "IMAGE LOAD ERROR",
            "",
            str(error_message)[:50] + "..." if len(str(error_message)) > 50 else str(error_message)
        ]
        
        y_offset = height // 4
        for line in text_lines:
            if font:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(line) * 6  # Approximate
                text_height = 11
                
            x = (width - text_width) // 2
            draw.text((x, y_offset), line, fill='white', font=font)
            y_offset += text_height + 10
            
        return image
    
    def load_image(self, url, referer="", user_agent="", timeout=30, verify_ssl=False):
        """
        Load an image from a URL with custom headers.
        
        Args:
            url (str): The URL of the image to load
            referer (str): Custom referer header
            user_agent (str): Custom user agent header
            timeout (int): Request timeout in seconds
            verify_ssl (bool): Whether to verify SSL certificates
            
        Returns:
            tuple: A tuple containing the loaded image as a tensor
        """
        error_message = None
        
        try:
            if not url or not url.strip():
                error_message = "Empty URL provided"
                raise ValueError(error_message)
            
            # Prepare headers
            headers = {}
            if user_agent:
                headers['User-Agent'] = user_agent
            if referer:
                headers['Referer'] = referer
            
            # Create session with retry strategy
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            print(f"[ArtUtils] Loading image from: {url}")
            
            # Make the request with SSL verification control
            response = session.get(
                url, 
                headers=headers, 
                timeout=timeout, 
                stream=True,
                verify=verify_ssl  # Allow bypassing SSL verification
            )
            response.raise_for_status()
            
            # Load image from response content
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print(f"[ArtUtils] Successfully loaded image: {image.size}")
            
            # Convert PIL image to tensor format expected by ComfyUI
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]  # Add batch dimension
            
            return (image_tensor,)
            
        except Exception as e:
            error_message = str(e)
            print(f"[ArtUtils ERROR] Failed to load {url}: {error_message}")
            
            # Create error image instead of raising exception
            error_image = self.create_error_image(error_message)
            
            # Convert error image to tensor
            image_np = np.array(error_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            return (image_tensor,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "URLImageLoader": URLImageLoader
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "URLImageLoader": "URL Image Loader"
}