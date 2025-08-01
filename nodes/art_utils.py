"""
Art-Utils for ComfyUI
A collection of utility nodes for creative workflows
"""

import torch
import numpy as np
from PIL import Image
import io
import os
import hashlib
import json
import datetime
import folder_paths
from minio import Minio
from minio.error import S3Error

# =============================================================================
# MINIO NODES
# =============================================================================

class ArtUtils_MinioUploader:
    """Upload images to MinIO S3 storage"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "minio_endpoint": ("STRING", {"default": "localhost:9000"}),
                "access_key": ("STRING", {"default": "minioadmin"}),
                "secret_key": ("STRING", {"default": "minioadmin"}),
                "bucket_name": ("STRING", {"default": "comfyui"}),
                "object_name_prefix": ("STRING", {"default": "image"}),
                "secure": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("minio_url", "object_name", "status")
    FUNCTION = "upload_to_minio"
    CATEGORY = "ArtUtils/Storage"
    OUTPUT_NODE = True

    def upload_to_minio(self, images, minio_endpoint, access_key, secret_key, 
                       bucket_name, object_name_prefix="image", secure=False, 
                       prompt=None, extra_pnginfo=None):
        try:
            client = Minio(minio_endpoint, access_key=access_key, 
                          secret_key=secret_key, secure=secure)

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                print(f"[ArtUtils] Created bucket: {bucket_name}")

            uploaded_urls = []
            uploaded_object_names = []
            
            for i, image_tensor in enumerate(images):
                image_np = image_tensor.cpu().numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

                buffer = io.BytesIO()
                image_pil.save(buffer, format="PNG")
                buffer.seek(0)
                image_data = buffer.read()

                sha256_hash = hashlib.sha256(image_data).hexdigest()
                object_name = f"{object_name_prefix}_{sha256_hash[:10]}_{i}.png"
                uploaded_object_names.append(object_name)

                client.put_object(bucket_name, object_name, io.BytesIO(image_data),
                                len(image_data), content_type='image/png')

                protocol = "https" if secure else "http"
                url = f"{protocol}://{minio_endpoint}/{bucket_name}/{object_name}"
                uploaded_urls.append(url)
                print(f"[ArtUtils] Uploaded: {url}")

            result_url = uploaded_urls[0] if uploaded_urls else ""
            result_object_name = uploaded_object_names[0] if uploaded_object_names else ""
            status = f"Uploaded {len(images)} image(s)"

            return {"ui": {"text": f"Uploaded to: {result_url}"}, 
                   "result": (result_url, result_object_name, status)}

        except Exception as e:
            error_msg = f"Upload error: {e}"
            print(f"[ArtUtils ERROR] {error_msg}")
            return {"ui": {"text": error_msg}, "result": ("", "", error_msg)}

class ArtUtils_MinioImageLoader:
    """Load images from MinIO S3 storage"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "minio_endpoint": ("STRING", {"default": "localhost:9000"}),
                "access_key": ("STRING", {"default": "minioadmin"}),
                "secret_key": ("STRING", {"default": "minioadmin"}),
                "bucket_name": ("STRING", {"default": "comfyui"}),
                "object_name": ("STRING", {"default": "image_0.png"}),
                "secure": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "status")
    FUNCTION = "load_image"
    CATEGORY = "ArtUtils/Storage"

    def load_image(self, minio_endpoint, access_key, secret_key, bucket_name, 
                  object_name, secure=False):
        try:
            client = Minio(minio_endpoint, access_key=access_key, 
                          secret_key=secret_key, secure=secure)

            print(f"[ArtUtils] Loading '{object_name}' from '{bucket_name}'")

            response = client.get_object(bucket_name, object_name)
            image_data = io.BytesIO(response.read())
            response.close()
            response.release_conn()

            img = Image.open(image_data)
            image_rgb = img.convert("RGB")
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if 'A' in img.getbands():
                mask_np = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,]
            else:
                mask_tensor = torch.ones((1, img.height, img.width), dtype=torch.float32)

            return (image_tensor, mask_tensor, "Image loaded successfully")

        except Exception as e:
            error_msg = f"Load error: {e}"
            print(f"[ArtUtils ERROR] {error_msg}")
            return (None, None, error_msg)

# =============================================================================
# IMAGE UTILITIES
# =============================================================================

class ArtUtils_ImageMetadataExtractor:
    """Extract metadata from images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata_json", "summary")
    FUNCTION = "extract_metadata"
    CATEGORY = "ArtUtils/Image"
    
    def extract_metadata(self, images):
        metadata_list = []
        
        for i, image_tensor in enumerate(images):
            image_np = image_tensor.cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            
            metadata = {
                "image_index": i,
                "size": image_pil.size,
                "mode": image_pil.mode,
                "tensor_shape": list(image_tensor.shape),
                "tensor_dtype": str(image_tensor.dtype)
            }
            metadata_list.append(metadata)
        
        metadata_json = json.dumps(metadata_list, indent=2)
        summary = f"Processed {len(images)} images. First: {metadata_list[0]['size']} {metadata_list[0]['mode']}"
        
        return (metadata_json, summary)

class ArtUtils_BatchImageSaver:
    """Save multiple images with custom naming"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "batch"}),
                "subfolder": ("STRING", {"default": ""}),
                "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
                "add_hash": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    FUNCTION = "save_images"
    CATEGORY = "ArtUtils/Image"
    OUTPUT_NODE = True
    
    def save_images(self, images, filename_prefix, subfolder, format, quality, add_timestamp, add_hash):
        saved_paths = []
        
        if subfolder:
            save_dir = os.path.join(self.output_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
        
        for i, image_tensor in enumerate(images):
            image_np = image_tensor.cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            
            filename_parts = [filename_prefix]
            if timestamp:
                filename_parts.append(timestamp)
            if add_hash:
                buffer = io.BytesIO()
                image_pil.save(buffer, format=format)
                hash_str = hashlib.md5(buffer.getvalue()).hexdigest()[:8]
                filename_parts.append(hash_str)
            filename_parts.append(f"{i:04d}")
            
            filename = "_".join(filename_parts) + f".{format.lower()}"
            filepath = os.path.join(save_dir, filename)
            
            if format == "JPEG":
                image_pil = image_pil.convert("RGB")
                image_pil.save(filepath, format=format, quality=quality)
            else:
                image_pil.save(filepath, format=format)
            
            saved_paths.append(filepath)
        
        return (json.dumps(saved_paths),)

# =============================================================================
# WORKFLOW UTILITIES
# =============================================================================

class ArtUtils_URLImageLoader:
    """Load images from URLs"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {"multiline": True, "default": "https://example.com/image.jpg"}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 300}),
                "user_agent": ("STRING", {"default": "Mozilla/5.0 (compatible; ArtUtils/1.0)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "status")
    FUNCTION = "load_from_urls"
    CATEGORY = "ArtUtils/Image"
    
    def load_from_urls(self, urls, timeout, user_agent):
        import urllib.request
        from urllib.parse import urlparse
        
        try:
            # Parse URLs (one per line)
            url_list = [url.strip() for url in urls.split('\n') if url.strip()]
            
            if not url_list:
                return (None, None, "No valid URLs provided")
            
            images = []
            masks = []
            loaded_count = 0
            max_width = 0
            max_height = 0
            
            # First pass: load images and find max dimensions
            temp_images = []
            for i, url in enumerate(url_list):
                try:
                    # Validate URL
                    parsed = urlparse(url)
                    if not parsed.scheme or not parsed.netloc:
                        print(f"[ArtUtils] Invalid URL: {url}")
                        continue
                    
                    # Create request with headers
                    req = urllib.request.Request(url, headers={'User-Agent': user_agent})
                    
                    # Download image
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        image_data = response.read()
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(image_data))
                    temp_images.append(img)
                    
                    # Track max dimensions
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)
                    
                    loaded_count += 1
                    print(f"[ArtUtils] Loaded image {i+1}: {url} ({img.width}x{img.height})")
                    
                except Exception as e:
                    print(f"[ArtUtils ERROR] Failed to load {url}: {e}")
                    continue
            
            if not temp_images:
                return (None, None, "No images loaded successfully")
            
            # Second pass: resize all images to max dimensions and convert to tensors
            for img in temp_images:
                # Resize to max dimensions
                if img.width != max_width or img.height != max_height:
                    img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Process image tensor
                image_rgb = img.convert("RGB")
                image_np = np.array(image_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                images.append(image_tensor)
                
                # Process mask
                if 'A' in img.getbands():
                    mask_np = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_np)[None,]
                else:
                    mask_tensor = torch.ones((1, max_height, max_width), dtype=torch.float32)
                masks.append(mask_tensor)
            
            # Combine all images and masks
            combined_images = torch.cat(images, dim=0)
            combined_masks = torch.cat(masks, dim=0)
            status = f"Loaded {loaded_count}/{len(url_list)} images, resized to {max_width}x{max_height}"
            
            return (combined_images, combined_masks, status)
            
        except Exception as e:
            error_msg = f"URL loading error: {e}"
            print(f"[ArtUtils ERROR] {error_msg}")
            return (None, None, error_msg)

class ArtUtils_WorkflowLogger:
    """Log workflow information"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {"default": "Workflow checkpoint"}),
                "log_level": (["INFO", "DEBUG", "WARNING", "ERROR"], {"default": "INFO"}),
                "save_to_file": ("BOOLEAN", {"default": False}),
            },
            "optional": {"data": ("STRING", {"default": ""})},
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_output",)
    FUNCTION = "log_message"
    CATEGORY = "ArtUtils/Workflow"
    
    def log_message(self, message, log_level, save_to_file, data=""):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {log_level}: {message}"
        
        if data:
            log_entry += f"\nData: {data}"
        
        print(f"[ArtUtils] {log_entry}")
        
        if save_to_file:
            log_dir = os.path.join(folder_paths.get_output_directory(), "art-utils-logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"workflow_{datetime.datetime.now().strftime('%Y%m%d')}.log")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        
        return (log_entry,)

# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ArtUtils_MinioUploader": ArtUtils_MinioUploader,
    "ArtUtils_MinioImageLoader": ArtUtils_MinioImageLoader,
    "ArtUtils_URLImageLoader": ArtUtils_URLImageLoader,
    "ArtUtils_ImageMetadataExtractor": ArtUtils_ImageMetadataExtractor,
    "ArtUtils_BatchImageSaver": ArtUtils_BatchImageSaver,
    "ArtUtils_WorkflowLogger": ArtUtils_WorkflowLogger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtUtils_MinioUploader": "🎨 MinIO Image Uploader",
    "ArtUtils_MinioImageLoader": "🎨 MinIO Image Loader",
    "ArtUtils_URLImageLoader": "🎨 URL Image Loader",
    "ArtUtils_ImageMetadataExtractor": "🎨 Image Metadata Extractor",
    "ArtUtils_BatchImageSaver": "🎨 Batch Image Saver",
    "ArtUtils_WorkflowLogger": "🎨 Workflow Logger",
}
