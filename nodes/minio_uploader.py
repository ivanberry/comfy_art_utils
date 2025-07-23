import torch
import numpy as np
from PIL import Image
import io
from minio import Minio
from minio.error import S3Error
import folder_paths
import os
import hashlib

class MinioUploader:
    """
    A custom node for ComfyUI that uploads an image to a MinIO S3 bucket.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "images": ("IMAGE",),
                "minio_endpoint": ("STRING", {"default": "localhost:9000"}),
                "access_key": ("STRING", {"default": "minioadmin"}),
                "secret_key": ("STRING", {"default": "minioadmin"}),
                "bucket_name": ("STRING", {"default": "comfyui"}),
                "object_name_prefix": ("STRING", {"default": "image"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("minio_url", "object_name")
    FUNCTION = "upload_to_minio"
    CATEGORY = "Image/Upload"
    OUTPUT_NODE = True

    def upload_to_minio(self, images, minio_endpoint, access_key, secret_key, bucket_name, object_name_prefix="image", prompt=None, extra_pnginfo=None):
        """
        The main function of the node. It takes an image tensor, uploads it to MinIO,
        and returns the URL and the object name.
        """
        try:
            # Initialize MinIO client
            client = Minio(
                minio_endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False  # Set to True if using HTTPS
            )

            # Check if the bucket exists, and create it if it doesn't
            found = client.bucket_exists(bucket_name)
            if not found:
                client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' created.")
            else:
                print(f"Bucket '{bucket_name}' already exists.")

            # Process and upload each image in the batch
            uploaded_urls = []
            uploaded_object_names = []
            for i, image_tensor in enumerate(images):
                # Convert tensor to PIL Image
                image_np = image_tensor.cpu().numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

                # Convert PIL Image to bytes
                buffer = io.BytesIO()
                image_pil.save(buffer, format="PNG")
                buffer.seek(0)
                image_data = buffer.read()
                
                # Generate a unique object name
                file_extension = "png"
                sha256_hash = hashlib.sha256(image_data).hexdigest()
                object_name = f"{object_name_prefix}_{sha256_hash[:10]}_{i}.{file_extension}"
                uploaded_object_names.append(object_name)

                # Upload the image data
                client.put_object(
                    bucket_name,
                    object_name,
                    io.BytesIO(image_data),
                    len(image_data),
                    content_type=f'image/{file_extension}'
                )

                # Construct the URL
                url = f"http://{minio_endpoint}/{bucket_name}/{object_name}"
                uploaded_urls.append(url)
                print(f"Successfully uploaded to {url}")

            # Return the URL and object name of the first image
            result_url = uploaded_urls[0] if uploaded_urls else ""
            result_object_name = uploaded_object_names[0] if uploaded_object_names else ""
            
            return {"ui": {"text": f"Uploaded to: {result_url}"}, "result": (result_url, result_object_name)}

        except S3Error as exc:
            print(f"ERROR: {exc}")
            return {"ui": {"text": f"Error: {exc}"}, "result": ("", "",)}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"ui": {"text": f"An unexpected error occurred: {e}"}, "result": ("", "",)}

class MinioImageLoader:
    """
    A custom node for ComfyUI that loads an image from a MinIO S3 bucket.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "minio_endpoint": ("STRING", {"default": "localhost:9000"}),
                "access_key": ("STRING", {"default": "minioadmin"}),
                "secret_key": ("STRING", {"default": "minioadmin"}),
                "bucket_name": ("STRING", {"default": "comfyui"}),
                "object_name": ("STRING", {"default": "image_0.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "Image/Load"

    def load_image(self, minio_endpoint, access_key, secret_key, bucket_name, object_name):
        try:
            # Initialize the MinIO client here, inside the main execution function
            client = Minio(
                minio_endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False # Set to True if using HTTPS
            )

            print(f"Attempting to load '{object_name}' from bucket '{bucket_name}'...")
            
            # Download the object from MinIO
            response = client.get_object(bucket_name, object_name)
            image_data = io.BytesIO(response.read())
            response.close()
            response.release_conn()

            print("Successfully downloaded image data.")

            # Convert the downloaded data into a PIL Image
            img = Image.open(image_data)
            
            # Process the image into a tensor for ComfyUI
            image_rgb = img.convert("RGB")
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            # Process the alpha channel into a mask if it exists
            if 'A' in img.getbands():
                mask_np = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,]
            else:
                # If no alpha, create a solid white mask
                mask_tensor = torch.ones((1, img.height, img.width), dtype=torch.float32)

            print("Image and mask tensors created successfully.")
            return (image_tensor, mask_tensor)

        except S3Error as exc:
            print(f"MinIO S3 Error: {exc}")
            return (None, None)
        except Exception as e:
            print(f"An unexpected error occurred during image loading: {e}")
            return (None, None)

# A dictionary that maps class names to class objects
NODE_CLASS_MAPPINGS = {
    "MinioUploader": MinioUploader,
    "MinioImageLoader": MinioImageLoader
}

# A dictionary that maps class names to display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinioUploader": "MinIO Image Uploader",
    "MinioImageLoader": "MinIO Image Loader"
}
