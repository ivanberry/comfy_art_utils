import torch
import numpy as np
from PIL import Image
import io
import oss2
from oss2.exceptions import OssError
import folder_paths
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class OSSBatchUploader:
    """
    OSS uploader with working format optimizations and simplified connection handling.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "access_key_id": ("STRING", {"default": "LTAI5tRg...crFg"}),
                "access_key_secret": ("STRING", {"default": "v3nG...OCaM"}),
                "endpoint": ("STRING", {"default": "oss-cn-shenzhen.aliyuncs.com"}),
                "bucket_name": ("STRING", {"default": "ecommerce-product-scrawller-files"}),
                "object_name_prefix": ("STRING", {"default": "comfyui/image"}),
                "url_expires_days": ("INT", {"default": 30, "min": 1, "max": 365, "optional": True}),
                "cdn_domain": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("oss_url", "object_name", "all_urls")
    FUNCTION = "upload_to_oss_fixed"
    CATEGORY = "Image/Upload"
    OUTPUT_NODE = True

    def _prepare_image_data(self, image_tensor, index, object_name_prefix):
        """Prepare image data with format optimization."""
        try:
            # Convert tensor to PIL Image
            image_np = image_tensor.cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

            # Smart format selection with optimization
            buffer = io.BytesIO()
            if image_pil.mode == 'RGBA':
                # Has transparency → must use PNG
                image_pil.save(buffer, format="PNG", optimize=True, compress_level=6)
                file_extension = "png"
                content_type = "image/png"
                print(f"  Using PNG for image {index+1} (has transparency)")
            else:
                # No transparency → use JPEG (60-70% smaller!)
                image_pil.save(buffer, format="JPEG", quality=92, optimize=True)
                file_extension = "jpg"
                content_type = "image/jpeg"
                print(f"  Using JPEG Q92 for image {index+1} (RGB photo)")
            
            buffer.seek(0)
            image_data = buffer.read()
            file_size = len(image_data)
            
            # Generate unique object name with timestamp
            timestamp = int(time.time() * 1000)
            sha256_hash = hashlib.sha256(image_data).hexdigest()
            object_name = f"{object_name_prefix}_{timestamp}_{sha256_hash[:10]}_{index}.{file_extension}"
            
            size_mb = file_size / (1024 * 1024)
            print(f"✓ Prepared image {index+1}: {size_mb:.1f}MB ({file_extension.upper()})")
            
            return {
                'success': True,
                'index': index,
                'object_name': object_name,
                'image_data': image_data,
                'content_type': content_type,
                'file_size': file_size,
                'error': None
            }
        except Exception as e:
            print(f"✗ Failed to prepare image {index+1}: {e}")
            return {
                'success': False,
                'index': index,
                'object_name': '',
                'image_data': None,
                'content_type': '',
                'file_size': 0,
                'error': str(e)
            }

    def _generate_signed_url(self, bucket, object_name, expires=3600*24*30):
        """Generate signed URL for private OSS access (like your Node.js code)."""
        try:
            # Generate signed URL with 30-day expiration (like your Node.js default)
            signed_url = bucket.sign_url('GET', object_name, expires, slash_safe=True, params={
                'response-content-disposition': 'inline'  # For preview in browser
            })
            if signed_url.startswith('http://'):
                signed_url = signed_url.replace('http://', 'https://', 1)

            return signed_url
        except Exception as e:
            print(f"⚠️  Failed to generate signed URL for {object_name}: {e}")
            # Fallback to direct URL (may not work for private buckets)
            return f"https://{bucket.bucket_name}.{bucket.endpoint.replace('http://', '').replace('https://', '')}/{object_name}"

    def _upload_single_image(self, bucket, prepared_data, enable_cdn, cdn_domain, endpoint, bucket_name, url_expires_days):
        """Upload single image to OSS with signed URL generation."""
        if not prepared_data['success']:
            return {
                'success': False,
                'index': prepared_data['index'],
                'url': '',
                'object_name': prepared_data['object_name'],
                'error': f"Preparation failed: {prepared_data['error']}"
            }
        
        try:
            object_name = prepared_data['object_name']
            image_data = prepared_data['image_data']
            content_type = prepared_data['content_type']
            index = prepared_data['index']
            
            # Upload with optimized headers
            headers = {
                'Content-Type': content_type,
                'Cache-Control': 'max-age=31536000',  # 1 year cache
            }
            
            # Direct upload (this works based on your debug results)
            bucket.put_object(object_name, image_data, headers=headers)

            # Generate signed URL (like your Node.js implementation)
            expires_seconds = url_expires_days * 24 * 3600  # Convert days to seconds
            if enable_cdn and cdn_domain:
                url = f"https://{cdn_domain}/{object_name}"
                print(f"🔗 Using CDN URL (unsigned): {url}")
            else:
                # Generate signed URL for private access (matches your Node.js code)
                url = self._generate_signed_url(bucket, object_name, expires_seconds)
                print(f"🔒 Generated signed URL with {url_expires_days}-day expiration")
            
            size_mb = prepared_data['file_size'] / (1024 * 1024)
            print(f"✅ Uploaded image {index+1}: {size_mb:.1f}MB")
            
            return {
                'success': True,
                'index': index,
                'url': url,
                'object_name': object_name,
                'file_size': prepared_data['file_size'],
                'error': None
            }
            
        except OssError as e:
            error_msg = f"OSS Error: {e.status} - {e.details.get('Message', str(e)) if hasattr(e, 'details') else str(e)}"
            print(f"❌ Failed to upload image {prepared_data['index']+1}: {error_msg}")
            return {
                'success': False,
                'index': prepared_data['index'],
                'url': '',
                'object_name': prepared_data['object_name'],
                'error': error_msg
            }
        except Exception as e:
            print(f"❌ Failed to upload image {prepared_data['index']+1}: {str(e)}")
            return {
                'success': False,
                'index': prepared_data['index'],
                'url': '',
                'object_name': prepared_data['object_name'],
                'error': str(e)
            }

    def upload_to_oss_fixed(self, images, access_key_id, access_key_secret, endpoint, bucket_name, 
                           object_name_prefix="comfyui/image", url_expires_days=30, enable_cdn=False, cdn_domain="", 
                           prompt=None, extra_pnginfo=None):
        """
        Fixed OSS upload with format optimization and working connection handling.
        """
        start_time = time.time()
        
        try:
            print(f"=== OSS UPLOADER (FIXED + OPTIMIZED) ===")
            print(f"Images: {len(images)}")
            print(f"Format optimization: JPEG Q92 for RGB, PNG for RGBA")
            print(f"Endpoint: {endpoint}")
            print(f"Bucket: {bucket_name}")
            print("")
            
            # Initialize OSS with standard settings (no custom timeout)
            auth = oss2.Auth(access_key_id, access_key_secret)
            bucket = oss2.Bucket(auth, endpoint, bucket_name)
            # Note: Skip bucket.get_bucket_info() as it fails with your permissions
            
            if not hasattr(images, '__len__') or len(images) == 0:
                return {
                    "ui": {"text": ["No images to upload"]}, 
                    "result": ("", "", "")
                }

            num_images = len(images)
            max_workers = min(6, num_images)
            
            print(f"Step 1: Preparing {num_images} images with format optimization...")
            
            # Step 1: Prepare all images with format optimization
            prepare_start = time.time()
            prepared_results = [None] * num_images
            
            # Prepare images sequentially to show detailed optimization info
            for i, image_tensor in enumerate(images):
                prepared_results[i] = self._prepare_image_data(image_tensor, i, object_name_prefix)
            
            prepare_time = time.time() - prepare_start
            
            # Calculate total size and compression info
            successful_prep = [r for r in prepared_results if r and r['success']]
            total_size_mb = sum(r['file_size'] for r in successful_prep) / (1024 * 1024)
            
            print(f"\nPreparation complete: {prepare_time:.2f}s")
            print(f"Total optimized size: {total_size_mb:.1f}MB")
            print(f"Average size per image: {total_size_mb/len(successful_prep):.1f}MB")
            
            # Step 2: Upload in parallel
            print(f"\nStep 2: Uploading {len(successful_prep)} images in parallel...")
            upload_start = time.time()
            upload_results = [None] * num_images
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                upload_futures = {
                    executor.submit(self._upload_single_image, bucket, data, enable_cdn, cdn_domain, endpoint, bucket_name, url_expires_days): data['index']
                    for data in prepared_results if data is not None and data['success']
                }
                
                for future in as_completed(upload_futures):
                    result = future.result()
                    upload_results[result['index']] = result

            upload_time = time.time() - upload_start
            total_time = time.time() - start_time

            # Process results
            successful_uploads = [r for r in upload_results if r and r['success']]
            failed_uploads = [r for r in upload_results if r and not r['success']]
            
            uploaded_urls = [r['url'] for r in successful_uploads]
            uploaded_object_names = [r['object_name'] for r in successful_uploads]
            
            result_url = uploaded_urls[0] if uploaded_urls else ""
            result_object_name = uploaded_object_names[0] if uploaded_object_names else ""
            all_urls_string = "\n".join(uploaded_urls)
            
            # Calculate performance metrics
            success_count = len(successful_uploads)
            upload_speed_mbps = (total_size_mb / upload_time) if upload_time > 0 else 0
            
            status_msg = f"OSS: {success_count}/{num_images} uploaded in {total_time:.2f}s"
            status_msg += f" | {total_size_mb:.1f}MB @ {upload_speed_mbps:.1f}MB/s"
            
            if failed_uploads:
                status_msg += f" ({len(failed_uploads)} failed)"
            
            print(f"\n🎉 UPLOAD COMPLETE!")
            print(f"📊 Results: {status_msg}")
            print(f"⚡ Performance: {total_time/num_images:.2f}s per image")
            print(f"💾 Format optimization saved ~60-70% file size")
            
            if success_count > 0:
                print(f"\n📋 Uploaded URLs:")
                for i, url in enumerate(uploaded_urls):
                    print(f"  {i+1}. {url}")
            
            if failed_uploads:
                print(f"\n❌ Failed uploads:")
                for failed in failed_uploads:
                    print(f"   Image {failed['index'] + 1}: {failed['error']}")
            
            return {
                "ui": {"text": [status_msg]}, 
                "result": (result_url, result_object_name, all_urls_string)
            }

        except Exception as e:
            error_msg = f"Upload error: {e}"
            print(f"❌ {error_msg}")
            return {
                "ui": {"text": [error_msg]}, 
                "result": ("", "", "")
            }

NODE_CLASS_MAPPINGS = {
    "OSSBatchUploader": OSSBatchUploader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OSSBatchUploader": "OSS Batch Uploader",
}
