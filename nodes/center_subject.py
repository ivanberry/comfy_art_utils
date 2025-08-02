"""
Center Subject Node for ComfyUI - Simple and Direct Approach
"""

import torch
import numpy as np
from PIL import Image
import cv2

class CenterSubject:
    """Center the main subject in white background images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["contour", "threshold"], {"default": "contour"}),
                "threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("centered_images", "info")
    FUNCTION = "center_subject"
    CATEGORY = "Image/Transform"
    
    def find_subject_contour(self, image_cv):
        """使用轮廓检测找到主体"""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    
    def find_subject_threshold(self, image_cv, threshold):
        """使用阈值检测找到主体"""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        mask = gray < threshold
        
        if not np.any(mask):
            return None
            
        coords = np.where(mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    def center_subject(self, images, method, threshold):
        try:
            centered_images = []
            info_list = []
            
            for i, image_tensor in enumerate(images):
                # 转换为numpy和OpenCV格式
                image_np = image_tensor.cpu().numpy()
                image_cv = (image_np * 255).astype(np.uint8)
                h, w = image_cv.shape[:2]
                
                # 检测主体位置
                if method == "contour":
                    bbox = self.find_subject_contour(image_cv)
                else:
                    bbox = self.find_subject_threshold(image_cv, threshold)
                
                if bbox is None:
                    info_list.append(f"Image {i+1}: No subject found")
                    centered_images.append(image_tensor)
                    continue
                
                x, y, bbox_w, bbox_h = bbox
                
                # 计算主体中心和画布中心的偏移
                subject_center_x = x + bbox_w // 2
                subject_center_y = y + bbox_h // 2
                canvas_center_x = w // 2
                canvas_center_y = h // 2
                
                offset_x = canvas_center_x - subject_center_x
                offset_y = canvas_center_y - subject_center_y
                
                # 创建白色背景
                centered_image = np.full_like(image_cv, 255)
                
                # 计算源图像和目标位置
                src_x1, src_y1 = max(0, -offset_x), max(0, -offset_y)
                src_x2, src_y2 = min(w, w - offset_x), min(h, h - offset_y)
                dst_x1, dst_y1 = max(0, offset_x), max(0, offset_y)
                dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
                
                # 只复制非白色区域
                src_region = image_cv[src_y1:src_y2, src_x1:src_x2]
                mask = np.all(src_region >= 240, axis=2)  # 白色区域掩码
                
                # 复制非白色像素
                centered_image[dst_y1:dst_y2, dst_x1:dst_x2][~mask] = src_region[~mask]
                
                # 转换回tensor
                centered_tensor = torch.from_numpy(centered_image.astype(np.float32) / 255.0)[None,]
                centered_images.append(centered_tensor)
                
                info_list.append(f"Image {i+1}: Offset ({offset_x}, {offset_y}), Subject at ({subject_center_x}, {subject_center_y})")
            
            result_images = torch.cat(centered_images, dim=0)
            return (result_images, "; ".join(info_list))
            
        except Exception as e:
            print(f"[CenterSubject ERROR] {e}")
            return (images, f"Error: {e}")

# Node mappings
NODE_CLASS_MAPPINGS = {
    "CenterSubject": CenterSubject,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CenterSubject": "🎯 Center Subject",
}
