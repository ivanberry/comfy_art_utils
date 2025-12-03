"""
Center Subject Node for ComfyUI - Enhanced with Largest Object Detection v2.0
"""

import torch
import numpy as np
import cv2

class CenterSubject:
    """Center the main subject in white background images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["contour", "threshold", "largest_object"], {"default": "largest_object"}),
                "threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "min_area": ("INT", {"default": 1000, "min": 100, "max": 50000}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("centered_images", "info")
    FUNCTION = "center_subject"
    CATEGORY = "Image/Transform"
    
    def _mask_info_from_bool(self, mask: np.ndarray, min_area: int):
        """Given a boolean mask, return bbox, centroid and area if it meets min_area."""
        area = int(mask.sum())
        if area < min_area:
            return None
        
        coords = np.column_stack(np.nonzero(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        bbox = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
        centroid_y, centroid_x = coords.mean(axis=0)
        return mask, bbox, (float(centroid_x), float(centroid_y)), area
    
    def find_subject_contour(self, image_cv, min_area=1000):
        """使用轮廓检测找到主体"""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 过滤太小的轮廓，找到最大的有效轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            return None
            
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # 计算掩码和质心，质心比bbox中心更稳定
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 1, cv2.FILLED)
        
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = x + (w - 1) / 2.0
            cy = y + (h - 1) / 2.0
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (mask.astype(bool), (x, y, w, h), (float(cx), float(cy)), int(cv2.contourArea(largest_contour)))
    
    def find_subject_threshold(self, image_cv, threshold, min_area=1):
        """使用阈值检测找到主体"""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        mask = gray < threshold
        
        if not np.any(mask):
            return None
        
        return self._mask_info_from_bool(mask, min_area)
    
    def find_largest_object(self, image_cv, threshold=240, min_area=1000):
        """找到最大的非白色连通区域"""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作去除噪声
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:  # 只有背景
            return None
        
        # 找最大的连通区域（排除背景）
        largest_area = 0
        largest_idx = -1
        
        for i in range(1, num_labels):  # 跳过背景(0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area and area > largest_area:
                largest_area = area
                largest_idx = i
        
        if largest_idx == -1:
            return None
            
        # 返回边界框
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        
        mask = labels == largest_idx
        centroid = centroids[largest_idx]  # (x, y)
        return (mask, (x, y, w, h), (float(centroid[0]), float(centroid[1])), int(largest_area))
    
    def center_subject(self, images, method, threshold, min_area, debug):
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
                    bbox = self.find_subject_contour(image_cv, min_area)
                elif method == "largest_object":
                    bbox = self.find_largest_object(image_cv, threshold, min_area)
                else:
                    bbox = self.find_subject_threshold(image_cv, threshold, min_area)
                
                if bbox is None:
                    info_list.append(f"Image {i+1}: No subject found")
                    centered_images.append(image_tensor)
                    continue
                
                _mask, (x, y, bbox_w, bbox_h), (cx, cy), area = bbox
                
                # 计算主体中心和画布中心的偏移（使用质心更稳定）
                canvas_center_x = (w - 1) / 2.0
                canvas_center_y = (h - 1) / 2.0
                
                offset_x = canvas_center_x - cx
                offset_y = canvas_center_y - cy
                
                # 平移整张图，边界填充白色，减少局部复制带来的像素误差
                transform = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                centered_image = cv2.warpAffine(
                    image_cv,
                    transform,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderValue=(255, 255, 255),
                )
                
                # 转换回tensor
                centered_tensor = torch.from_numpy(centered_image.astype(np.float32) / 255.0)[None,]
                centered_images.append(centered_tensor)
                
                info_text = (
                    f"Image {i+1}: Method={method}, BBox=({x},{y},{bbox_w},{bbox_h}), "
                    f"Centroid=({cx:.2f},{cy:.2f}), Offset=({offset_x:.2f},{offset_y:.2f})"
                )
                if debug:
                    info_text += f", Area={area}"
                info_list.append(info_text)
            
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
    "CenterSubject": "🎯 Center Subject v2",
}
