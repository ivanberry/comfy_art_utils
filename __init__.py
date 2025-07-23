"""
Comfy Art Utils
A collection of utility nodes for ComfyUI focused on art generation workflows, including URL image loader with custom headers support.
"""

from .nodes.url_image_loader import NODE_CLASS_MAPPINGS as URL_LOADER_MAPPINGS
from .nodes.url_image_loader import NODE_DISPLAY_NAME_MAPPINGS as URL_LOADER_DISPLAY_MAPPINGS
from .nodes.minio_uploader import NODE_CLASS_MAPPINGS as MINIO_MAPPINGS
from .nodes.minio_uploader import NODE_DISPLAY_NAME_MAPPINGS as MINIO_DISPLAY_MAPPINGS

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add URL Image Loader node
NODE_CLASS_MAPPINGS.update(URL_LOADER_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(URL_LOADER_DISPLAY_MAPPINGS)

# Add MinIO nodes
NODE_CLASS_MAPPINGS.update(MINIO_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MINIO_DISPLAY_MAPPINGS)

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']