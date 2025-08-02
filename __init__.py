"""
Comfy Art Utils
A collection of utility nodes for ComfyUI focused on art generation workflows, including URL image loader with custom headers support.
"""

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import core nodes (always available)
try:
    from .nodes.url_image_loader import NODE_CLASS_MAPPINGS as URL_LOADER_MAPPINGS
    from .nodes.url_image_loader import NODE_DISPLAY_NAME_MAPPINGS as URL_LOADER_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(URL_LOADER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(URL_LOADER_DISPLAY_MAPPINGS)
    print("[ComfyArtUtils] Loaded URL Image Loader")
except Exception as e:
    print(f"[ComfyArtUtils] Failed to load URL Image Loader: {e}")

try:
    from .nodes.art_utils import NODE_CLASS_MAPPINGS as ART_UTILS_MAPPINGS
    from .nodes.art_utils import NODE_DISPLAY_NAME_MAPPINGS as ART_UTILS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(ART_UTILS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ART_UTILS_DISPLAY_MAPPINGS)
    print("[ComfyArtUtils] Loaded Art Utils")
except Exception as e:
    print(f"[ComfyArtUtils] Failed to load Art Utils: {e}")

try:
    from .nodes.center_subject import NODE_CLASS_MAPPINGS as CENTER_SUBJECT_MAPPINGS
    from .nodes.center_subject import NODE_DISPLAY_NAME_MAPPINGS as CENTER_SUBJECT_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(CENTER_SUBJECT_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CENTER_SUBJECT_DISPLAY_MAPPINGS)
    print("[ComfyArtUtils] Loaded Center Subject")
except Exception as e:
    print(f"[ComfyArtUtils] Failed to load Center Subject: {e}")

try:
    from .nodes.show_text import NODE_CLASS_MAPPINGS as SHOW_TEXT_MAPPINGS
    from .nodes.show_text import NODE_DISPLAY_NAME_MAPPINGS as SHOW_TEXT_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(SHOW_TEXT_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(SHOW_TEXT_DISPLAY_MAPPINGS)
    print("[ComfyArtUtils] Loaded Show Text")
except Exception as e:
    print(f"[ComfyArtUtils] Failed to load Show Text: {e}")

# Import optional nodes (require additional dependencies)
try:
    from .nodes.oss_uploader import NODE_CLASS_MAPPINGS as OSS_UPLOADER_MAPPINGS
    from .nodes.oss_uploader import NODE_DISPLAY_NAME_MAPPINGS as OSS_UPLOADER_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(OSS_UPLOADER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(OSS_UPLOADER_DISPLAY_MAPPINGS)
    print("[ComfyArtUtils] Loaded OSS Uploader")
except Exception as e:
    print(f"[ComfyArtUtils] Failed to load OSS Uploader (missing oss2?): {e}")

try:
    from .nodes.qwenVLLoader import NODE_CLASS_MAPPINGS as QWEN_VL_MAPPINGS
    from .nodes.qwenVLLoader import NODE_DISPLAY_NAME_MAPPINGS as QWEN_VL_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(QWEN_VL_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(QWEN_VL_DISPLAY_MAPPINGS)
    print("[ComfyArtUtils] Loaded Qwen VL")
except Exception as e:
    print(f"[ComfyArtUtils] Failed to load Qwen VL (missing dependencies?): {e}")

print(f"[ComfyArtUtils] Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes")

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']