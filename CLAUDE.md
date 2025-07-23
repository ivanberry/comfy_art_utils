# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a ComfyUI custom nodes collection focused on image loading functionality:

```
comfyui-custom-nodes/
├── __init__.py              # Main entry point with node mappings
├── nodes/                   # Custom node implementations
│   └── url_image_loader.py  # URL image loader with custom headers
└── requirements.txt         # Python dependencies
```

## Architecture

This repository follows the standard ComfyUI custom node pattern:

- **Node Registration**: `__init__.py` imports and registers all custom nodes via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
- **Node Implementation**: Each node in `nodes/` directory implements:
  - `INPUT_TYPES()` classmethod defining input parameters
  - `RETURN_TYPES` and `RETURN_NAMES` for output specification
  - Main execution function specified by `FUNCTION` attribute
  - `CATEGORY` for UI organization

## Current Nodes

### URLImageLoader (`nodes/url_image_loader.py`)
Loads images from URLs with custom HTTP headers support:
- **Inputs**: URL, referer, user-agent, timeout
- **Output**: Image tensor in ComfyUI format
- **Features**: Custom headers, error handling, RGB conversion, tensor formatting

## Dependencies

Key dependencies defined in `requirements.txt`:
- `requests>=2.25.0` - HTTP requests with header support
- `Pillow>=8.0.0` - Image processing and format conversion
- `torch>=1.9.0` - Tensor operations for ComfyUI compatibility
- `numpy>=1.21.0` - Array operations

## Development Notes

- No build scripts, linting, or test configuration found
- Standard Python package structure for ComfyUI integration
- Nodes return tensors in ComfyUI's expected format: `[batch, height, width, channels]` with values normalized to 0-1 range
- Error handling uses exceptions that ComfyUI can display to users