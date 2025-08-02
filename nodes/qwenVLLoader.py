import os
import requests
import folder_paths
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenVLModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                    ],
                    {"default": "Qwen/Qwen2.5-VL-7B-Instruct"},
                ),
                "quantization": (["none", "4bit", "8bit"], {"default": "8bit"}),
            },
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "ArtUtils/VL"

    def load_model(self, model, quantization):
        model_directory = os.path.join(folder_paths.models_dir, "VLM")
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download first using the original DownloadAndLoadQwen2_5_VLModel node.")
        
        # For AWQ models, ignore quantization setting (already pre-quantized)
        if "AWQ" in model:
            print(f"Loading AWQ model (pre-quantized): {model}")
            quantization_config = None
        else:
            # Apply quantization for non-AWQ models
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None

        model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            quantization_config=quantization_config,
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        return ({"model": model_obj, "processor": processor, "model_path": model_path},)





class QwenVLInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": ("INT", {"default": 256, "min": 64, "max": 1280}),
                "max_pixels": ("INT", {"default": 1280, "min": 64, "max": 2048}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("results", "boolean_list")
    FUNCTION = "process_batch"
    CATEGORY = "ArtUtils/VL"

    def process_batch(self, model, images, prompt, max_tokens, min_pixels, max_pixels):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        
        qwen_model = model["model"]
        processor = model["processor"]
        
        results = []
        boolean_results = []
        
        print(f"Processing {images.shape[0]} images")
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            try:
                print(f"Processing image {i+1}/{images.shape[0]}")
                
                # Convert tensor back to PIL Image
                image_array = (images[i] * 255).cpu().numpy().astype(np.uint8)
                pil_image = Image.fromarray(image_array)
                print(f"Image {i+1} converted to PIL, size: {pil_image.size}")
                
                # Prepare message content
                content = [
                    {
                        "type": "image",
                        "image": pil_image,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt}
                ]
                
                messages = [{"role": "user", "content": content}]
                
                # Apply chat template
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                print(f"Image {i+1} chat template applied")
                
                # Process vision info
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                print(f"Image {i+1} vision info processed")
                
                # Prepare inputs
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
                inputs = inputs.to(qwen_model.device)
                print(f"Image {i+1} inputs prepared and moved to device")
                
                # Generate response
                generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                result = output_text[0].strip()
                print(f"Image {i+1} result: {result}")
                
                # Extract boolean from result
                if "TRUE" in result.upper():
                    boolean_results.append("TRUE")
                elif "FALSE" in result.upper():
                    boolean_results.append("FALSE")
                else:
                    boolean_results.append("UNKNOWN")
                
                results.append(f"Image {i+1}: {result}")
                
            except Exception as e:
                error_msg = f"Image {i+1}: Error - {str(e)}"
                print(error_msg)
                results.append(error_msg)
                boolean_results.append("ERROR")
        
        final_result = "\n\n".join(results)
        boolean_list = str(boolean_results)  # Convert to string like [TRUE, FALSE, TRUE]
        
        print(f"Final combined result: {final_result}")
        print(f"Boolean list: {boolean_list}")
        
        return (final_result, boolean_list)





NODE_CLASS_MAPPINGS = {
    "QwenVLModelLoader": QwenVLModelLoader,
    "QwenVLInference": QwenVLInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLModelLoader": "Qwen VL Model Loader",
    "QwenVLInference": "Qwen VL Inference",
}