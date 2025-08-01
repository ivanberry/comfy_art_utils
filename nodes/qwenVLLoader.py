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
                "model_path": ("STRING", {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "8bit"}),
            },
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen2.5-VL/Simplified"

    def load_model(self, model_path, quantization):
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            quantization_config=quantization_config,
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        return ({"model": model, "processor": processor, "model_path": model_path},)





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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)
    FUNCTION = "process_batch"
    CATEGORY = "Qwen2.5-VL/Simplified"

    def process_batch(self, model, images, prompt, max_tokens, min_pixels, max_pixels):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        
        qwen_model = model["model"]
        processor = model["processor"]
        
        results = []
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            try:
                # Convert tensor back to PIL Image
                image_array = (images[i] * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array)
                
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
                
                # Process vision info
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                
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
                
                results.append(f"Image {i+1}: {output_text[0]}")
                
            except Exception as e:
                results.append(f"Image {i+1}: Error - {str(e)}")
        
        return ("\n\n".join(results),)





NODE_CLASS_MAPPINGS = {
    "QwenVLModelLoader": QwenVLModelLoader,
    "QwenVLInference": QwenVLInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLModelLoader": "Qwen2.5-VL Model Loader",
    "QwenVLInference": "Qwen2.5-VL Inference",
}