import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
import folder_paths

# --------------------------------------------------------------------------------
# Helper function to find LLM models
# --------------------------------------------------------------------------------
def get_llm_models():
    """Scans the 'models/LLM' directory for models and returns a list of sub-folders."""
    try:
        # Try to get the LLM folder path
        llms_dir = folder_paths.get_folder_paths("LLM")
        print(f"LLM models: Found registered LLM paths: {', '.join(llms_dir) if llms_dir else 'None'}")
    except KeyError:
        # If 'LLM' folder type doesn't exist, try to find it manually
        try:
            base_models_dir = folder_paths.get_folder_paths("checkpoints")[0]  # Get models folder
            llm_path = os.path.join(os.path.dirname(base_models_dir), "LLM")
            llms_dir = [llm_path]
            print(f"LLM models: Using fallback path: {llm_path}")
            
            # If that doesn't exist either, return empty list
            if not os.path.exists(llm_path):
                print(f"LLM models directory not found at {llm_path}. Please create 'models/LLM' folder.")
                return ["No LLM models found - create models/LLM folder"]
        except Exception as e:
            print(f"LLM models: Error finding LLM directory: {e}")
            return [f"Error: Cannot locate LLM directory - {str(e)}"]
    
    if not llms_dir:
        return ["No LLM models found"]
    
    models = []
    for base_path in llms_dir:
        if os.path.isdir(base_path):
            try:
                for item in os.listdir(base_path):
                    # We assume a model is a directory containing a config.json
                    item_path = os.path.join(base_path, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
                        # To make the name prettier in the dropdown, we show the relative path
                        # from the base 'LLM' directory.
                        relative_path = os.path.relpath(item_path, base_path)
                        models.append(relative_path)
            except Exception as e:
                print(f"Error scanning LLM models directory: {e}")
                continue
    
    if not models:
        return ["No LLM models found - add models to models/LLM folder"]
    
    return models

# --------------------------------------------------------------------------------
# The Main Node Class
# --------------------------------------------------------------------------------
class LLMLoaderAndGenerator:
    # --- Class-level cache for models, tokenizers, and processors ---
    _models = {}
    _tokenizers = {}
    _processors = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_llm_models(), ),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe what you see in this image."
                }),
                "max_tokens": ("INT", {"default": 150, "min": 10, "max": 4096, "step": 1}),
                # Add a seed for reproducibility
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "LLM Nodes" # This will be the category in the ComfyUI menu

    def generate(self, model_name, prompt, max_tokens, seed, image=None):
        # Handle error messages from get_llm_models
        if model_name.startswith("No LLM models found"):
            return ([f"Error: {model_name}. Please add LLM models to your models/LLM directory."],)
        
        # --- Determine the compute device ---
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"LLM Node: Using device '{device}'")
        
        # --- Set seed for reproducibility ---
        torch.manual_seed(seed)

        # --- Find the full path to the model ---
        try:
            model_path = folder_paths.get_full_path("LLM", model_name)
            print(f"LLM Node: folder_paths.get_full_path returned: {model_path}")
        except Exception as e:
            print(f"LLM Node: folder_paths.get_full_path failed: {e}")
            model_path = None
        
        if not model_path:
            # Fallback: construct path manually
            try:
                base_models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                models_base = os.path.dirname(base_models_dir)
                model_path = os.path.join(models_base, "LLM", model_name)
                print(f"LLM Node: Fallback path constructed: {model_path}")
            except Exception as e:
                print(f"LLM Node: Fallback path construction failed: {e}")
                return ([f"Error: Cannot construct path to model {model_name}. Error: {str(e)}"],)
        
        print(f"LLM Node: Final model path: {model_path}")
        print(f"LLM Node: Path exists: {os.path.exists(model_path)}")
        
        if not model_path or not os.path.exists(model_path):
            # List what's actually in the LLM directory for debugging
            try:
                base_models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                llm_dir = os.path.join(os.path.dirname(base_models_dir), "LLM")
                if os.path.exists(llm_dir):
                    available_models = os.listdir(llm_dir)
                    # Filter to only show directories (actual models)
                    model_dirs = [item for item in available_models if os.path.isdir(os.path.join(llm_dir, item))]
                    return ([f"Error: Model {model_name} not found. Available models in LLM directory: {', '.join(model_dirs) if model_dirs else 'None'}"],)
                else:
                    return ([f"Error: LLM directory not found at {llm_dir}"],)
            except:
                return ([f"Error: Model {model_name} not found. Please check 'models/LLM' directory."],)

        # --- Load model, tokenizer, and processor from cache or disk ---
        if model_name in self._models:
            print(f"LLM Node: Loading '{model_name}' from cache.")
            model = self._models[model_name]
            tokenizer = self._tokenizers[model_name]
            processor = self._processors.get(model_name, None)
        else:
            print(f"LLM Node: Loading '{model_name}' from disk. This may take a moment...")
            # Check model type and load appropriate components
            model_is_phi3_vision = "phi-3-vision" in model_name.lower() or "phi3-vision" in model_name.lower()
            
            try:
                # Try to load as vision-language model first
                if model_is_phi3_vision:
                    print("LLM Node: Detected Phi-3 Vision model, loading with special handling...")
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                    print("LLM Node: Loaded Phi-3 Vision model with processor.")
                else:
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    # Check if processor actually has vision capabilities
                    if hasattr(processor, 'image_processor') or hasattr(processor, 'feature_extractor'):
                        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                        print("LLM Node: Loaded as vision-language model with processor.")
                    else:
                        # Processor exists but no vision capabilities
                        processor = None
                        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                        print("LLM Node: Processor found but no vision capabilities. Loaded as text-only model.")
            except Exception as e:
                # Fallback to text-only model
                processor = None
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"LLM Node: Failed to load processor ({e}). Loaded as text-only model.")
            
            # Model-specific loading optimizations for MPS/CPU
            if model_is_phi3_vision:
                print("LLM Node: Loading Phi-3 Vision with MPS/CPU optimizations...")
                
                # Load config first and disable flash attention
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                
                # Force disable flash attention in config
                if hasattr(config, '_attn_implementation'):
                    config._attn_implementation = "eager"
                if hasattr(config, 'attn_implementation'):
                    config.attn_implementation = "eager"
                if hasattr(config, '_flash_attn_2_enabled'):
                    config._flash_attn_2_enabled = False
                if hasattr(config, 'use_flash_attention_2'):
                    config.use_flash_attention_2 = False
                
                print("LLM Node: Disabled flash attention in model config.")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=device,
                    attn_implementation="eager"  # Use standard attention (works on MPS/CPU)
                )
                print("LLM Node: Loaded Phi-3 Vision with standard attention for MPS.")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=device,
                    attn_implementation="eager"  # Explicitly use eager attention
                )
            print("LLM Node: Model loaded successfully.")
            
            # --- Store in cache for future use ---
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
            self._processors[model_name] = processor

        # --- Process batch images or single text input ---
        results = []
        
        if image is not None:
            if processor is not None:
                # Vision-language model with image input - process each image in batch
                print(f"LLM Node: Processing batch of {image.shape[0]} images with vision-language model...")
                batch_size = image.shape[0]
            else:
                # Text-only model but image provided - warn user and process text only
                print(f"LLM Node: Warning - Images provided but {model_name} is a text-only model. Processing text prompt only.")
                print("LLM Node: To use images, please use a vision-language model like Qwen-VL, LLaVA, or similar.")
                batch_size = 1  # Process once with text only
            
            for i in range(batch_size):
                if processor is not None:
                    print(f"LLM Node: Processing image {i+1}/{batch_size}")
                    
                    # Convert ComfyUI image tensor to PIL Image
                    # ComfyUI images are in format [batch, height, width, channels] with values 0-1
                    img_array = (image[i].cpu().numpy() * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_array)
                    
                    # Model-specific message formatting
                    model_is_phi3_vision = "phi-3-vision" in model_name.lower() or "phi3-vision" in model_name.lower()
                    
                    try:
                        if model_is_phi3_vision:
                            # Phi-3 Vision specific format using tokenizer directly
                            print("LLM Node: Using Phi-3 Vision format")
                            
                            # Format the text prompt with image placeholder (keep it concise)
                            # Truncate very long prompts to avoid huge input_ids
                            truncated_prompt = prompt[:200] if len(prompt) > 200 else prompt
                            formatted_prompt = f"<|image_1|>\n{truncated_prompt}"
                            
                            # Use tokenizer directly for text processing
                            messages = [{"role": "user", "content": formatted_prompt}]
                            text_inputs = tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt"
                            )
                            
                            # Create attention mask for the text inputs
                            attention_mask = torch.ones_like(text_inputs)
                            
                            # Process image separately
                            image_inputs = processor.image_processor(pil_image, return_tensors="pt")
                            
                            # Combine text and image inputs
                            model_inputs = {
                                "input_ids": text_inputs,
                                "attention_mask": attention_mask,
                                "images": image_inputs["pixel_values"]
                            }
                        else:
                            # Standard vision-language format
                            messages = [
                                {
                                    "role": "user", 
                                    "content": [
                                        {"type": "image", "image": pil_image},
                                        {"type": "text", "text": prompt}
                                    ]
                                }
                            ]
                            
                            model_inputs = processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt"
                            )
                    except Exception as e1:
                        print(f"LLM Node: Primary format failed: {e1}")
                        try:
                            # Format 2: Direct processor call with text and images
                            if model_is_phi3_vision:
                                # For Phi-3 Vision, use the processor's __call__ method
                                model_inputs = processor(
                                    images=pil_image,
                                    text=f"<|image_1|>\n{prompt}",
                                    return_tensors="pt"
                                )
                            else:
                                model_inputs = processor(
                                    text=prompt,
                                    images=pil_image,
                                    return_tensors="pt"
                                )
                        except Exception as e2:
                            print(f"LLM Node: Simple format failed: {e2}")
                            try:
                                # Format 3: Basic processor format
                                model_inputs = processor(
                                    prompt,
                                    images=pil_image,
                                    return_tensors="pt"
                                )
                            except Exception as e3:
                                print(f"LLM Node: All vision formats failed. Errors: {e1}, {e2}, {e3}")
                                # Fallback to text-only processing
                                model_inputs = tokenizer.apply_chat_template(
                                    [{"role": "user", "content": prompt}],
                                    add_generation_prompt=True,
                                    return_tensors="pt"
                                )
                else:
                    # Text-only model - process text prompt only
                    print(f"LLM Node: Processing text-only prompt (ignoring image)")
                    model_inputs = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                
                # Move to device
                if isinstance(model_inputs, dict):
                    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                else:
                    model_inputs = model_inputs.to(device)
                
                # Generate text for this image
                try:
                    # Model-specific generation parameters
                    gen_kwargs = {
                        "max_new_tokens": max_tokens,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    
                    # Add Phi-3 Vision specific parameters
                    model_is_phi3_vision = "phi-3-vision" in model_name.lower() or "phi3-vision" in model_name.lower()
                    if model_is_phi3_vision:
                        # Simplified parameters for Phi-3 Vision to avoid hanging
                        # Limit max_new_tokens for initial testing
                        limited_tokens = min(max_tokens, 50)  # Start with shorter responses
                        gen_kwargs.update({
                            "max_new_tokens": limited_tokens,
                            "do_sample": False,  # Use greedy decoding for stability
                            "num_beams": 1,      # Single beam for speed
                            "use_cache": False,  # Disable cache to avoid potential issues
                            "eos_token_id": tokenizer.eos_token_id,
                            "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
                            # Removed early_stopping as it's not valid for this model
                        })
                        print(f"LLM Node: Using optimized Phi-3 Vision generation parameters (limited to {limited_tokens} tokens)")
                    else:
                        # Standard parameters for other models
                        gen_kwargs.update({
                            "do_sample": True,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "repetition_penalty": 1.1,
                        })
                    
                    if isinstance(model_inputs, dict):
                        # Vision-language model with dict inputs
                        if model_is_phi3_vision and 'images' in model_inputs:
                            # For Phi-3 Vision, the model expects pixel_values, not images
                            model_inputs_copy = model_inputs.copy()
                            if 'images' in model_inputs_copy:
                                model_inputs_copy['pixel_values'] = model_inputs_copy.pop('images')
                            
                            # Ensure all tensors are on the correct device
                            for key, tensor in model_inputs_copy.items():
                                if isinstance(tensor, torch.Tensor):
                                    model_inputs_copy[key] = tensor.to(device)
                                    print(f"LLM Node: Moved {key} to {device}, shape: {tensor.shape}")
                            
                            print(f"LLM Node: Phi-3 Vision inputs: {list(model_inputs_copy.keys())}")
                            print(f"LLM Node: Input shapes - input_ids: {model_inputs_copy['input_ids'].shape}, pixel_values: {model_inputs_copy['pixel_values'].shape}")
                            if 'attention_mask' in model_inputs_copy:
                                print(f"LLM Node: Attention mask shape: {model_inputs_copy['attention_mask'].shape}")
                            print(f"LLM Node: Generation parameters: {gen_kwargs}")
                            print(f"LLM Node: Starting generation with timeout monitoring...")
                            
                            import time
                            start_time = time.time()
                            generated_ids = model.generate(
                                **model_inputs_copy,
                                **gen_kwargs
                            )
                            end_time = time.time()
                            generation_time = end_time - start_time
                            print(f"LLM Node: Generation completed in {generation_time:.2f} seconds")
                            
                            # Performance warning for MPS
                            if device == "mps" and generation_time > 30:
                                print("⚠️  LLM Node: PERFORMANCE WARNING!")
                                print("   Phi-3 Vision is running very slowly on MPS (Apple Silicon).")
                                print("   Consider using:")
                                print("   1. A smaller vision model (like Qwen-VL-Chat)")
                                print("   2. Quantized model versions") 
                                print("   3. CPU inference might actually be faster")
                                print(f"   Current speed: {generation_time/limited_tokens:.2f}s per token")
                        else:
                            # Standard vision model format
                            generated_ids = model.generate(
                                **model_inputs,
                                **gen_kwargs
                            )
                        
                        # For vision models, get the input length from input_ids
                        if 'input_ids' in model_inputs:
                            input_length = model_inputs['input_ids'].shape[-1]
                        else:
                            # Fallback: assume generated sequence starts from beginning
                            input_length = 0
                    else:
                        # Text-only model with tensor input
                        generated_ids = model.generate(
                            model_inputs,
                            **gen_kwargs
                        )
                        input_length = model_inputs.shape[-1]

                    # Decode the generated tokens, skipping the prompt part
                    if input_length > 0:
                        completion_ids = generated_ids[0][input_length:]
                    else:
                        completion_ids = generated_ids[0]
                    
                    # Use the appropriate tokenizer/processor for decoding
                    if hasattr(processor, 'decode') and processor is not None:
                        generated_text = processor.decode(completion_ids, skip_special_tokens=True)
                    elif hasattr(processor, 'tokenizer') and processor is not None:
                        generated_text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
                    else:
                        generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    
                    results.append(generated_text)
                    
                except Exception as gen_error:
                    print(f"LLM Node: Generation failed for image {i+1}: {gen_error}")
                    results.append(f"Error processing image {i+1}: {str(gen_error)}")
                
        else:
            # Text-only model or no image provided - single generation
            print("LLM Node: Processing text-only input...")
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            # apply_chat_template will format the input correctly for the model
            model_inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            # Generate text with optimized parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            generated_ids = model.generate(
                model_inputs,
                **gen_kwargs
            )
            input_length = model_inputs.shape[-1]

            # Decode the generated tokens, skipping the prompt part
            completion_ids = generated_ids[0][input_length:]
            generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append(generated_text)
        
        print(f"LLM Node: Generation complete. Generated {len(results)} results.")
        
        # Ensure all results are strings
        results = [str(result) if result is not None else "Error: Empty result" for result in results]

        # --- Free up memory if needed (optional, depends on your VRAM) ---
        # torch.cuda.empty_cache()
        # if device == "mps":
        #    torch.mps.empty_cache()

        # The node must return a tuple with list for batch output
        return (results,)


# --------------------------------------------------------------------------------
# ComfyUI Registration
# --------------------------------------------------------------------------------
# A dictionary that maps class names to class objects
NODE_CLASS_MAPPINGS = {
    "LLMLoaderAndGenerator": LLMLoaderAndGenerator
}

# A dictionary that maps human-readable names to class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM Loader & Generator": "LLMLoaderAndGenerator"
}