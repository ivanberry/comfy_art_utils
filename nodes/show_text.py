class ShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "show_text"
    OUTPUT_NODE = True
    CATEGORY = "ArtUtils/Text"

    def show_text(self, text, prompt=None, extra_pnginfo=None):
        print(f"ShowText received: {text} (type: {type(text)})")
        # Convert to string if not already
        text_str = str(text) if text is not None else ""
        
        return {
            "ui": {
                "text": [text_str]
            }, 
            "result": (text_str,)
        }


NODE_CLASS_MAPPINGS = {
    "ShowText": ShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShowText": "Show Text",
}