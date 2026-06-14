"""
DoodleBook — ZeroGPU Version (Free HF Hosting)

Runs directly on HF ZeroGPU without Modal.
Slower but completely free.
"""

import gradio as gr
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from config import FLUX_MODEL, STORY_MODEL, GENERATION_PARAMS, BASE_SEED


# ============================================================================
# ZEROGPU INFERENCE (No Modal)
# ============================================================================

@torch.inference_mode()
def generate_story_zerogpu(hero_name: str, theme: str, age: int = 5) -> dict:
    """Generate story using MiniCPM5-1B on ZeroGPU."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_id = STORY_MODEL.hub_id
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda().eval()
    
    prompt = f"""Write a 6-page children's storybook for age {age} about {hero_name} with theme: {theme}.
Return ONLY valid JSON:
{{"title": "Title", "character_description": "Description", "pages": [{{"page": 1, "text": "Text", "scene": "Scene"}}]}}"""
    
    inputs = tok(prompt, return_tensors="pt").cuda()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, do_sample=False)
    
    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON
    import re, json
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    # Fallback
    return {
        "title": f"{hero_name}'s Adventure",
        "character_description": f"A friendly character named {hero_name}",
        "pages": [{"page": i+1, "text": f"Page {i+1} text", "scene": f"Scene {i+1}"} for i in range(6)]
    }


@torch.inference_mode()
def generate_images_zerogpu(character_desc: str, scenes: list) -> list:
    """Generate images using FLUX on ZeroGPU."""
    from diffusers import FluxPipeline
    
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL.hub_id,
        torch_dtype=torch.bfloat16
    ).cuda()
    
    images = []
    for i, scene in enumerate(scenes):
        prompt = f"{character_desc}, {scene}, crayon drawing style"
        generator = torch.Generator("cuda").manual_seed(BASE_SEED + i)
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            width=768,
            height=512,
            generator=generator
        ).images[0]
        
        images.append(image)
    
    return images


# ============================================================================
# MAIN FUNCTION (ZeroGPU compatible)
# ============================================================================

def create_book_zerogpu(doodle_image, character_name, theme, hero_name, tiny_mode=False):
    """
    Book creation without Modal.
    Uses ZeroGPU for inference.
    """
    import time
    from book_builder import build_book_html
    import io, base64
    
    # Generate story
    story = generate_story_zerogpu(hero_name, theme)
    title = story.get("title", "Story")
    pages = story.get("pages", [])
    char_desc = story.get("character_description", "")
    scenes = [p.get("scene", "") for p in pages]
    texts = [p.get("text", "") for p in pages]
    
    # Generate images
    images = generate_images_zerogpu(char_desc, scenes)
    
    # Convert to bytes
    img_bytes = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes.append(buf.getvalue())
    
    # Build HTML
    html = build_book_html(img_bytes, texts, title)
    
    return html, f"Complete: {title}", None, None


# ============================================================================
# GRADIO UI
# ============================================================================

if __name__ == "__main__":
    with gr.Blocks(title="DoodleBook (Free)") as demo:
        gr.Markdown("# 📚 DoodleBook (ZeroGPU Version)")
        
        with gr.Row():
            with gr.Column():
                doodle = gr.Image(label="Doodle", type="numpy")
                name = gr.Textbox(label="Character name")
                theme = gr.Dropdown(["brave adventure", "making a friend"], label="Theme")
                btn = gr.Button("Make book!")
            
            with gr.Column():
                output = gr.HTML()
                status = gr.Textbox(label="Status")
        
        btn.click(
            create_book_zerogpu,
            inputs=[doodle, name, theme, name],
            outputs=[output, status]
        )
    
    demo.launch()
