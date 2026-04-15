from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import uvicorn
import io
import sys
import requests
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str
    style: str = "cartoon"
    strength: float = 0.7

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(req: ImageRequest):
    # 1. Base64 decode
    if ',' in req.image:
        img_b64 = req.image.split(',')[1]
    else:
        img_b64 = req.image

    try:
        image_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((512, 512))   # HF API-ի համար 512x512-ը օպտիմալ է
    except Exception as e:
        return {
            "status": "error",
            "result": req.image,
            "error": f"Invalid image: {e}"
        }

    # 2. Prompts
    prompts = {
        "cartoon": "cartoon style, colorful, children's book illustration, white background, simple, cute, high quality",
        "3d": "3D render, Pixar style, soft lighting, cute character, white background, high quality",
        "watercolor": "watercolor painting, soft artistic colors, gentle textures, high quality",
        "comic": "comic book style, bold lines, pop art, vibrant, high quality",
        "abstract": "abstract art, geometric shapes, colorful, modern, high quality",
        "animation": "anime style, vibrant colors, cute character, high quality"
    }
    prompt = prompts.get(req.style, prompts["cartoon"])

    try:
        print(f"🚀 Using Hugging Face public API (img2img) with style '{req.style}'", file=sys.stderr)

        # Hugging Face public img2img (աշխատում է առանց token-ի)
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Content-Type": "application/json"}

        payload = {
            "inputs": {
                "image": img_b64,
                "prompt": prompt,
                "strength": req.strength,
                "guidance_scale": 7.5,
                "num_inference_steps": 25
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        if response.status_code != 200:
            raise Exception(f"HF API error {response.status_code}: {response.text}")

        result_bytes = response.content
        result_b64 = base64.b64encode(result_bytes).decode()
        return {
            "status": "ok",
            "result": f"data:image/png;base64,{result_b64}",
            "style": req.style
        }

    except Exception as e:
        error_msg = str(e)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback",
            "error_details": f"{type(e).__name__}: {error_msg[:500]}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)