from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import uvicorn
import io
import sys
from PIL import Image
import requests

# ---------- ՆԱԽ ՍԱՀՄԱՆԵԼ ENV-Ը, ՀԵՏՈ ՆԵՐՄՈՒԾԵԼ REPLICATE ----------
REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN", "")
if REPLICATE_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
    print(f"✅ REPLICATE_API_TOKEN set. Length: {len(REPLICATE_TOKEN)}, prefix: {REPLICATE_TOKEN[:8]}...", file=sys.stderr)
else:
    print("❌ REPLICATE_TOKEN environment variable is NOT SET!", file=sys.stderr)

# replicate-ը ներմուծում ենք token-ը սահմանելուց հետո
import replicate

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
    return {
        "status": "ok",
        "replicate_token_present": bool(REPLICATE_TOKEN),
        "replicate_token_prefix": REPLICATE_TOKEN[:8] + "..." if REPLICATE_TOKEN else "none"
    }

@app.post("/generate")
async def generate(req: ImageRequest):
    # 1. Decode base64
    if ',' in req.image:
        img_b64 = req.image.split(',')[1]
    else:
        img_b64 = req.image

    try:
        image_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((512, 512))
    except Exception as e:
        return {
            "status": "error",
            "result": req.image,
            "error": f"Invalid image: {e}"
        }

    # 2. Prompts
    prompts = {
        "cartoon": "cartoon style, colorful, children's book illustration, white background, simple, cute",
        "3d": "3D render, Pixar style, soft lighting, cute character, white background",
        "watercolor": "watercolor painting, soft artistic colors, gentle textures",
        "comic": "comic book style, bold lines, pop art, vibrant",
        "abstract": "abstract art, geometric shapes, colorful, modern",
        "animation": "anime style, vibrant colors, cute character"
    }
    prompt = prompts.get(req.style, prompts["cartoon"])

    if not REPLICATE_TOKEN:
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback",
            "error_details": "REPLICATE_TOKEN not set"
        }

    try:
        print(f"🚀 Calling Replicate with style '{req.style}', strength={req.strength}", file=sys.stderr)

        input_image_uri = f"data:image/png;base64,{img_b64}"
        model_id = "stability-ai/stable-diffusion-img2img:527d2e262f7f45a04c9b2ef8df6c1d6c5c3f3a7e1d1b5f7c8e9d0a1b2c3d4e5f6"

        output = replicate.run(
            model_id,
            input={
                "image": input_image_uri,
                "prompt": prompt,
                "strength": req.strength,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "negative_prompt": "realistic, scary, dark, violent, complex background, blurry, low quality"
            }
        )

        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        else:
            result_url = output

        if hasattr(result_url, 'url'):
            response = requests.get(result_url.url)
        else:
            response = requests.get(str(result_url))

        if response.status_code != 200:
            raise Exception(f"Failed to download result image: HTTP {response.status_code}")

        result_b64 = base64.b64encode(response.content).decode()
        return {
            "status": "ok",
            "result": f"data:image/png;base64,{result_b64}",
            "style": req.style
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Replicate call failed: {error_msg}", file=sys.stderr)
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback",
            "error_details": f"{type(e).__name__}: {error_msg[:500]}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)