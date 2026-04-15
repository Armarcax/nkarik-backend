from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import replicate
import os
import uvicorn
import io
import sys
import requests
from PIL import Image

# ---------- Token-ի սահմանում import-ից առաջ ----------
REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN", "")
if REPLICATE_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
    print(f"✅ REPLICATE_TOKEN set. Prefix: {REPLICATE_TOKEN[:8]}...", file=sys.stderr)
else:
    print("❌ REPLICATE_TOKEN missing!", file=sys.stderr)

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
    # 1. Base64 decode
    if ',' in req.image:
        img_b64 = req.image.split(',')[1]
    else:
        img_b64 = req.image

    try:
        image_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((1024, 1024))  # SDXL-ի համար 1024x1024
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

    if not REPLICATE_TOKEN:
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback",
            "error_details": "REPLICATE_TOKEN not set"
        }

    try:
        print(f"🚀 Calling Replicate SDXL with style '{req.style}', strength={req.strength}", file=sys.stderr)

        # SDXL Img2Img մոդել (հաստատ աշխատում է)
        model_id = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

        input_image_uri = f"data:image/png;base64,{img_b64}"

        output = replicate.run(
            model_id,
            input={
                "image": input_image_uri,
                "prompt": prompt,
                "strength": req.strength,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "negative_prompt": "realistic, scary, dark, violent, complex background, blurry, low quality"
            }
        )

        # output-ը list է FileOutput-ներից
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
        # Տպենք ամբողջական traceback-ը Render-ի logs-ում
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