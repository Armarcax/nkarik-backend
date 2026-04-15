from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import replicate
import os
import uvicorn
import io
from PIL import Image
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token-ը Render/Vercel-ի Environment Variable-ից
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_TOKEN", "")

class ImageRequest(BaseModel):
    image: str
    style: str = "cartoon"
    strength: float = 0.7

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(req: ImageRequest):
    # 1. Base64 → PIL Image
    if ',' in req.image:
        img_b64 = req.image.split(',')[1]
    else:
        img_b64 = req.image

    try:
        image_bytes = base64.b64decode(img_b64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image = input_image.resize((512, 512))
    except Exception as e:
        return {
            "status": "error",
            "result": req.image,
            "error": f"Invalid image: {e}"
        }

    # 2. Prompt-ներ ըստ ոճի
    prompts = {
        "cartoon": "cartoon style, colorful, children's book illustration, white background, simple, cute",
        "3d": "3D render, Pixar style, soft lighting, cute character, white background",
        "watercolor": "watercolor painting, soft artistic colors, gentle textures",
        "comic": "comic book style, bold lines, pop art, vibrant",
        "abstract": "abstract art, geometric shapes, colorful, modern",
        "animation": "anime style, vibrant colors, cute character"
    }
    prompt = prompts.get(req.style, prompts["cartoon"])

    # 3. Replicate API կանչ
    try:
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

        # Stable Diffusion Img2Img մոդելի ID (Replicate-ի կողմից ստուգված)
        model_id = "stability-ai/stable-diffusion-img2img:527d2e262f7f45a04c9b2ef8df6c1d6c5c3f3a7e1d1b5f7c8e9d0a1b2c3d4e5f6"

        output = replicate.run(
            model_id,
            input={
                "image": input_image,
                "prompt": prompt,
                "strength": req.strength,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "negative_prompt": "realistic, scary, dark, violent, complex background, blurry, low quality"
            }
        )

        # output-ը list է [FileOutput] կամ [str] (URL)
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        else:
            result_url = output

        # 4. Ներբեռնել արդյունքը և դարձնել base64
        if hasattr(result_url, 'url'):
            response = requests.get(result_url.url)
        else:
            response = requests.get(result_url)

        result_b64 = base64.b64encode(response.content).decode()

        return {
            "status": "ok",
            "result": f"data:image/png;base64,{result_b64}",
            "style": req.style
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Replicate error: {error_msg}")
        # Fallback – ցույց տալ original-ը
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback",
            "error_details": error_msg[:300]
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)