from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import requests
from PIL import Image
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ՓՈԽԻՐ ՔՈ TOKEN-ՈՎ
HF_TOKEN = os.environ.get("HF_TOKEN", "")

class ImageRequest(BaseModel):
    image: str
    style: str = "cartoon"
    strength: float = 0.7

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(req: ImageRequest):
    # 1. Decode base64 image
    if ',' in req.image:
        img_b64 = req.image.split(',')[1]
    else:
        img_b64 = req.image
    image_bytes = base64.b64decode(img_b64)

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

    # 3. Hugging Face Inference API (img2img-ի համար)
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Քանի որ API-ն սպասում է multipart/form-data
    files = {"image": image_bytes}
    data = {"prompt": prompt}

    try:
        response = requests.post(API_URL, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise Exception(f"Hugging Face error {response.status_code}: {response.text}")

        # 4. Արդյունքը base64
        result_b64 = base64.b64encode(response.content).decode()
        return {
            "status": "ok",
            "result": f"data:image/png;base64,{result_b64}",
            "style": req.style
        }
    except Exception as e:
        # Fallback – եթե API-ն չի աշխատում, վերադարձնում ենք նույն նկարը
        print("Fallback used:", str(e))
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)