from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import requests
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # 1. Decode base64 (հեռացնել data:image/png;base64,)
    if ',' in req.image:
        img_b64 = req.image.split(',')[1]
    else:
        img_b64 = req.image

    # 2. Prompt-ներ
    prompts = {
        "cartoon": "cartoon style, colorful, children's book illustration, white background, simple, cute",
        "3d": "3D render, Pixar style, soft lighting, cute character, white background",
        "watercolor": "watercolor painting, soft artistic colors, gentle textures",
        "comic": "comic book style, bold lines, pop art, vibrant",
        "abstract": "abstract art, geometric shapes, colorful, modern",
        "animation": "anime style, vibrant colors, cute character"
    }
    prompt = prompts.get(req.style, prompts["cartoon"])

    # 3. Hugging Face Inference API (img2img աջակցող մոդել)
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": {
            "image": img_b64,               # միայն base64, առանց prefix-ի
            "prompt": prompt,
            "strength": req.strength,
            "guidance_scale": 7.5,
            "num_inference_steps": 25
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        if response.status_code != 200:
            error_detail = response.text
            raise Exception(f"HF API error {response.status_code}: {error_detail}")

        # API-ն կարող է վերադարձնել JSON կամ ուղղակի bytes
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            result_json = response.json()
            # Որոշ մոդելներ վերադարձնում են {"generated_image": "base64..."}
            if "generated_image" in result_json:
                img_bytes = base64.b64decode(result_json["generated_image"])
            else:
                raise Exception(f"Unexpected JSON response: {result_json}")
        else:
            img_bytes = response.content

        result_b64 = base64.b64encode(img_bytes).decode()
        return {
            "status": "ok",
            "result": f"data:image/png;base64,{result_b64}",
            "style": req.style
        }

    except Exception as e:
        # Fallback + մանրամասն սխալ (կտեսնես frontend-ում)
        error_msg = str(e)
        print(f"Fallback triggered: {error_msg}")
        return {
            "status": "ok",
            "result": req.image,
            "style": req.style,
            "note": "fallback",
            "error_details": error_msg[:300]   # կարճ տարբերակը
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)