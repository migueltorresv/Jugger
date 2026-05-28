import io
import base64
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

pipe = None
MODEL_PATH = "/mnt/models/juggernaut-xl-v9"

# Prefijo de prompt optimizado para reconstrucción 3D
SYSTEM_PROMPT_PREFIX = (
    "single object, studio lighting, dramatic shadows, physically based rendering, "
    "subsurface scattering, depth of field, centered, white background, "
    "hyperdetailed photography, 8k, "
)

# --- Modelos de request/response compatibles con el formato Gemini ---

class Part(BaseModel):
    text: str

class Content(BaseModel):
    parts: List[Part]

class GenerateRequest(BaseModel):
    contents: List[Content]


def load_model():
    global pipe
    if pipe is not None:
        return

    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    logger.info("Cargando modelo Juggernaut XL v9...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )

    # Scheduler rápido: DPM++ 2M Karras
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )

    pipe.to("cuda")
    logger.info("Modelo cargado correctamente.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        # Extraer el prompt del formato Gemini
        prompt = request.contents[0].parts[0].text
    except (IndexError, AttributeError):
        raise HTTPException(status_code=400, detail="Formato de request inválido")

    # Cargar modelo la primera vez (lazy loading)
    load_model()

    full_prompt = SYSTEM_PROMPT_PREFIX + prompt

    logger.info(f"Generando imagen para prompt: {prompt}")

    image = pipe(
        prompt=full_prompt,
        width=1024,
        height=1024,
        num_inference_steps=20,
        guidance_scale=5.0,
    ).images[0]

    # Convertir a base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Respuesta en formato compatible con el parser de Unity
    return {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": ""},
                    {"inlineData": {"data": image_base64, "mimeType": "image/png"}}
                ]
            }
        }]
    }
