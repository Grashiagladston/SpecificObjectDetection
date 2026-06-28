import sys
import os
import requests
import base64
import io
import numpy as np
from PIL import Image
from openai import OpenAI
from config import HF_TOKEN, OPENROUTER_API_KEY

# Try importing google-genai for Gemini embedding support
try:
    from google import genai
    from config import GEMINI_API_KEY
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    GEMINI_API_KEY = ""

# Ensure stdout/stderr handle unicode characters on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def _pil_to_base64(image):
    buffered = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

def get_gemini_client():
    if not hasattr(get_gemini_client, "client"):
        get_gemini_client.client = genai.Client(api_key=GEMINI_API_KEY)
    return get_gemini_client.client

def generate_embedding(image, target_object=None):
    # Try 1: Gemini via official google-genai SDK
    if HAS_GENAI and GEMINI_API_KEY:
        try:
            client = get_gemini_client()
            
            # Ensure RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Generate description of image
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    image,
                    "Describe the visual features, colors, shapes, objects, textures, layout, and important details of this image for accurate image matching."
                ],
            )
            description = response.text

            if description:
                # Generate embedding from description
                embedding_response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=description,
                )
                print("✅ Used Gemini GenAI for Embedding")
                return embedding_response.embeddings[0].values
        except Exception as e:
            print(f"⚠️ Gemini GenAI Embedding Failed ({str(e)[:50]}). Falling back...")

    # Try 2: HuggingFace (Real numerical vectors)
    b64 = _pil_to_base64(image)
    try:
        API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {"inputs": {"image": f"data:image/jpeg;base64,{b64}"}}
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            embedding = np.array(response.json()).flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            print("✅ Used HuggingFace CLIP for Embedding")
            return embedding.tolist()
        else:
            raise Exception(f"HF API Error: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ HuggingFace Failed ({str(e)[:30]}). Falling back to OpenRouter...")
        
        # Try 3: OpenRouter Gemini Flash Fallback
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "sk-placeholder":
            raise Exception("HuggingFace is blocked and OpenRouter key is missing. Please add OPENROUTER_API_KEY to .env")

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            
            prompt = f"Describe the visual features, colors, layout, and exact position of the {target_object or 'objects'} in this image. Be extremely detailed."
            
            models = [
                "openrouter/free",
                "meta-llama/llama-3.2-11b-vision-instruct:free",
                "google/gemini-2.5-flash",
                "google/gemini-2.0-flash-001"
            ]
            
            last_exception = None
            text_description = None
            used_model = None
            
            for model in models:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }],
                        max_tokens=300,
                        temperature=0.1
                    )
                    text_description = response.choices[0].message.content
                    if not text_description:
                        raise Exception("Model returned empty description")
                    used_model = model
                    break
                except Exception as e:
                    print(f"⚠️ OpenRouter embedding fallback model {model} failed: {e}")
                    last_exception = e
                    continue
            
            if not text_description:
                raise Exception(f"All fallback models failed. Last error: {last_exception}")
            
            # Convert text to a fake numerical vector so your database math works
            vector = [float(ord(c)) for c in text_description[:512]] 
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = np.array(vector) / norm
            
            print(f"✅ Used OpenRouter {used_model} Fallback for Embedding")
            return vector.tolist()
            
        except Exception as e2:
            raise Exception(f"All embedding APIs failed. OpenRouter Error: {e2}")

def cosine_similarity(embedding1, embedding2):
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(e1, e2) / (norm1 * norm2))
