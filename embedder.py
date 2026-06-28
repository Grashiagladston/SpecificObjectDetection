import sys
import os
import requests
import base64
import io
import json
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
            import time
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            image,
                            "Describe the visual features, colors, shapes, objects, textures, layout, and important details of this image for accurate image matching."
                        ],
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "Quota" in str(e) or "limit" in str(e).lower():
                        if attempt < 2:
                            time.sleep(2 * (attempt + 1))
                            continue
                    raise e

            description = response.text

            if description:
                # Generate embedding from description
                for attempt in range(3):
                    try:
                        embedding_response = client.models.embed_content(
                            model="gemini-embedding-2",
                            contents=description,
                        )
                        break
                    except Exception as e:
                        if "429" in str(e) or "Quota" in str(e) or "limit" in str(e).lower():
                            if attempt < 2:
                                time.sleep(2 * (attempt + 1))
                                continue
                        raise e
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
                "google/gemma-4-31b-it:free",
                "google/gemma-4-26b-a4b-it:free",
                "nvidia/nemotron-nano-12b-v2-vl:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "google/gemini-2.0-flash-001",
                "google/gemini-2.5-flash"
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

def generate_embeddings_batch(images, target_object=None):
    if not images:
        return []
        
    # Try 1: Gemini via official google-genai SDK (Batched)
    if HAS_GENAI and GEMINI_API_KEY:
        try:
            client = get_gemini_client()
            
            # Prepare the contents list for Gemini
            contents = []
            for i, img in enumerate(images):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                contents.append(f"Image {i+1}:")
                contents.append(img)
            
            prompt = (
                "Describe the visual features, colors, shapes, objects, textures, layout, and important details of each image listed above, in order, for accurate image matching. "
                "Return a JSON object with a 'descriptions' key containing a list of strings, one description for each image in the exact order they were provided."
            )
            contents.append(prompt)
            
            from google.genai import types
            import time
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "Quota" in str(e) or "limit" in str(e).lower():
                        if attempt < 2:
                            time.sleep(3 * (attempt + 1))
                            continue
                    raise e
            
            res_text = response.text
            if res_text:
                data = json.loads(res_text.strip())
                descriptions = data.get("descriptions", [])
                
                if len(descriptions) == len(images):
                    for attempt in range(3):
                        try:
                            embedding_response = client.models.embed_content(
                                model="gemini-embedding-2",
                                contents=descriptions,
                            )
                            break
                        except Exception as e:
                            if "429" in str(e) or "Quota" in str(e) or "limit" in str(e).lower():
                                if attempt < 2:
                                    time.sleep(2 * (attempt + 1))
                                    continue
                            raise e
                    print(f"✅ Used Gemini GenAI for Batch Embedding ({len(images)} images)")
                    return [emb.values for emb in embedding_response.embeddings]
                else:
                    print(f"⚠️ Gemini GenAI Batch returned {len(descriptions)} descriptions for {len(images)} images. Falling back...")
        except Exception as e:
            print(f"⚠️ Gemini GenAI Batch Embedding Failed ({str(e)[:50]}). Falling back...")
            
    # Fallback: Generate one by one
    results = []
    for i, img in enumerate(images):
        if i > 0:
            import time
            time.sleep(1.5)
        try:
            emb = generate_embedding(img, target_object)
            results.append(emb)
        except Exception as e:
            print(f"⚠️ Individual embedding failed: {e}")
            results.append(None)
    return results

def cosine_similarity(embedding1, embedding2):
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(e1, e2) / (norm1 * norm2))
