import io
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

def get_gemini_client():
    if not hasattr(get_gemini_client, 'client'):
        get_gemini_client.client = genai.Client(api_key=GEMINI_API_KEY)
    return get_gemini_client.client

def generate_embedding(pil_image):
    client = get_gemini_client()
    
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    # Step 1: Use Gemini to describe the image (lightweight)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), 
            "Describe the visual features, colors, shapes, layout, and objects in this image in extreme detail for image matching."
        ]
    )
    description = response.text
    
    # Step 2: Embed the description (very small & fast)
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=description
    )
    
    return result.embeddings[0].values
