from google import genai
from PIL import Image
from config import GEMINI_API_KEY


def get_gemini_client():
    if not hasattr(get_gemini_client, "client"):
        get_gemini_client.client = genai.Client(api_key=GEMINI_API_KEY)
    return get_gemini_client.client


def generate_embedding(pil_image):
    """
    Generate an embedding for an image using Gemini.
    The image is first described by Gemini, then the description
    is converted into an embedding.
    """

    client = get_gemini_client()

    # Ensure RGB format
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Generate description of image
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            pil_image,
            "Describe the visual features, colors, shapes, objects, textures, layout, and important details of this image for accurate image matching."
        ],
    )

    description = response.text

    # Generate embedding from description
    embedding_response = client.models.embed_content(
        model="text-embedding-004",
        contents=description,
    )

    return embedding_response.embeddings[0].values
