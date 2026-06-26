from google import genai
from google.genai import types
from pydantic import BaseModel
from config import GEMINI_API_KEY

class ObjectDetection(BaseModel):
    box_2d: list[int]
    label: str
    is_target: bool

class DetectionResult(BaseModel):
    objects: list[ObjectDetection]

class GeminiDetector:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def detect(self, image, target_object):
        prompt = f"""
        Identify and locate all instances of '{target_object}' in this image. For each object, specify:
        1. Its label name classification under `label`.
        2. Whether it matches the target '{target_object}' under `is_target`.
        3. Its bounding box `box_2d` in [ymin, xmin, ymax, xmax] normalized format (0-1000 scale).
        Be very thorough.
        """
        
        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DetectionResult,
            )
        )
        return response.parsed
