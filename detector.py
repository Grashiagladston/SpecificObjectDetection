import sys
import requests
import base64
import io
import json
import time
import numpy as np
from PIL import Image
from openai import OpenAI
from config import HF_TOKEN, OPENROUTER_API_KEY

# Ensure stdout/stderr handle unicode characters on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

class DetectedObject:
    def __init__(self, label, is_target, box_2d, score=1.0):
        self.label = label
        self.is_target = is_target
        self.box_2d = box_2d
        self.score = score

    def __repr__(self):
        return f"DetectedObject(label='{self.label}', is_target={self.is_target}, box_2d={self.box_2d}, score={self.score})"

class DetectionResult:
    def __init__(self, objects):
        self.objects = objects

    def __repr__(self):
        return f"DetectionResult(objects={self.objects})"

class ObjectDetector:
    def __init__(self):
        self.detection_url = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
        self.hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    def _pil_to_bytes(self, image):
        buffered = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        return buffered.getvalue()
        
    def _pil_to_base64(self, image):
        return base64.b64encode(self._pil_to_bytes(image)).decode()
    
    def detect(self, image, target_object, max_retries=2):
        """Try HuggingFace first, fallback to OpenRouter Gemini/Llama"""
        
        # TRY 1: HuggingFace DETR
        try:
            raw_objects = self._detect_hf(image, target_object, max_retries)
            wrapped_objects = [
                DetectedObject(
                    label=obj["label"],
                    is_target=obj["is_target"],
                    box_2d=obj["box_2d"],
                    score=obj.get("score", 1.0)
                ) for obj in raw_objects
            ]
            return DetectionResult(wrapped_objects)
        except Exception as e:
            print(f"⚠️ HuggingFace Detection Failed ({str(e)[:30]}). Falling back to OpenRouter...")
            
        # TRY 2: OpenRouter Vision Fallback
        try:
            raw_objects = self._detect_openrouter(image, target_object)
            wrapped_objects = [
                DetectedObject(
                    label=obj.get("label", target_object),
                    is_target=obj.get("is_target", True),
                    box_2d=obj.get("box_2d") or obj.get("bbox"),
                    score=obj.get("score", 1.0)
                ) for obj in raw_objects
            ]
            return DetectionResult(wrapped_objects)
        except Exception as e:
            print(f"❌ OpenRouter Detection Failed ({str(e)}).")
            return DetectionResult([])

    def _detect_hf(self, image, target_object, max_retries):
        img_bytes = self._pil_to_bytes(image)
        img_width, img_height = image.size
        
        for attempt in range(max_retries):
            response = requests.post(self.detection_url, headers=self.hf_headers, data=img_bytes, timeout=15)
            
            if response.status_code == 200:
                results = response.json()
                return self._format_hf_results(results, target_object, img_width, img_height)
            elif response.status_code == 503:
                wait_time = response.json().get("estimated_time", 20)
                time.sleep(wait_time)
            else:
                raise Exception(f"HF Error: {response.status_code}")
        raise Exception("HF Model failed to load")

    def _format_hf_results(self, results, target_object, img_width, img_height):
        objects = []
        for detection in results:
            label = detection.get("label", "")
            is_target = target_object.lower() in label.lower()
            
            box = detection.get("box", {})
            xmin = int((box.get("xmin", 0) / img_width) * 1000)
            ymin = int((box.get("ymin", 0) / img_height) * 1000)
            xmax = int((box.get("xmax", 0) / img_width) * 1000)
            ymax = int((box.get("ymax", 0) / img_height) * 1000)
            
            objects.append({
                "label": label,
                "is_target": is_target,
                "score": detection.get("score", 0),
                "box_2d": [ymin, xmin, ymax, xmax]
            })
        return [obj for obj in objects if obj["is_target"]]

    def _detect_openrouter(self, image, target_object):
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "sk-placeholder":
            raise Exception("HuggingFace blocked AND OpenRouter key is missing!")
            
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        img_b64 = self._pil_to_base64(image)
        
        # Prompt for JSON formatting
        prompt = f"""
        Identify and locate all instances of '{target_object}' in this image. For each object:
        1. "label": object name
        2. "is_target": true if it matches '{target_object}', else false
        3. "box_2d": [ymin, xmin, ymax, xmax] normalized 0-1000 scale
        
        Return ONLY this JSON format:
        {{"objects": [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "name", "is_target": true}}]}}
        If none found: {{"objects": []}}
        """
        
        models = [
            "openrouter/free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "google/gemini-2.5-flash",
            "google/gemini-2.0-flash-001"
        ]
        
        last_exception = None
        for model in models:
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ],
                    "temperature": 0.1
                }
                
                # Exclude response_format for Llama or free/router models to prevent compatibility errors
                if not any(x in model.lower() for x in ["llama", "free", "openrouter"]):
                    kwargs["response_format"] = {"type": "json_object"}
                    
                response = client.chat.completions.create(**kwargs)
                
                content = response.choices[0].message.content
                if not content:
                    raise Exception("Model returned empty content")
                content_stripped = content.strip()
                
                # Robust extraction of JSON substring
                first_brace = content_stripped.find('{')
                last_brace = content_stripped.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = content_stripped[first_brace:last_brace+1]
                else:
                    json_str = content_stripped
                    
                data = json.loads(json_str)
                objects = data.get("objects", [])
                for obj in objects:
                    obj["bbox"] = obj.get("box_2d", [])
                    if "score" not in obj:
                        obj["score"] = 1.0
                print(f"✅ Used OpenRouter {model} for Detection")
                return objects
            except Exception as e:
                print(f"⚠️ OpenRouter model {model} failed: {e}")
                last_exception = e
                continue
                
        raise Exception(f"All OpenRouter models failed. Last error: {last_exception}")

