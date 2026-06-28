from PIL import Image
from detector import ObjectDetector
from embedder import generate_embedding, cosine_similarity

# Test with a sample image
print("Creating test image...")
test_img = Image.new("RGB", (400, 400), color="white")

print("\n--- Testing Detection ---")
detector = ObjectDetector()
try:
    results = detector.detect(test_img, "cat")
    print(f"Detections: {results}")
except Exception as e:
    print(f"Detection error: {e}")

print("\n--- Testing Embedding ---")
try:
    embedding = generate_embedding(test_img)
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"Embedding error: {e}")

print("\nDone!")