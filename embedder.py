import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

def get_embedding_model():
    if not hasattr(get_embedding_model, 'model'):
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.eval()
        get_embedding_model.model = model
    return get_embedding_model.model

def generate_embedding(pil_image):
    model = get_embedding_model()
    
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
        
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        
    embedding_vector = output.squeeze().numpy().tolist()
    magnitude = sum(x**2 for x in embedding_vector) ** 0.5
    
    if magnitude > 0:
        embedding_vector = [x / magnitude for x in embedding_vector]
        
    return embedding_vector