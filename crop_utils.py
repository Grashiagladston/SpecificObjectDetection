from PIL import Image

def crop_detected_objects(image, detections, target_object):
    """
    Crops the detected objects from the original image based on bounding boxes.
    Returns: List of tuples (cropped_image, label, box_2d)
    """
    width, height = image.size
    cropped_images_data = []
    target_object_lower = target_object.strip().lower()
    
    for obj in detections.objects:
        # Check if it matches the target
        is_match = (
            obj.is_target or 
            (target_object_lower in obj.label.lower()) or 
            (obj.label.lower() in target_object_lower)
        )
        
        if is_match:
            ymin, xmin, ymax, xmax = obj.box_2d
            
            # Scale coordinates (0-1000) to actual pixels
            left = int((xmin / 1000.0) * width)
            top = int((ymin / 1000.0) * height)
            right = int((xmax / 1000.0) * width)
            bottom = int((ymax / 1000.0) * height)
            
            # Crop the image
            # Box = (left, upper, right, lower)
            crop = image.crop((left, top, right, bottom))
            
            cropped_images_data.append({
                "image": crop,
                "label": obj.label,
                "box_2d": obj.box_2d
            })
            
    return cropped_images_data