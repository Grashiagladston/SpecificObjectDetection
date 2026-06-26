from PIL import Image, ImageDraw, ImageFont

def draw_annotations(image, detections, target_object):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = annotated_image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        font = ImageFont.load_default()
        
    target_count = 0
    target_object_lower = target_object.strip().lower()
    boxes_list = []
    box_color = "#00F0FF"
    
    for obj in detections.objects:
        is_match = (
            obj.is_target or 
            (target_object_lower in obj.label.lower()) or 
            (obj.label.lower() in target_object_lower)
        )
        
        if is_match:
            target_count += 1
            ymin, xmin, ymax, xmax = obj.box_2d
            
            boxes_list.append({
                "index": target_count,
                "label": obj.label,
                "box_2d": obj.box_2d
            })
            
            left = (xmin / 1000.0) * width
            top = (ymin / 1000.0) * height
            right = (xmax / 1000.0) * width
            bottom = (ymax / 1000.0) * height
            
            draw.rectangle([left, top, right, bottom], outline=box_color, width=5)
            label_text = f"{obj.label.upper()} #{target_count}"
            
            if font:
                left_t, top_t, right_t, bottom_t = font.getbbox(label_text)
                text_w = right_t - left_t
                text_h = bottom_t - top_t
            else:
                text_w, text_h = len(label_text) * 8, 12
                
            text_offset_y = top - text_h - 6
            if text_offset_y < 0:
                text_offset_y = top + 6
                
            draw.rectangle([left, text_offset_y, left + text_w + 12, text_offset_y + text_h + 6], fill=box_color)
            draw.text((left + 6, text_offset_y + 3), label_text, fill="black", font=font)
            
    return annotated_image, target_count, boxes_list