"""
Vision extractor for media2html.
Uses YOLOv8m for object detection and EasyOCR for text extraction.
All processing runs on CPU to leave GPU free for the text LLM.
"""

import sys
import subprocess
import numpy as np
from PIL import Image, ImageEnhance
from collections import Counter
from ..html_builder import Obj, BBox, TextRegion, Color


def non_max_suppression(objects: list, iou_threshold: float = 0.4) -> list:
    """Filters overlapping bounding boxes."""
    if not objects: return []
    objects = sorted(objects, key=lambda x: x.conf, reverse=True)
    kept = []
    for obj in objects:
        overlap = False
        for k in kept:
            xA = max(obj.bbox.x1, k.bbox.x1)
            yA = max(obj.bbox.y1, k.bbox.y1)
            xB = min(obj.bbox.x2, k.bbox.x2)
            yB = min(obj.bbox.y2, k.bbox.y2)
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (obj.bbox.x2 - obj.bbox.x1) * (obj.bbox.y2 - obj.bbox.y1)
            boxBArea = (k.bbox.x2 - k.bbox.x1) * (k.bbox.y2 - k.bbox.y1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(obj)
    return kept


def merge_same_label_boxes(objects: list, iou_threshold: float = 0.3, min_horizontal_overlap: float = 0.0) -> list:
    """Merges boxes of the same label with significant overlap or horizontal alignment."""
    if not objects: return []
    objects = sorted(objects, key=lambda x: x.conf, reverse=True)
    merged = []
    used = [False] * len(objects)
    
    for i in range(len(objects)):
        if used[i]: continue
        current = objects[i]
        
        for j in range(i + 1, len(objects)):
            if used[j]: continue
            next_obj = objects[j]
            
            if current.label == next_obj.label:
                # Calculate IoU
                xA = max(current.bbox.x1, next_obj.bbox.x1)
                yA = max(current.bbox.y1, next_obj.bbox.y1)
                xB = min(current.bbox.x2, next_obj.bbox.x2)
                yB = min(current.bbox.y2, next_obj.bbox.y2)
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (current.bbox.x2 - current.bbox.x1) * (current.bbox.y2 - current.bbox.y1)
                boxBArea = (next_obj.bbox.x2 - next_obj.bbox.x1) * (next_obj.bbox.y2 - next_obj.bbox.y1)
                iou = interArea / float(boxAArea + boxBArea - interArea)
                
                # Calculate horizontal overlap ratio (intersection / smaller box span)
                overlap_x = max(0, xB - xA)
                smaller_box_span_x = min(boxAArea, boxBArea) / max(yA, yB)  # Estimate span from area
                horizontal_overlap = overlap_x / smaller_box_span_x if smaller_box_span_x > 0 else 0
                
                # Merge if IoU > threshold OR high horizontal overlap OR ANY overlap exists
                if iou > iou_threshold or horizontal_overlap > min_horizontal_overlap or interArea > 0:
                    # Merge next_obj into current
                    current.bbox.x1 = min(current.bbox.x1, next_obj.bbox.x1)
                    current.bbox.y1 = min(current.bbox.y1, next_obj.bbox.y1)
                    current.bbox.x2 = max(current.bbox.x2, next_obj.bbox.x2)
                    current.bbox.y2 = max(current.bbox.y2, next_obj.bbox.y2)
                    used[j] = True
        merged.append(current)
        used[i] = True
        
    return merged


def detect_objects(img_path: str, conf_threshold: float = 0.25):
    """Uses YOLO-World for open-vocabulary detection with SAHI for small objects."""
    from ultralytics import YOLOWorld
    from PIL import Image
    from ..html_builder import Obj, BBox
    import sahi

    classes = ["car", "person", "umbrella", "gate", "tree", "building", "house", "guardrail", "streetlight", "puddle"]

    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(classes)

    im = Image.open(img_path)
    W, H = im.size

    # Use SAHI for slicing if image is large (4K+)
    if W * H > 1920 * 1080:  # 1080p threshold
        print("Using SAHI for sliced inference on large image")

        def prediction_with_sahi():
            result = sahi.predict.get_sliced_prediction(
                image_path=img_path,
                model=model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                conf_threshold=conf_threshold,
                iou_threshold=0.3,
                verbose=False
            )

            # Merge boxes from all slices
            all_boxes = []
            for prediction in result.object_predictions_list:
                for obj in prediction:
                    x1, y1, x2, y2 = obj.bbox.to_xyxy()
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    label = classes[int(obj.category_id)]
                    conf = float(obj.score)
                    all_boxes.append(Obj(label, BBox(x1/W, y1/H, x2/W, y2/H), conf))

            # Apply NMS and merge same-label boxes
            filtered = non_max_suppression(all_boxes)
            return merge_same_label_boxes(filtered)

        return prediction_with_sahi()
    else:
        # Original inference for smaller images
        res = model(img_path, verbose=False, conf=conf_threshold, iou=0.3)

        out = []
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = classes[int(box.cls)]
                conf = float(box.conf)
                out.append(Obj(label, BBox(x1/W, y1/H, x2/W, y2/H), conf))

        # Apply Python NMS, then merge same-label stacks
        filtered = non_max_suppression(out)
        return merge_same_label_boxes(filtered)

def extract_ocr(img_path: str, min_conf=0.3):
    """Extract text regions using RapidOCR (CPU-only to avoid VRAM contention)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        
        result, _ = engine(img_path)
        
        if result is None:
            return []
        
        im = Image.open(img_path)
        W, H = im.size
        out = []
        
        for item in result:
            # RapidOCR returns: [bbox, (text, conf)] or [bbox, text, conf]
            if len(item) == 2:
                bbox, (text, conf) = item
            elif len(item) == 3:
                bbox, text, conf = item
            else:
                continue
            
            # Convert conf to float if it's a string
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                continue
            
            if conf < min_conf:
                continue
            
            # Normalize coordinates
            x1 = min(p[0] for p in bbox) / W
            y1 = min(p[1] for p in bbox) / H
            x2 = max(p[0] for p in bbox) / W
            y2 = max(p[1] for p in bbox) / H
            
            out.append(TextRegion(text, BBox(x1, y1, x2, y2)))
        
        # Group adjacent lines into paragraphs
        out = group_ocr_blocks(out)
        
        return out
        
    except ImportError:
        print("Warning: RapidOCR not installed. Falling back to EasyOCR.")
        return extract_ocr_easyocr(img_path, min_conf)
    except Exception as e:
        print(f"Warning: OCR extraction failed ({e}). Skipping.")
        return []


def extract_ocr_easyocr(img_path: str, min_conf=0.5):
    """Fallback OCR using EasyOCR if RapidOCR fails."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(img_path)
        
        im = Image.open(img_path)
        W, H = im.size
        out = []
        
        for (bbox, text, conf) in result:
            if conf < min_conf:
                continue
            
            # Normalize coordinates
            x1 = min(p[0] for p in bbox) / W
            y1 = min(p[1] for p in bbox) / H
            x2 = max(p[0] for p in bbox) / W
            y2 = max(p[1] for p in bbox) / H
            
            out.append(TextRegion(text, BBox(x1, y1, x2, y2)))
        
        # Group adjacent lines into paragraphs
        out = group_ocr_blocks(out)
        
        return out
        
    except Exception as e:
        print(f"Warning: EasyOCR fallback failed ({e}). Skipping.")
        return []


def group_ocr_blocks(text_regions: list, vertical_threshold: float = 0.02) -> list:
    """Groups adjacent OCR lines into paragraph blocks based on y-axis proximity.
    
    Merges text lines that are vertically close together into coherent blocks,
    reducing token count for document/UI images with many text lines.
    """
    if not text_regions:
        return []
    
    # Sort by top y-coordinate
    sorted_regions = sorted(text_regions, key=lambda t: t.bbox.y1)
    
    blocks = []
    current_block = [sorted_regions[0]]
    
    for i in range(1, len(sorted_regions)):
        prev = sorted_regions[i-1]
        curr = sorted_regions[i]
        
        # If the current line is close to the previous line, group them
        if curr.bbox.y1 - prev.bbox.y2 < vertical_threshold:
            current_block.append(curr)
        else:
            blocks.append(current_block)
            current_block = [curr]
            
    blocks.append(current_block)
    
    # Convert blocks back to TextRegion objects with merged bboxes and text
    merged_regions = []
    for block in blocks:
        text = " ".join([t.text for t in block])
        x1 = min(t.bbox.x1 for t in block)
        y1 = min(t.bbox.y1 for t in block)
        x2 = max(t.bbox.x2 for t in block)
        y2 = max(t.bbox.y2 for t in block)
        merged_regions.append(TextRegion(text, BBox(x1, y1, x2, y2)))
        
    return merged_regions


def dominant_colors(img_path: str, k=5, resize=96):
    """Extract dominant colors from image."""
    im = Image.open(img_path).convert("RGB").resize((resize, resize))
    pixels = list(im.getdata())
    quant = [(r//24*24, g//24*24, b//24*24) for r,g,b in pixels]
    c = Counter(quant)
    total = sum(c.values())
    return [Color(f"#{r:02x}{g:02x}{b:02x}", n/total) for (r,g,b), n in c.most_common(k)]


def get_caption(objects, text_regions, colors, tags=None, layout=None):
    """
    Generate algorithmic caption without VLM.
    Creates a zero-compute summary based on detected elements.
    """
    parts = []
    
    # Object summary
    if objects:
        obj_labels = [obj.label for obj in objects[:5]]  # Top 5 objects
        parts.append(f"Scene contains: {', '.join(obj_labels)}.")
    
    # Text summary
    if text_regions:
        texts = [tr.text for tr in text_regions[:3]]  # Top 3 text regions
        parts.append(f"Text detected: '{' '.join(texts)}'.")
    
    # Color summary
    if colors:
        dominant = colors[0].hex
        parts.append(f"Dominant color: {dominant}.")
    
    # Tags summary
    if tags:
        parts.append(f"Tags: {', '.join(tags[:5])}.")
    
    # Semantic layout summary
    if layout:
        layout_str = ', '.join([f"{k} ({v:.0%})" for k, v in layout.items()])
        parts.append(f"Environment: {layout_str}.")
    
    return " ".join(parts) if parts else "No significant content detected."


def get_visual_grid(img_path: str, mode: str = "compact") -> str:
    """Imports ASCILINE core directly to get a token-optimized visual HTML grid."""
    try:
        return transcode_image_core(img_path)
    except Exception as e:
        print(f"Warning: ASCILINE transcode failed: {e}")
        return ""


def get_spatial_relations(objects: list) -> list[str]:
    """Algorithmically determines spatial relationships between objects.

    Uses x-IoU to avoid emitting "left of" / "right of" for vertically
    stacked objects whose bounding boxes overlap horizontally.
    """
    relations = []
    for i, o1 in enumerate(objects):
        for j, o2 in enumerate(objects):
            # Skip comparing the object with itself
            if i == j: continue

            c1_x, c1_y = (o1.bbox.x1 + o1.bbox.x2)/2, (o1.bbox.y1 + o1.bbox.y2)/2
            c2_x, c2_y = (o2.bbox.x1 + o2.bbox.x2)/2, (o2.bbox.y1 + o2.bbox.y2)/2

            # Horizontal containment check: if one bbox's x-range fully contains
            # the other's, they are vertically stacked — skip lateral relations.
            o1_contains_o2 = (o1.bbox.x1 <= o2.bbox.x1 and o1.bbox.x2 >= o2.bbox.x2)
            o2_contains_o1 = (o2.bbox.x1 <= o1.bbox.x1 and o2.bbox.x2 >= o1.bbox.x2)
            if o1_contains_o2 or o2_contains_o1:
                if c1_y < c2_y: relations.append(f"{o1.label} is above {o2.label}")
                if c2_y < c1_y: relations.append(f"{o2.label} is above {o1.label}")
                continue

            if c1_x < c2_x: relations.append(f"{o1.label} is left of {o2.label}")
            if c1_y < c2_y: relations.append(f"{o1.label} is above {o2.label}")

    # Remove duplicates and return top 5
    return list(set(relations))[:5]


def get_semantic_layout(img_path: str) -> dict:
    """Uses SegFormer to find 'stuff' classes like sky, road, buildings."""
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        from PIL import Image
        import torch
        
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        
        im = Image.open(img_path).convert("RGB")
        inputs = processor(images=im, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()
        
        # Get top 3 dominant semantic classes
        upsampled = torch.nn.functional.interpolate(logits, size=im.size[::-1], mode='bilinear', align_corners=False)
        seg = upsampled.argmax(dim=1)[0]
        
        # ADE20K labels (subset)
        labels = model.config.id2label
        counts = torch.bincount(seg.flatten())
        top_classes = torch.topk(counts, 3).indices.tolist()
        
        return {labels[c]: (counts[c]/counts.sum()).item() for c in top_classes}
    except Exception as e:
        print(f"Warning: Semantic segmentation failed: {e}")
        return {}


def get_semantic_tags(img_path: str) -> list[str]:
    """Uses CLIP for zero-shot semantic scene tagging."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        image = Image.open(img_path)
        candidate_tags = ["rain", "night", "dusk", "shrine", "torii gate", "cherry blossom", "street", "wet road", "car", "person", "japanese"]

        inputs = processor(text=candidate_tags, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

        # Return tags with probability > 0.10
        top_tags = [candidate_tags[i] for i, p in enumerate(probs) if p > 0.10]
        return top_tags
    except Exception as e:
        print(f"CLIP tagging failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_depth_layout(img_path: str) -> dict:
    """Uses Depth-Anything to categorize objects into foreground, midground, and background."""
    try:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        from PIL import Image
        import torch
        import numpy as np

        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Upsample to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        depth_map = prediction.squeeze().cpu().numpy()

        # Normalize depth to [0, 1] using percentile-based scaling
        depth_min, depth_max = np.percentile(depth_map, (1, 99))
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

        # Detect objects and categorize by depth
        objects = detect_objects(img_path, conf_threshold=0.25)
        scene = {"foreground": [], "midground": [], "background": []}

        for obj in objects:
            # Get center of bounding box
            cx = (obj.bbox.x1 + obj.bbox.x2) / 2
            cy = (obj.bbox.y1 + obj.bbox.y2) / 2

            # Convert normalized coordinates to pixel coordinates
            px = cx * image.size[0]
            py = cy * image.size[1]

            # Get depth value at center
            if 0 <= px < image.size[0] and 0 <= py < image.size[1]:
                depth_val = depth_norm[int(py), int(px)]
            else:
                continue

            # Categorize by normalized depth (Depth-Anything outputs disparity: higher = closer)
            if depth_val > 0.66:
                scene["foreground"].append(obj.label)
            elif depth_val < 0.33:
                scene["background"].append(obj.label)
            else:
                scene["midground"].append(obj.label)

        return scene

    except Exception as e:
        print(f"Warning: Depth-Anything inference failed: {e}")
        return {"foreground": [], "midground": [], "background": []}


def get_accent_colors(img_path: str) -> list[Color]:
    """Extracts vibrant accent colors using OpenCV HSV color space.

    Steps:
      1. Convert image to HSV (OpenCV: Hue 0-179, Sat 0-255, Val 0-255).
      2. Create a binary mask with cv2.inRange for bright, saturated pixels.
         Threshold: Saturation > 30 AND Value > 180 (catches bright pastels).
      3. Extract only masked pixels via numpy boolean indexing.
      4. If vibrant pixels < 1% of total, return [].
      5. Quantize masked HSV pixels (divide-and-multiply by 32).
      6. Count each unique quantized color, calculate true pct = count / total_pixels.
      7. Filter out dominant colors (pct > 5%), keep top 3 accent colors.
      8. Convert each quantized HSV back to BGR via cv2.cvtColor, then to hex.
    """
    try:
        import cv2
        import numpy as np

        # 1. Load image as BGR, convert to HSV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return []

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 2. Create mask: Value > 180 AND Saturation > 30
        #    - V>180 captures bright pastels (e.g. pink cherry blossoms at V=222-233)
        #    - S>30 excludes near-grey pixels
        #    OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
        lower_vibrant = np.array([0, 31, 181], dtype=np.uint8)
        upper_vibrant = np.array([179, 255, 255], dtype=np.uint8)
        vibrant_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)

        # 3. Extract only masked pixels using numpy boolean indexing
        masked_hsv = hsv[vibrant_mask > 0]

        # 4. 1% threshold against total image pixels
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        masked_count = len(masked_hsv)
        if masked_count < total_pixels * 0.01:
            return []

        # 5. Quantize the masked HSV pixels (divide-and-multiply by 32)
        quant_hue = (masked_hsv[:, 0] // 32) * 32
        quant_sat = (masked_hsv[:, 1] // 32) * 32
        quant_val = (masked_hsv[:, 2] // 32) * 32

        # 6. Count each unique quantized color by raw pixel count
        color_counts = {}
        for h, s, v in zip(quant_hue, quant_sat, quant_val):
            key = (int(h), int(s), int(v))
            color_counts[key] = color_counts.get(key, 0) + 1

        # 7. Calculate true percentage: (count of this quantized color) / (total image pixels)
        #    Then filter out dominant colors (>5% of the image).
        #    An accent color must be a small area.
        ACCENT_MAX_PCT = 0.05  # 5% threshold
        filtered_colors = []
        for (h, s, v), count in color_counts.items():
            pct = count / total_pixels  # true percentage of total image
            if pct <= ACCENT_MAX_PCT:
                filtered_colors.append(((h, s, v), count, pct))

        # Sort by count descending, take top 3
        filtered_colors.sort(key=lambda x: x[1], reverse=True)
        top_colors = filtered_colors[:3]

        # 8. Convert HSV → BGR → RGB for hex (CRITICAL: must use cv2.cvtColor)
        colors = []
        for (h, s, v), count, pct in top_colors:
            # Build single-pixel BGR array in HSV space
            pixel = np.array([[[h, s, v]]], dtype=np.uint8)
            # Convert HSV → BGR
            bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
            # BGR → RGB
            r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(Color(hex_color, pct))

        return colors

    except Exception as e:
        print(f"Warning: Accent color extraction failed: {e}")
        return []
