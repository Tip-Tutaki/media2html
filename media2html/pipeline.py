import os
import tempfile
from PIL import Image
from .cache import cache, get_cache_key
from .html_builder import ImageSummary, Obj, BBox, TextRegion, Color
from .extractors import vision, audio, video


def image_to_html(path: str, mode: str = "compact") -> str:
    """Convert image to structured HTML.

    Modes:
    - minimal: Caption + 3 objects + 2 colors (~200 tokens)
    - compact: + OCR (top-N) + 5 objects + 3 colors + visual grid (~800 tokens)
    - rich: + Layout + all OCR + all objects + all colors + spatial relations + tags + depth scene + accent colors (~2500 tokens)
    """
    key = get_cache_key(path, f"img_{mode}")
    if key in cache:
        return cache[key]

    im = Image.open(path)
    W, H = im.size
    s = ImageSummary(width=W, height=H, source=os.path.basename(path))

    # Extract all features
    s.objects = vision.detect_objects(path)
    s.colors = vision.dominant_colors(path)
    s.accent_colors = vision.get_accent_colors(path) if mode == "rich" else []

    # Add OCR and layout for compact/rich modes
    if mode in ["compact", "rich"]:
        s.text_regions = vision.extract_ocr(path)

    # Visual grid (compact and rich)
    if mode in ["compact", "rich"]:
        s.visual_grid = vision.get_visual_grid(path, mode)

    # Spatial relations (rich only)
    if mode == "rich":
        s.spatial_relations = vision.get_spatial_relations(s.objects)

    # Semantic tags (rich only) - use RAM via subprocess
    if mode == "rich":
        s.tags = vision.get_semantic_tags(path)

    # Depth scene (rich only)
    if mode == "rich":
        s.scene = vision.get_depth_layout(path)

    # Generate algorithmic caption (zero-compute, no external model)
    s.caption = vision.get_caption(s.objects, s.text_regions, s.colors, s.tags, s.layout)

    # Generate HTML
    html = s.to_html()
    cache[key] = html
    return html


def video_to_html(path: str, mode: str = "compact") -> str:
    key = get_cache_key(path, f"vid_{mode}")
    if key in cache: return cache[key]


    duration = video.get_duration(path)
    scenes = video.detect_scenes(path)
    tmp_dir = tempfile.mkdtemp()
    
    # Audio extraction (global)
    segments, lang = audio.transcribe(path)
    matched_speech = audio.diarize(path, segments)
    sfx_events = audio.sound_events(path)


    blocks = []
    for i, (start, end) in enumerate(scenes):
        t = start + (end - start) / 2
        kf_path = video.extract_keyframe(path, t, tmp_dir, i)
        kf_html = image_to_html(kf_path, mode="minimal") # Use minimal for keyframes
        
        # Interleave audio for this scene
        scene_speech = [s for s in matched_speech if start <= s["start"] <= end]
        scene_sfx = [e for e in sfx_events if start <= e["t"] <= end]
        
        speech_html = "\n".join(
            f'      <u spk="{s["spk"]}" t="{s["start"]:.1f}-{s["end"]:.1f}">{s["text"]}</u>' 
            for s in scene_speech
        )
        sfx_html = "\n".join(
            f'      <e type="{e["label"]}" t="{e["t"]:.1f}"/>'
            for e in scene_sfx
        )


        blocks.append(
            f'  <scene start="{start:.2f}" end="{end:.2f}">\n'
            f'    <visual>\n      {kf_html}\n    </visual>\n'
            f'    <audio start="{start:.2f}" end="{end:.2f}">\n'
            f'      <speech>\n{speech_html}\n      </speech>\n'
            f'      <non-speech>\n{sfx_html}\n      </non-speech>\n'
            f'    </audio>\n'
            f'  </scene>'
        )


    html = f'<video-summary duration="{duration:.2f}" source="{os.path.basename(path)}">\n' + "\n".join(blocks) + '\n</video-summary>'
    cache[key] = html
    return html


def audio_to_html(path: str, mode: str = "compact") -> str:
    key = get_cache_key(path, f"aud_{mode}")
    if key in cache: return cache[key]

    segments, lang = audio.transcribe(path)
    matched_speech = audio.diarize(path, segments)
    sfx_events = audio.sound_events(path)

    # Handle empty segments gracefully
    if not matched_speech:
        html = (
            f'<audio-summary duration="0.0" source="{os.path.basename(path)}">\n'
            f'  <language>{lang}</language>\n'
            f'  <transcript>\n  </transcript>\n'
            f'  <events>\n  </events>\n'
            f'</audio-summary>'
        )
        cache[key] = html
        return html

    speech_html = "\n".join(
        f'    <u spk="{s["spk"]}" t="{s["start"]:.1f}-{s["end"]:.1f}">{s["text"]}</u>' 
        for s in matched_speech
    )
    sfx_html = "\n".join(
        f'    <e type="{e["label"]}" t="{e["t"]:.1f}"/>'
        for e in sfx_events
    )

    duration = matched_speech[-1]["end"]

    html = (
        f'<audio-summary duration="{duration:.1f}" source="{os.path.basename(path)}">\n'
        f'  <language>{lang}</language>\n'
        f'  <transcript>\n{speech_html}\n  </transcript>\n'
        f'  <events>\n{sfx_html}\n  </events>\n'
        f'</audio-summary>'
    )
    cache[key] = html
    return html


def media_to_html(path: str, mode: str = "compact") -> str:
    """Agent Tool Interface."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return image_to_html(path, mode)
    elif ext in [".mp4", ".mkv", ".avi", ".mov"]:
        return video_to_html(path, mode)
    elif ext in [".mp3", ".wav", ".flac", ".m4a"]:
        return audio_to_html(path, mode)
    raise ValueError(f"Unsupported file extension: {ext}")
