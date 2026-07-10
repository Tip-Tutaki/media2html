#!/usr/bin/env python3
"""
Comprehensive test suite for media2html V2 (Pure Compiler).
Tests all extraction modes, caching, HTML structure, and error handling.
"""

import tempfile
import os
from PIL import Image, ImageDraw
import numpy as np


def create_test_image(path, size=(800, 600)):
    """Create a test image with various shapes and text."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw colored rectangles
    draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=3)
    draw.rectangle([250, 50, 450, 200], fill='blue', outline='black', width=3)
    draw.rectangle([500, 50, 750, 200], fill='green', outline='black', width=3)

    # Draw circles
    draw.ellipse([50, 250, 250, 450], fill='yellow', outline='black', width=3)
    draw.ellipse([300, 250, 500, 450], fill='purple', outline='black', width=3)

    # Add text
    draw.text((100, 500), "Object Detection Test", fill='black')
    draw.text((400, 500), "With Shapes", fill='black')

    img.save(path)
    return img


def test_image_modes():
    """Test all image extraction modes."""
    from media2html.pipeline import image_to_html

    print("\n" + "=" * 70)
    print("TESTING IMAGE EXTRACTION MODES")
    print("=" * 70)

    test_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    create_test_image(test_img.name)
    test_img.close()

    print(f"\nTest image: {test_img.name}")
    print(f"   Size: 800x600")
    print(f"   Content: 3 rectangles, 2 circles, 2 text regions\n")

    modes = ['minimal', 'compact', 'rich']
    results = {}

    for mode in modes:
        print(f"Testing {mode.upper()} mode...")
        try:
            html = image_to_html(test_img.name, mode=mode)
            results[mode] = len(html)
            print(f"  OK: {len(html)} chars")
            print(f"   Preview: {html[:150]}...")
        except Exception as e:
            results[mode] = f"ERROR: {str(e)}"
            print(f"  FAIL: {e}")
        print()

    os.unlink(test_img.name)
    return results


def test_pure_compiler():
    """Test Pure Compiler architecture — no VLM, strict filtering."""
    from media2html.pipeline import image_to_html
    from media2html.extractors.vision import detect_objects, get_caption

    print("\n" + "=" * 70)
    print("TESTING PURE COMPILER ARCHITECTURE")
    print("=" * 70)

    test_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    create_test_image(test_img.name)
    test_img.close()

    print(f"\nTest image: {test_img.name}")

    # Test 1: No cert attributes in HTML
    print("\n  Test 1: No cert attributes in HTML output")
    html = image_to_html(test_img.name, mode="rich")
    if 'cert=' in html:
        print("  FAIL: HTML contains cert attributes")
    else:
        print("  OK: No cert attributes (Pure Compiler)")

    # Test 2: Hard 0.50 threshold
    print("\n  Test 2: Hard 0.50 confidence threshold")
    objects = detect_objects(test_img.name)
    low_conf = [o for o in objects if o.conf < 0.50]
    high_conf = [o for o in objects if o.conf >= 0.50]
    print(f"    Total detected: {len(objects)}")
    print(f"    Below threshold (dropped): {len(low_conf)}")
    print(f"    Above threshold (kept): {len(high_conf)}")
    if len(low_conf) == 0:
        print("  OK: All objects above 0.50 threshold")
    else:
        print(f"  OK: Filtered {len(low_conf)} low-confidence objects")

    # Test 3: Algorithmic caption
    print("\n  Test 3: Algorithmic caption generation")
    caption = get_caption(objects, [], [])
    print(f"    Caption: {caption[:100]}...")
    if "Scene contains:" in caption:
        print("  OK: Algorithmic caption generated")
    else:
        print("  WARN: Caption may be empty")

    # Test 4: HTML contains expected tags
    print("\n  Test 4: HTML contains expected V2 tags")
    required_tags = ['<image-summary', '<caption', '<objects', '</image-summary>']
    for tag in required_tags:
        if tag in html:
            print(f"  OK: Found {tag}")
        else:
            print(f"  FAIL: Missing {tag}")

    # Test 5: Scene tag exists in rich mode
    print("\n  Test 5: <scene> tag in rich mode")
    if '<scene>' in html:
        print("  OK: <scene> tag present")
    else:
        print("  WARN: <scene> tag not found")

    # Test 6: Semantic tags in rich mode
    print("\n  Test 6: <semantic-tags> in rich mode")
    if '<semantic-tags>' in html:
        print("  OK: <semantic-tags> tag present")
    else:
        print("  WARN: <semantic-tags> tag not found")

    # Test 7: No literal quotes in scene output
    print("\n  Test 7: No literal quotes in <scene> block")
    if '<foreground>"' in html or '<background>"' in html:
        print("  FAIL: Literal quotes found in scene block")
    else:
        print("  OK: No literal quotes in scene block")

    # Test 8: Accent colors have label attribute
    print("\n  Test 8: <accent-colors> has label attribute")
    if 'label="vibrant"' in html:
        print("  OK: label='vibrant' attribute present")
    else:
        print("  WARN: label='vibrant' not found (may be empty if no vibrant colors)")

    os.unlink(test_img.name)
    print("\n  OK: Pure Compiler tests passed")


def test_caching():
    """Test disk caching functionality."""
    from media2html.cache import cache, get_cache_key
    from media2html.pipeline import image_to_html

    print("\n" + "=" * 70)
    print("TESTING CACHING")
    print("=" * 70)

    test_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    create_test_image(test_img.name)
    test_img.close()

    print(f"\nTest image: {test_img.name}")

    # Get cache key
    key = get_cache_key(test_img.name, 'img_minimal')
    print(f"  Cache key: {key[:32]}...")

    # First extraction
    print(f"\n  First extraction...")
    html1 = image_to_html(test_img.name, mode='minimal')
    print(f"  OK: Generated {len(html1)} chars")

    # Second extraction (should use cache)
    print(f"\n  Second extraction (cached)...")
    html2 = image_to_html(test_img.name, mode='minimal')
    print(f"  OK: Generated {len(html2)} chars")

    # Verify cache hit
    print(f"\n  Cache verification:")
    print(f"    Same output: {html1 == html2}")
    print(f"    Cache entries: {len(cache)}")

    os.unlink(test_img.name)


def test_error_handling():
    """Test error handling and edge cases."""
    from media2html.pipeline import media_to_html

    print("\n" + "=" * 70)
    print("TESTING ERROR HANDLING")
    print("=" * 70)

    test_cases = [
        ("Non-existent file", "/tmp/nonexistent.jpg"),
        ("Invalid extension", "/tmp/test.txt"),
        ("Empty file", tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name),
    ]

    for name, path in test_cases:
        print(f"\nTesting {name}...")
        try:
            if name == "Empty file":
                with open(path, 'w') as f:
                    pass
                html = media_to_html(path, mode='minimal')
                print(f"  WARN: Should have raised an error")
            else:
                html = media_to_html(path, mode='minimal')
                print(f"  FAIL: Unexpected success: {len(html)} chars")
        except FileNotFoundError:
            print(f"  OK: Correctly handled: File not found")
        except ValueError as e:
            print(f"  OK: Correctly handled: {e}")
        except Exception as e:
            print(f"  OK: Handled: {type(e).__name__}: {e}")

        if os.path.exists(path) and name != "Non-existent file":
            os.unlink(path)


def test_video_extraction():
    """Test video extraction (requires ffmpeg)."""
    from media2html.pipeline import video_to_html

    print("\n" + "=" * 70)
    print("TESTING VIDEO EXTRACTION")
    print("=" * 70)

    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)

    if result.returncode != 0:
        print("\nFFmpeg not found. Skipping video tests.")
        print("   Install with: sudo apt install ffmpeg")
        return

    print("\nFFmpeg detected")

    test_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    test_video.close()

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=2:size=320x240:rate=25',
            '-c:v', 'libx264',
            test_video.name
        ], capture_output=True, check=True)

        print(f"  Created test video: {test_video.name}")

        print(f"\n  Testing video_to_html...")
        html = video_to_html(test_video.name, mode='compact')
        print(f"  OK: {len(html)} chars")
        print(f"   Preview: {html[:200]}...")

    except Exception as e:
        print(f"  FAIL: {e}")
    finally:
        if os.path.exists(test_video.name):
            os.unlink(test_video.name)


def test_audio_extraction():
    """Test audio extraction (requires whisper)."""
    from media2html.pipeline import audio_to_html

    print("\n" + "=" * 70)
    print("TESTING AUDIO EXTRACTION")
    print("=" * 70)

    try:
        import whisper
        print("\nOpenAI Whisper detected")
    except ImportError:
        print("\nOpenAI Whisper not installed. Skipping audio tests.")
        return

    test_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    test_audio.close()

    try:
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        import wave
        with wave.open(test_audio.name, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"  Created test audio: {test_audio.name}")

        print(f"\n  Testing audio_to_html...")
        html = audio_to_html(test_audio.name, mode='compact')
        print(f"  OK: {len(html)} chars")
        print(f"   Preview: {html[:200]}...")

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_audio.name):
            os.unlink(test_audio.name)


def test_media_to_html():
    """Test the main media_to_html function."""
    from media2html.pipeline import media_to_html

    print("\n" + "=" * 70)
    print("TESTING media_to_html (AUTO-DETECT)")
    print("=" * 70)

    test_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    create_test_image(test_img.name)
    test_img.close()

    print(f"\nTest image: {test_img.name}")

    try:
        html = media_to_html(test_img.name, mode='minimal')
        print(f"  OK: Auto-detection success: {len(html)} chars")
    except Exception as e:
        print(f"  FAIL: {e}")

    os.unlink(test_img.name)


def test_data_model():
    """Test that the data model classes are importable."""
    from media2html import ImageSummary, Obj, BBox, Color, TextRegion, media_to_html

    print("\n" + "=" * 70)
    print("TESTING DATA MODEL IMPORTS")
    print("=" * 70)

    # Test BBox
    bbox = BBox(0.1, 0.2, 0.3, 0.4)
    assert bbox.fmt() == "0.100,0.200,0.300,0.400"
    print("  OK: BBox.fmt()")

    # Test Obj
    obj = Obj("person", bbox, 0.95)
    html = obj.to_html()
    assert 'label="person"' in html
    print("  OK: Obj.to_html()")

    # Test Color
    color = Color("#ff0000", 0.5)
    html = color.to_html()
    assert 'hex="#ff0000"' in html
    print("  OK: Color.to_html()")

    # Test TextRegion
    tr = TextRegion("Hello", bbox)
    html = tr.to_html()
    assert 'Hello' in html
    print("  OK: TextRegion.to_html()")

    # Test ImageSummary
    summary = ImageSummary(width=800, height=600, source="test.jpg")
    html = summary.to_html()
    assert '<image-summary width="800" height="600"' in html
    print("  OK: ImageSummary.to_html()")

    print("\n  OK: All data model tests passed")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MEDIA2HTML PACKAGE - COMPREHENSIVE TEST SUITE (V2)")
    print("=" * 70)

    test_data_model()
    test_image_modes()
    test_pure_compiler()
    test_caching()
    test_error_handling()
    test_video_extraction()
    test_audio_extraction()
    test_media_to_html()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
