"""Media2HTML — A VLM-Free Pure Compiler for Multimodal Agents."""

from .pipeline import media_to_html
from .html_builder import ImageSummary, Obj, BBox, Color, TextRegion

__all__ = [
    "media_to_html",
    "ImageSummary",
    "Obj",
    "BBox",
    "Color",
    "TextRegion",
]
