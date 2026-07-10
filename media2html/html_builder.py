"""
HTML builder for media2html.
Generates semantic HTML from extracted media features.
All objects in HTML have confidence >= 0.50 (filtered at extraction).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import xml.sax.saxutils as su


def esc(s: str) -> str:
    """Escape special XML/HTML characters."""
    return su.escape(s or "")


@dataclass
class BBox:
    """Normalized bounding box [0,1] range."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def fmt(self) -> str:
        """Format bounding box as comma-separated string."""
        return f"{self.x1:.3f},{self.y1:.3f},{self.x2:.3f},{self.y2:.3f}"


@dataclass
class Obj:
    """Detected object with confidence score.
    
    All objects in HTML have conf >= 0.50 (filtered at extraction).
    No cert attribute needed - confidence is implicit.
    """
    label: str
    bbox: BBox
    conf: float
    
    def to_html(self) -> str:
        """Generate HTML for this object."""
        return f'<obj label="{esc(self.label)}" bbox="{self.bbox.fmt()}"/>'


@dataclass
class TextRegion:
    """Extracted text region with bounding box."""
    text: str
    bbox: BBox
    
    def to_html(self) -> str:
        """Generate HTML for this text region."""
        return f'<t bbox="{self.bbox.fmt()}">{esc(self.text)}</t>'


@dataclass
class Color:
    """Dominant color with percentage."""
    hex: str
    pct: float
    
    def to_html(self) -> str:
        """Generate HTML for this color."""
        return f'<c hex="{self.hex}" pct="{self.pct:.2f}"/>'


@dataclass
class ImageSummary:
    """Summary of an image with all extracted features."""
    width: int
    height: int
    source: str
    caption: str = ""
    objects: list[Obj] = field(default_factory=list)
    text_regions: list[TextRegion] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)
    visual_grid: str = ""
    spatial_relations: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    layout: str = ""
    scene: dict = field(default_factory=dict)  # Foreground, midground, background
    accent_colors: list[Color] = field(default_factory=list)
    def to_html(self) -> str:
        """Generate complete HTML for the image summary."""
        parts = []
        parts.append(f'<image-summary width="{self.width}" height="{self.height}" source="{self.source}">')

        # Visual grid (if available)
        if self.visual_grid:
            parts.append(f"  {self.visual_grid}")

        # Caption
        if self.caption:
            parts.append(f"  <caption>{self.caption}</caption>")

        # Spatial relations
        if self.spatial_relations:
            parts.append("  <spatial-graph>")
            for rel in self.spatial_relations:
                parts.append(f"    <rel>{rel}</rel>")
            parts.append("  </spatial-graph>")

        # Semantic tags
        if self.tags:
            parts.append("  <semantic-tags>")
            for tag in self.tags:
                parts.append(f"    <t>{tag}</t>")
            parts.append("  </semantic-tags>")

        # Scene (foreground, midground, background)
        if self.scene:
            parts.append("  <scene>")
            for level in ["foreground", "midground", "background"]:
                if level in self.scene:
                    items = ", ".join(self.scene[level])
                    parts.append(f"    <{level}>{items}</{level}>")
            parts.append("  </scene>")

        # Accent colors
        if self.accent_colors:
            parts.append("  <accent-colors>")
            for c in self.accent_colors:
                parts.append(f"    <c hex=\"{c.hex}\" pct=\"{c.pct:.2f}\" label=\"vibrant\"/>")
            parts.append("  </accent-colors>")

        # Objects
        if self.objects:
            parts.append("  <objects>")
            for obj in self.objects:
                parts.append(f"    {obj.to_html()}")
            parts.append("  </objects>")

        # Text regions
        if self.text_regions:
            parts.append("  <text-regions>")
            for tr in self.text_regions:
                parts.append(f"    {tr.to_html()}")
            parts.append("  </text-regions>")

        # Colors
        if self.colors:
            parts.append("  <colors>")
            for c in self.colors:
                parts.append(f"    {c.to_html()}")
            parts.append("  </colors>")

        # Layout
        if self.layout:
            parts.append(f"  <layout>{self.layout}</layout>")

        parts.append("</image-summary>")
        return "\n".join(parts)
