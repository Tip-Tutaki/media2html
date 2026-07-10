"""MCP server for Media2HTML — exposes media transcoding as an AI agent tool."""

import sys
import os

# Allow running as `python3 media2html/mcp_server.py` directly
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from mcp.server.fastmcp import FastMCP
from media2html.pipeline import media_to_html

mcp = FastMCP("Media2HTML")


@mcp.tool()
def transcode_media(file_path: str, mode: str = "rich") -> str:
    """
    Converts an image, video, or audio file into a structured HTML
    representation for text-based LLM reasoning.
    Use this tool whenever you need to 'see' or analyze a media file.

    Args:
        file_path: Local path to the media file.
        mode: Extraction detail ('minimal', 'compact', 'rich').
    """
    return media_to_html(file_path, mode=mode)


if __name__ == "__main__":
    mcp.run()
