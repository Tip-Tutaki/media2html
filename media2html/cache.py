import hashlib
import diskcache
import os


CACHE_DIR = os.path.join(os.getcwd(), ".media2html_cache")
cache = diskcache.Cache(CACHE_DIR)


def get_cache_key(file_path: str, mode: str) -> str:
    """Generates a unique cache key based on file content and extraction mode."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536) # Read first 64KB for speed
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return f"{hasher.hexdigest()}_{mode}"
