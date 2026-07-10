import ffmpeg
import os
import tempfile
from scenedetect import detect, ContentDetector


def get_duration(path: str) -> float:
    probe = ffmpeg.probe(path)
    return float(probe['format']['duration'])


def detect_scenes(path: str, max_scenes: int = 12):
    scenes = detect(path, ContentDetector(threshold=27.0))
    if not scenes:
        scenes = [(0.0, get_duration(path))]
    
    if len(scenes) > max_scenes:
        step = len(scenes) / max_scenes
        scenes = [scenes[int(i*step)] for i in range(max_scenes)]
    return scenes


def extract_keyframe(path: str, t: float, tmp_dir: str, idx: int) -> str:
    kf = os.path.join(tmp_dir, f"frame_{idx:03d}.jpg")
    (
        ffmpeg.input(path, ss=t)
             .output(kf, vframes=1, vcodec="mjpeg", q=2, loglevel="error")
             .overwrite_output().run()
    )
    return kf
