"""
Audio extractor for media2html.
Uses Whisper for transcription and Pyannote for speaker diarization.
"""

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def transcribe(audio_path: str):
    """Transcribe audio using Whisper. Returns segments and language."""
    if not WHISPER_AVAILABLE:
        print("Warning: Whisper not installed. Skipping transcription.")
        return [], "unknown"
    
    try:
        model = whisper.load_model("medium")
        res = model.transcribe(audio_path, word_timestamps=True)
        return res["segments"], res["language"]
    except Exception as e:
        print(f"Warning: Transcription failed ({e}). Returning empty segments.")
        return [], "unknown"


def diarize(audio_path: str, segments: list):
    """Diarize speakers using Pyannote. Falls back to single speaker."""
    try:
        from pyannote.audio import Pipeline
        import os
        
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        dh = pipe(audio_path)
        turns = [(t.start, t.end, t.speaker) for t in dh.itertracks(yield_label=True)]
    except Exception:
        # Fallback: all segments are speaker S1
        turns = [(s["start"], s["end"], "S1") for s in segments]
    
    # Match segments to speakers
    matched = []
    for s in segments:
        spk = "S1"
        for t_start, t_end, t_spk in turns:
            if t_start <= s["start"] <= t_end:
                spk = t_spk
                break
        matched.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
            "spk": spk
        })
    
    return matched


def sound_events(audio_path: str):
    """Detect sound events (stub - returns empty list)."""
    # TODO: Implement YAMNet or similar for sound event detection
    return []
