from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import structlog

from miip.config import settings
from miip.state import IncidentState

log = structlog.get_logger()


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str):
    import logging
    import warnings
    from transformers import pipeline as hf_pipeline
    # Suppress verbose transformers info/warnings that aren't actionable
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )


_WHISPER_SR = 16_000  # Whisper native sample rate


def _load_audio(path: str) -> dict:
    """Load WAV to a 16 kHz float32 mono array without requiring ffmpeg or torchaudio."""
    import numpy as np
    import scipy.io.wavfile as wav_io
    import scipy.signal as sig

    suffix = Path(path).suffix.lower()
    if suffix != ".wav":
        # Non-WAV files still require ffmpeg; pass path and let the pipeline try.
        return path  # type: ignore[return-value]

    sr, data = wav_io.read(path)

    # Normalise to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:           # stereo → mono
        data = data.mean(axis=1)

    if sr != _WHISPER_SR:       # resample to 16 kHz
        target_len = int(len(data) * _WHISPER_SR / sr)
        data = sig.resample(data, target_len).astype(np.float32)
        sr = _WHISPER_SR

    return {"array": data, "sampling_rate": sr}


def _transcribe(path: str, model_name: str) -> str:
    pipe = _load_pipeline(model_name)
    audio = _load_audio(path)
    result = pipe(audio)
    raw = result["text"] if isinstance(result, dict) else str(result)
    return _clean(raw)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{3,}", "...", text)
    return text


def asr_agent(state: IncidentState) -> IncidentState:
    log.info("asr_agent invoked", incident_id=state["incident_id"])

    memos = state.get("voice_memo_paths") or []

    if not memos:
        log.info("no voice memos — skipping ASR", incident_id=state["incident_id"])
        return {
            **state,
            "asr_transcription": None,
            "completed_agents": [*state.get("completed_agents", []), "asr"],
        }

    transcripts: list[str] = []
    for raw_path in memos:
        p = Path(raw_path)
        if not p.exists():
            log.warning("audio file not found", path=raw_path)
            transcripts.append(f"[{p.name}] (file not found)")
            continue
        try:
            text = _transcribe(str(p), settings.whisper_model)
            label = text if text else "(no speech detected)"
            log.info("transcribed", path=raw_path, chars=len(text))
            transcripts.append(f"[{p.name}]\n{label}")
        except Exception as exc:
            log.warning("transcription failed", path=raw_path, error=str(exc))
            transcripts.append(f"[{p.name}] (transcription error: {exc})")

    return {
        **state,
        "asr_transcription": "\n\n".join(transcripts) if transcripts else None,
        "completed_agents": [*state.get("completed_agents", []), "asr"],
    }
