import io
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
import librosa


DEFAULT_TIMEOUT_SEC = 15


def _ensure_wav_16k_mono(audio_bytes: bytes) -> bytes:
    """Return audio as 16 kHz mono 16-bit PCM WAV bytes.

    Attempts to decode arbitrary input formats via soundfile, resamples via librosa,
    and then re-encodes to a standard WAV suitable for most ASR/metrics APIs.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes is empty")

    # Read input into float32 mono array with original sample rate
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            data = f.read(dtype="float32")
            sr = f.samplerate
            if data.ndim == 2:  # stereo -> mono
                data = np.mean(data, axis=1)
    except Exception:
        # Fallback: let librosa try to decode
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        data = y.astype(np.float32)

    # Resample to 16k mono
    if sr != 16000:
        data = librosa.resample(y=data, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Encode to 16-bit PCM WAV
    out = io.BytesIO()
    sf.write(out, data, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()


@dataclass
class AudioAPIConfig:
    transcribe_url: Optional[str]
    metrics_url: Optional[str]
    api_key: Optional[str]
    timeout_sec: int = DEFAULT_TIMEOUT_SEC

    @classmethod
    def from_env(cls) -> "AudioAPIConfig":
        return cls(
            transcribe_url=os.environ.get("AUDIO_TRANSCRIBE_URL"),
            metrics_url=os.environ.get("AUDIO_METRICS_URL"),
            api_key=os.environ.get("AUDIO_API_KEY"),
            timeout_sec=int(os.environ.get("AUDIO_API_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC))),
        )


class AudioAPIClient:
    def __init__(self, config: Optional[AudioAPIConfig] = None) -> None:
        self.config = config or AudioAPIConfig.from_env()

    def is_transcribe_configured(self) -> bool:
        return bool(self.config.transcribe_url)

    def is_metrics_configured(self) -> bool:
        return bool(self.config.metrics_url)

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> Optional[str]:
        """Call external transcription API. Returns transcript text or None on failure."""
        if not self.is_transcribe_configured():
            return None
        wav_bytes = _ensure_wav_16k_mono(audio_bytes)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"language": language}
        try:
            resp = requests.post(
                self.config.transcribe_url,
                headers=self._auth_headers(),
                files=files,
                data=data,
                timeout=self.config.timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            # Common keys across APIs: 'transcript', 'text'
            transcript = payload.get("transcript") or payload.get("text")
            if isinstance(transcript, str) and transcript.strip():
                return transcript.strip()
            return None
        except Exception:
            return None

    def metrics(self, audio_bytes: bytes) -> Optional[Dict[str, float]]:
        """Call external metrics API. Returns dict or None on failure."""
        if not self.is_metrics_configured():
            return None
        wav_bytes = _ensure_wav_16k_mono(audio_bytes)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        try:
            resp = requests.post(
                self.config.metrics_url,
                headers=self._auth_headers(),
                files=files,
                timeout=self.config.timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
            # Normalize keys to expected names
            result: Dict[str, float] = {}
            # Accept common variations
            if "speech_rate_wpm" in payload:
                result["speech_rate_wpm"] = float(payload["speech_rate_wpm"])  # type: ignore[arg-type]
            if "pause_count" in payload:
                result["pause_count"] = float(payload["pause_count"])  # type: ignore[arg-type]
            if "pitch_variation_semitones" in payload:
                result["pitch_variation_semitones"] = float(payload["pitch_variation_semitones"])  # type: ignore[arg-type]
            # Some APIs might use different keys
            if "wpm" in payload and "speech_rate_wpm" not in result:
                result["speech_rate_wpm"] = float(payload["wpm"])  # type: ignore[arg-type]
            if "pitch_stdev" in payload and "pitch_variation_semitones" not in result:
                result["pitch_variation_semitones"] = float(payload["pitch_stdev"])  # type: ignore[arg-type]
            # Cast pause count back to int-like if present
            if "pause_count" in result:
                result["pause_count"] = int(round(result["pause_count"]))
            return result or None
        except Exception:
            return None


_client = AudioAPIClient()


def _compute_local_pause_count(y: np.ndarray, sr: int) -> int:
    frame_length = 512
    hop_length = 512
    # Simple energy per frame
    energies = np.array([
        float(np.sum(np.square(y[i : i + frame_length])))
        for i in range(0, len(y), hop_length)
    ])
    threshold = np.percentile(energies, 10)
    return int(np.sum(energies < threshold))


def _compute_local_pitch_variation_semitones(y: np.ndarray, sr: int) -> float:
    # Use librosa.yin for F0 then compute semitone std versus median F0
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]
    if f0.size < 5:
        return float("nan")
    median_f0 = np.median(f0)
    semitones = 12.0 * np.log2(f0 / median_f0)
    return float(np.nanstd(semitones))


def _load_audio_to_16k(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    wav_bytes = _ensure_wav_16k_mono(audio_bytes)
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return y, sr


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> Optional[str]:
    """Public helper: Transcribe via API if configured. Returns None on failure."""
    return _client.transcribe(audio_bytes, language=language)


def compute_speech_metrics(audio_bytes: bytes) -> Dict[str, float]:
    """Compute speech metrics via API if configured; otherwise use local lightweight analysis.

    Returns a dict with keys: 'Speech Rate (words/min)', 'Pause Count', 'Pitch Variation (semitones)'.
    Values may be NaN when not computable locally.
    """
    # Try API first
    api_result = _client.metrics(audio_bytes)
    if api_result:
        return {
            "Speech Rate (words/min)": float(api_result.get("speech_rate_wpm", np.nan)),
            "Pause Count": int(api_result.get("pause_count", np.nan)) if not np.isnan(api_result.get("pause_count", np.nan)) else int(np.nan),
            "Pitch Variation (semitones)": float(api_result.get("pitch_variation_semitones", np.nan)),
        }

    # Local fallback
    y, sr = _load_audio_to_16k(audio_bytes)
    duration_sec = max(len(y) / float(sr), 1e-6)

    pause_count = _compute_local_pause_count(y, sr)
    pitch_var = _compute_local_pitch_variation_semitones(y, sr)

    # Try to derive WPM using transcription API if available
    wpm = np.nan
    transcript = _client.transcribe(audio_bytes)
    if transcript:
        num_words = len([w for w in transcript.strip().split() if w])
        wpm = float(num_words) / (duration_sec / 60.0)

    return {
        "Speech Rate (words/min)": float(wpm),
        "Pause Count": int(pause_count),
        "Pitch Variation (semitones)": float(pitch_var),
    }
