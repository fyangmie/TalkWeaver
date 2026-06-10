"""Audio loading and preprocessing for the TalkWeaver ASR baseline."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import ROOT_DIR


TARGET_SAMPLE_RATE = 16_000
TARGET_PEAK = 0.95


def _decode_pcm(raw_audio: bytes, sample_width: int) -> np.ndarray:
    """Decode little-endian PCM bytes to floating-point samples."""

    if sample_width == 1:
        samples = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0
    if sample_width == 2:
        samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
        return samples / 32_768.0
    if sample_width == 3:
        bytes_24 = np.frombuffer(raw_audio, dtype=np.uint8).reshape(-1, 3)
        samples = (
            bytes_24[:, 0].astype(np.int32)
            | (bytes_24[:, 1].astype(np.int32) << 8)
            | (bytes_24[:, 2].astype(np.int32) << 16)
        )
        samples = np.where(samples & 0x800000, samples - 0x1000000, samples)
        return samples.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        samples = np.frombuffer(raw_audio, dtype="<i4").astype(np.float32)
        return samples / 2_147_483_648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def _load_pcm_wav(path: Path) -> tuple[np.ndarray, int, int, str]:
    """Load an uncompressed WAV file using the Python standard library."""

    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(
                f"Unsupported WAV compression: {wav_file.getcomptype()}"
            )
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw_audio = wav_file.readframes(frame_count)

    samples = _decode_pcm(raw_audio, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    else:
        samples = samples.reshape(-1, 1)
    return samples, sample_rate, channels, "wave"


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int, int, str]:
    """Load audio through soundfile when it is installed."""

    import soundfile as sf

    samples, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    return samples, int(sample_rate), int(samples.shape[1]), "soundfile"


def _load_with_pydub(path: Path) -> tuple[np.ndarray, int, int, str]:
    """Load compressed formats through pydub and FFmpeg."""

    from pydub import AudioSegment

    audio = AudioSegment.from_file(path)
    channels = audio.channels
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples.reshape(-1, channels)
    scale = float(1 << (8 * audio.sample_width - 1))
    samples = samples / scale
    return samples, int(audio.frame_rate), channels, "pydub"


def load_audio(path: str | Path) -> tuple[np.ndarray, int, int, str]:
    """Load audio through the first available compatible decoder."""

    audio_path = Path(path)
    errors: list[str] = []

    if audio_path.suffix.lower() == ".wav":
        try:
            return _load_pcm_wav(audio_path)
        except (ValueError, wave.Error, EOFError) as exc:
            errors.append(f"wave: {exc}")

    try:
        return _load_with_soundfile(audio_path)
    except Exception as exc:
        errors.append(f"soundfile: {exc}")

    try:
        return _load_with_pydub(audio_path)
    except Exception as exc:
        errors.append(f"pydub: {exc}")

    details = "; ".join(errors)
    raise RuntimeError(
        "Unable to decode audio. PCM WAV works without optional decoders; "
        "install soundfile for additional formats or pydub plus FFmpeg for "
        f"MP3/M4A. Decoder errors: {details}"
    )


def _resample_audio(
    audio: np.ndarray,
    input_rate: int,
    target_rate: int,
) -> tuple[np.ndarray, str]:
    """Resample mono audio, preferring polyphase resampling when available."""

    if input_rate == target_rate:
        return audio.astype(np.float32, copy=False), "not_needed"

    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(input_rate, target_rate)
        resampled = resample_poly(
            audio,
            target_rate // divisor,
            input_rate // divisor,
        )
        return np.asarray(resampled, dtype=np.float32), "scipy_resample_poly"
    except ImportError:
        output_length = max(
            1,
            int(round(len(audio) * target_rate / input_rate)),
        )
        source_positions = np.arange(len(audio), dtype=np.float64)
        target_positions = np.linspace(
            0,
            max(0, len(audio) - 1),
            output_length,
        )
        resampled = np.interp(target_positions, source_positions, audio)
        return resampled.astype(np.float32), "numpy_linear"


def _normalize_peak(
    audio: np.ndarray,
    target_peak: float = TARGET_PEAK,
) -> tuple[np.ndarray, float, float]:
    """Peak-normalize audio without amplifying digital silence."""

    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_before > 0:
        audio = audio * (target_peak / peak_before)
    peak_after = float(np.max(np.abs(audio))) if audio.size else 0.0
    return np.clip(audio, -1.0, 1.0), peak_before, peak_after


def _write_pcm_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono floating-point samples as 16-bit PCM WAV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def preprocess_audio(
    audio_path: str | Path | None,
    *,
    mock: bool = False,
    denoise: bool = False,
    output_path: str | Path | None = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, Any]:
    """Convert audio to normalized mono 16 kHz PCM WAV.

    Mock mode validates no input file and writes no fabricated waveform.
    """

    if mock:
        return {
            "mode": "mock_demo",
            "input_path": str(audio_path) if audio_path else None,
            "output_path": None,
            "input_sample_rate": None,
            "input_channels": None,
            "sample_rate": target_sample_rate,
            "channels": 1,
            "duration_seconds": 9.4,
            "normalized": True,
            "denoise_requested": denoise,
            "denoise_applied": False,
            "resampler": "mock_demo",
            "loader": "mock_demo",
            "warnings": [
                "Mock mode did not read or create an audio waveform."
            ],
        }

    if audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")

    input_path = Path(audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive.")

    samples, input_rate, input_channels, loader = load_audio(input_path)
    if samples.size == 0:
        raise ValueError(f"Audio file contains no samples: {input_path}")

    samples = np.nan_to_num(samples, copy=False)
    mono = samples.mean(axis=1, dtype=np.float32)
    resampled, resampler = _resample_audio(
        mono,
        input_rate,
        target_sample_rate,
    )

    warnings: list[str] = []
    denoise_applied = False
    if denoise:
        try:
            import noisereduce as noise_reducer
        except ImportError:
            warnings.append(
                "noisereduce is not installed; continuing without denoising."
            )
        else:
            resampled = np.asarray(
                noise_reducer.reduce_noise(
                    y=resampled,
                    sr=target_sample_rate,
                ),
                dtype=np.float32,
            )
            denoise_applied = True

    normalized, peak_before, peak_after = _normalize_peak(resampled)
    destination = (
        Path(output_path)
        if output_path is not None
        else ROOT_DIR
        / "data"
        / "processed"
        / f"{input_path.stem}_mono_16k.wav"
    )
    _write_pcm_wav(destination, normalized, target_sample_rate)

    return {
        "mode": "real",
        "input_path": str(input_path),
        "output_path": str(destination),
        "input_sample_rate": input_rate,
        "input_channels": input_channels,
        "sample_rate": target_sample_rate,
        "channels": 1,
        "duration_seconds": round(
            len(normalized) / target_sample_rate,
            3,
        ),
        "normalized": True,
        "peak_before_normalization": round(peak_before, 6),
        "peak_after_normalization": round(peak_after, 6),
        "denoise_requested": denoise,
        "denoise_applied": denoise_applied,
        "resampler": resampler,
        "loader": loader,
        "warnings": warnings,
    }
