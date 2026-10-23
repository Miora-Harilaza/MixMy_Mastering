from functools import lru_cache
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .dynamics import opto_compress
from .filters import highpass, peaking_eq


KEY_NAMES = ('Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa', 'Fa#', 'Sol',
             'Sol#', 'La', 'La#', 'Si')
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@lru_cache(maxsize=1)
def _load_model():
    from demucs.pretrained import get_model

    model = get_model('htdemucs')
    model.to('cpu').eval()
    return model


def _separate(audio: np.ndarray, sample_rate: int,
              progress=None) -> tuple[np.ndarray, np.ndarray, int]:
    if audio.shape[1] > 2:
        raise ValueError(
            'Le mastering vocal automatique prend en charge les fichiers '
            'mono ou stéréo uniquement.'
        )
    try:
        import torch
        from demucs.apply import apply_model
    except ImportError as exc:
        raise RuntimeError(
            'Le mastering vocal automatique nécessite Demucs, PyTorch et '
            'torchaudio. Réinstallez MixMaster AI.'
        ) from exc

    if progress:
        progress('Chargement du modèle vocal local', 16)
    model = _load_model()
    source = audio if audio.shape[1] == 2 else np.repeat(audio, 2, axis=1)
    model_rate = int(model.samplerate)
    if sample_rate != model_rate:
        factor = gcd(sample_rate, model_rate)
        source = resample_poly(source, model_rate // factor,
                               sample_rate // factor, axis=0).astype(np.float32)

    waveform = torch.from_numpy(np.ascontiguousarray(source.T.copy()))
    reference = waveform.mean(dim=0)
    mean = reference.mean()
    scale = reference.std().clamp(min=1e-8)
    normalized = ((waveform - mean) / scale)[None]
    if progress:
        progress('Séparation automatique de la voix et de l’instrumental', 20)
    with torch.inference_mode():
        stems = apply_model(
            model,
            normalized,
            device='cpu',
            shifts=2,
            split=True,
            overlap=0.75,
            progress=False,
        )[0]
    if stems.ndim != 3 or stems.shape[0] < 2:
        raise RuntimeError('La séparation Demucs ne retourne pas de stems exploitables.')

    stems = (stems * scale + mean).cpu().numpy().astype(np.float32)
    vocal_index = model.sources.index('vocals')
    if vocal_index >= stems.shape[0]:
        raise RuntimeError('Le modèle Demucs ne fournit pas de stem vocal exploitable.')

    vocals = stems[vocal_index].T
    instrumental = np.delete(stems, vocal_index, axis=0).sum(axis=0).T

    if sample_rate != model_rate:
        factor = gcd(sample_rate, model_rate)
        vocals = resample_poly(vocals, sample_rate // factor,
                               model_rate // factor, axis=0).astype(np.float32)
        instrumental = resample_poly(
            instrumental, sample_rate // factor, model_rate // factor,
            axis=0,
        ).astype(np.float32)

    length = len(audio)
    if len(vocals) < length:
        padding = length - len(vocals)
        vocals = np.pad(vocals, ((0, padding), (0, 0)))
        instrumental = np.pad(instrumental, ((0, padding), (0, 0)))
    vocals = np.clip(vocals[:length], -1.0, 1.0).astype(np.float32)
    instrumental = np.clip(instrumental[:length], -1.0, 1.0).astype(np.float32)
    if not np.isfinite(vocals).all() or not np.isfinite(instrumental).all():
        raise RuntimeError('La séparation Demucs a produit des échantillons invalides.')
    return vocals, instrumental, sample_rate


def _estimate_key(vocals: np.ndarray, sample_rate: int) -> tuple[int, str]:
    import librosa

    mono = vocals[:sample_rate * 60].mean(axis=1).astype(np.float32)
    if len(mono) < sample_rate // 2 or float(np.max(np.abs(mono))) < 1e-5:
        return 0, 'Indéterminée'
    analysis_rate = 22050
    if sample_rate != analysis_rate:
        factor = gcd(sample_rate, analysis_rate)
        mono = resample_poly(mono, analysis_rate // factor,
                             sample_rate // factor).astype(np.float32)
    chroma = librosa.feature.chroma_stft(y=mono, sr=analysis_rate,
                                         n_fft=4096, hop_length=1024)
    profile = chroma.mean(axis=1)
    candidates = []
    for mode, template in (('majeur', MAJOR_PROFILE), ('mineur', MINOR_PROFILE)):
        for root in range(12):
            score = float(np.corrcoef(profile, np.roll(template, root))[0, 1])
            candidates.append((score, root, mode))
    score, root, mode = max(candidates)
    if not np.isfinite(score) or score < 0.12:
        return 0, 'Indéterminée'
    return root, f'{KEY_NAMES[root]} {mode}'


def _correct_pitch(vocals: np.ndarray, sample_rate: int, root: int,
                   mode: str, progress=None) -> np.ndarray:
    import librosa

    analysis_rate = 22050
    mono = vocals.mean(axis=1).astype(np.float32)
    if sample_rate != analysis_rate:
        factor = gcd(sample_rate, analysis_rate)
        analysis = resample_poly(mono, analysis_rate // factor,
                                 sample_rate // factor).astype(np.float32)
    else:
        analysis = mono

    hop = 512
    pitches, voiced, probabilities = librosa.pyin(
        analysis, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'), sr=analysis_rate,
        frame_length=2048, hop_length=hop,
    )
    if not len(pitches):
        return vocals
    key_scale = (0, 2, 4, 5, 7, 9, 11) if mode == 'majeur' else (
        0, 2, 3, 5, 7, 8, 10)
    allowed = np.array([
        root + interval + octave * 12
        for octave in range(2, 8) for interval in key_scale
    ], dtype=np.float64)
    midi = librosa.hz_to_midi(np.maximum(np.nan_to_num(pitches, nan=0), 1e-6))
    valid = voiced & np.isfinite(pitches) & (probabilities >= 0.75)
    corrections = np.zeros(len(pitches), dtype=np.float64)
    nearest = np.abs(midi[valid, None] - allowed[None, :]).argmin(axis=1)
    corrections[valid] = np.clip(
        allowed[nearest] - midi[valid], -2.0, 2.0,
    ) * 0.75

    frame = max(1, int(analysis_rate * 0.36))
    hop_samples = frame // 2
    window = np.hanning(frame).astype(np.float64)
    output = np.zeros_like(vocals, dtype=np.float64)
    weights = np.zeros(len(vocals), dtype=np.float64)
    for start in range(0, len(vocals), hop_samples):
        end = min(start + frame, len(vocals))
        segment = vocals[start:end]
        center = start + (end - start) / 2
        analysis_center = center * analysis_rate / sample_rate
        pitch_index = min(len(corrections) - 1,
                          max(0, int(analysis_center / hop)))
        shift = float(corrections[pitch_index])
        if shift and valid[pitch_index] and len(segment) >= 2048:
            shifted = librosa.effects.pitch_shift(
                segment.T, sr=sample_rate, n_steps=shift,
            ).T
            segment = shifted[:len(segment)]
        local_window = window[:len(segment)]
        output[start:end] += segment * local_window[:, None]
        weights[start:end] += local_window
    if progress:
        progress('Correction douce de justesse vocale', 56)
    return (output / np.maximum(weights[:, None], 1e-8)).astype(np.float32)


def _balance_gain(vocals: np.ndarray, instrumental: np.ndarray) -> float:
    vocal_rms = float(np.sqrt(np.mean(vocals.astype(np.float64) ** 2)))
    instrumental_rms = float(np.sqrt(np.mean(instrumental.astype(np.float64) ** 2)))
    if vocal_rms < 1e-8 or instrumental_rms < 1e-8:
        return 1.0
    target_ratio = 0.72
    gain_db = np.clip(
        20 * np.log10(target_ratio * instrumental_rms / vocal_rms),
        0.0, 4.0,
    )
    return float(10 ** (gain_db / 20))


def enhance_vocals(audio: np.ndarray, sample_rate: int, progress=None
                   ) -> tuple[np.ndarray, list[str]]:
    """Tune, balance, compress, and gently shape the vocal before mastering."""
    vocals, instrumental, _ = _separate(audio, sample_rate, progress)
    root, key = _estimate_key(vocals, sample_rate)
    if key == 'Indéterminée':
        tuned = vocals
        warnings = [
            'Tonalité vocale indéterminée : correction de justesse ignorée '
            'pour éviter une correction hasardeuse.'
        ]
    else:
        mode = key.rsplit(' ', 1)[-1]
        tuned = _correct_pitch(vocals, sample_rate, root, mode, progress)
        warnings = [f'Correction vocale automatique en {key}.']

    vocal = opto_compress(tuned, sample_rate, threshold_db=-25.0,
                          ratio=2.0, attack_ms=18.0, release_ms=240.0,
                          knee_db=8.0)
    vocal = highpass(vocal, sample_rate, 75.0)
    vocal = peaking_eq(vocal, sample_rate, 280.0, -0.8, 0.9)
    vocal = peaking_eq(vocal, sample_rate, 3200.0, 1.2, 0.9)
    vocal = peaking_eq(vocal, sample_rate, 10500.0, 0.5, 0.8)
    vocal *= _balance_gain(vocal, instrumental)
    enhanced = vocal + instrumental
    if audio.shape[1] == 1:
        enhanced = enhanced.mean(axis=1, keepdims=True)
    if not np.isfinite(enhanced).all():
        raise ValueError('Le traitement vocal a produit un signal invalide.')
    if progress:
        progress('Voix équilibrée et mise en avant', 64)
    return enhanced.astype(np.float32), warnings
