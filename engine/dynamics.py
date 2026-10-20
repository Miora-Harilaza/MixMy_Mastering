import numpy as np
from scipy.ndimage import gaussian_filter1d


def vocal_rider(audio: np.ndarray, sample_rate: int, max_gain_db: float = 4.5) -> np.ndarray:
    """Smooth linked-channel level automation with a noise floor and bounded gain."""
    if audio.size == 0 or max_gain_db <= 0:
        return audio

    block_size = max(1, int(sample_rate * 0.05))
    block_count = (len(audio) + block_size - 1) // block_size
    levels = np.empty(block_count, dtype=np.float64)
    for index in range(block_count):
        block = audio[index * block_size:(index + 1) * block_size].astype(np.float64)
        rms = float(np.sqrt(np.mean(block * block)))
        levels[index] = 20 * np.log10(max(rms, 1e-10))

    active = levels > -50
    if not np.any(active):
        return audio

    reference = float(np.mean(levels[active]))
    gain_db = np.zeros(block_count, dtype=np.float64)
    gain_db[active] = np.clip(reference - levels[active], -max_gain_db, max_gain_db)
    sigma = max(1.0, 0.12 * sample_rate / block_size)
    gain_db = gaussian_filter1d(gain_db, sigma=sigma, mode='nearest')

    block_centres = (np.arange(block_count) + 0.5) * block_size
    sample_positions = np.arange(len(audio))
    sample_gain_db = np.interp(sample_positions, block_centres, gain_db,
                               left=gain_db[0], right=gain_db[-1])
    sample_gain = np.power(10.0, sample_gain_db / 20.0)
    return (audio.astype(np.float64) * sample_gain[:, None]).astype(np.float32)


def opto_compress(audio: np.ndarray, sample_rate: int, threshold_db: float = -22.0,
                  ratio: float = 2.2, attack_ms: float = 18.0,
                  release_ms: float = 180.0, knee_db: float = 6.0) -> np.ndarray:
    """Original RMS-based, slow program compressor for smooth level control."""
    if audio.size == 0 or sample_rate <= 0 or ratio <= 1:
        return audio

    x = audio.astype(np.float64)
    power = np.mean(x * x, axis=1)
    envelope = np.empty_like(power)
    attack = np.exp(-1.0 / max(1.0, attack_ms * sample_rate / 1000.0))
    release = np.exp(-1.0 / max(1.0, release_ms * sample_rate / 1000.0))
    previous = 0.0
    for index, value in enumerate(power):
        coefficient = attack if value > previous else release
        previous = coefficient * previous + (1.0 - coefficient) * value
        envelope[index] = previous

    level_db = 10.0 * np.log10(np.maximum(envelope, 1e-20))
    over = level_db - threshold_db
    slope = 1.0 - 1.0 / ratio
    half_knee = knee_db / 2.0
    reduction_db = np.where(
        over > half_knee,
        over * slope,
        np.where(over < -half_knee, 0.0,
                 (over + half_knee) ** 2 * slope / (2.0 * knee_db)),
    )
    gain = np.power(10.0, -reduction_db / 20.0)
    return (x * gain[:, None]).astype(np.float32)
