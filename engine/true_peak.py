import numpy as np
from scipy.signal import resample_poly


def true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    """Estimate inter-sample peak with chunked polyphase oversampling."""
    if audio.size == 0:
        return 0.0
    if oversample < 2:
        raise ValueError('Le suréchantillonnage true-peak doit être au moins 2x.')

    context = 64
    chunk_frames = 65536
    peak = 0.0
    for start in range(0, len(audio), chunk_frames):
        end = min(len(audio), start + chunk_frames)
        padded_start = max(0, start - context)
        padded_end = min(len(audio), end + context)
        block = resample_poly(
            audio[padded_start:padded_end].astype(np.float64),
            oversample, 1, axis=0,
        )
        core_start = (start - padded_start) * oversample
        core_end = core_start + (end - start) * oversample
        peak = max(peak, float(np.max(np.abs(block[core_start:core_end]))))
    return peak
