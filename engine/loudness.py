import numpy as np
from scipy.signal import lfilter

def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float | None:
    """Approximate BS.1770 K-weighted loudness; block gating is intentionally omitted."""
    if audio.size == 0 or not np.isfinite(audio).all(): return None
    # BS.1770 K-weighting consists of a high-shelf and RLB high-pass stage.
    shelf_b, shelf_a = _shelf(sample_rate)
    hp_b, hp_a = _highpass(sample_rate)
    weighted = lfilter(hp_b, hp_a, lfilter(shelf_b, shelf_a, audio, axis=0), axis=0)
    channel_energy = np.mean(weighted ** 2, axis=0)
    # Standard surround channel weighting (LFE excluded); stereo defaults to unity.
    weights = np.ones(len(channel_energy));
    if len(weights) > 3: weights[3] = 0
    loudness = -0.691 + 10*np.log10(max(float(np.sum(channel_energy*weights)), 1e-20))
    return float(loudness)

def _shelf(fs: int):
    # RBJ high shelf, +4 dB at the BS.1770 K-weighting corner.
    f0=1681.974450955533; gain=3.99984385397; q=0.7071752369554196
    A=10**(gain/40); w=2*np.pi*f0/fs; c=np.cos(w); s=np.sin(w); alpha=s/(2*q); root=2*np.sqrt(A)*alpha
    b=np.array([A*((A+1)+(A-1)*c+root),-2*A*((A-1)+(A+1)*c),A*((A+1)+(A-1)*c-root)])
    a=np.array([(A+1)-(A-1)*c+root,2*((A-1)-(A+1)*c),(A+1)-(A-1)*c-root])
    return b/a[0],a/a[0]

def _highpass(fs: int):
    from scipy.signal import butter
    return butter(2, 38, btype='highpass', fs=fs)
