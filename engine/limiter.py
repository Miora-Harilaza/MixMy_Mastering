import numpy as np
from scipy.ndimage import maximum_filter1d
from .true_peak import true_peak

def limit(audio: np.ndarray, ceiling_db: float=-1.0,
          true_peak_margin_db: float=0.1, sample_rate: int=44100) -> np.ndarray:
    """Control sample peaks, then enforce a conservative oversampled true-peak ceiling."""
    if audio.size == 0:
        return audio
    ceiling=10**(ceiling_db/20); x=audio.astype(np.float64); peak=np.max(np.abs(x),axis=1)
    rate=max(1,int(sample_rate))
    lookahead=max(1,int(round(0.005*rate)))
    peak_envelope=maximum_filter1d(peak,size=lookahead,mode='nearest')
    target=np.minimum(1.0,ceiling/np.maximum(peak_envelope,1e-12))
    gain=np.empty_like(target); state=1.0
    release=np.exp(-1.0/max(1.0,0.12*rate))
    for i,desired in enumerate(target):
        state=desired if desired<state else 1.0-(1.0-state)*release
        gain[i]=state
    y=np.clip((x*gain[:,None]).astype(np.float32),-ceiling,ceiling)
    measured_true_peak=true_peak(y)
    safe_ceiling=10**((ceiling_db-true_peak_margin_db)/20)
    if measured_true_peak>safe_ceiling:
        y=(y*(safe_ceiling/measured_true_peak)).astype(np.float32)
    return y
