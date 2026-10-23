import numpy as np
from scipy.signal import butter, sosfiltfilt

def highpass(audio: np.ndarray, sample_rate: int, cutoff: float = 24.0) -> np.ndarray:
    cutoff=min(max(cutoff, 18), sample_rate*0.45)
    sos=butter(2,cutoff,btype='highpass',fs=sample_rate,output='sos')
    return sosfiltfilt(sos,audio,axis=0).astype(np.float32)

def peaking_eq(audio: np.ndarray, sample_rate: int, frequency: float, gain_db: float, q: float=0.8) -> np.ndarray:
    if abs(gain_db)<0.05 or frequency >= sample_rate/2: return audio
    # RBJ peaking EQ biquad.
    A=10**(gain_db/40); w=2*np.pi*frequency/sample_rate; alpha=np.sin(w)/(2*q); c=np.cos(w)
    bb=np.array([1+alpha*A,-2*c,1-alpha*A])/(1+alpha/A)
    aa=np.array([1+alpha/A,-2*c,1-alpha/A])/(1+alpha/A)
    from scipy.signal import lfilter
    return lfilter(bb,aa,audio,axis=0).astype(np.float32)
