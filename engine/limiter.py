import numpy as np

def limit(audio: np.ndarray, ceiling_db: float=-1.0) -> np.ndarray:
    """Conservative sample-peak safety limiter. This does not claim true-peak control."""
    ceiling=10**(ceiling_db/20); x=audio.astype(np.float64); peak=np.max(np.abs(x),axis=1)
    target=np.minimum(1.0,ceiling/np.maximum(peak,1e-12))
    # Immediate linked-channel attack and a short, controlled release.
    gain=np.empty_like(target); state=1.0
    for i,desired in enumerate(target):
        state=min(desired,state+0.002) if desired>=state else desired
        gain[i]=state
    y=(x*gain[:,None]).astype(np.float32)
    return np.clip(y,-ceiling,ceiling)
