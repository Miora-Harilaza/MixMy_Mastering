import numpy as np

def compress(audio: np.ndarray, sample_rate: int, threshold_db: float=-14, ratio: float=1.7,
             makeup_db: float=0.0, attack_ms: float=4.0,
             release_ms: float=100.0) -> np.ndarray:
    """Linked-channel peak compressor with attack and release in milliseconds."""
    if audio.size == 0 or ratio <= 1:
        return audio
    x=audio.astype(np.float64); peak=np.max(np.abs(x),axis=1)
    env=np.empty_like(peak)
    attack_coef=np.exp(-1.0/max(1.0,attack_ms*1e-3*sample_rate))
    release_coef=np.exp(-1.0/max(1.0,release_ms*1e-3*sample_rate))
    prev=0.0
    for i,value in enumerate(peak):
        coef=attack_coef if value>prev else release_coef
        prev=coef*prev+(1-coef)*value
        env[i]=prev
    env_db=20*np.log10(np.maximum(env,1e-9)); over=np.maximum(env_db-threshold_db,0)
    reduction=over*(1-1/ratio); gain=10**((-reduction+makeup_db)/20)
    return (x*gain[:,None]).astype(np.float32)
