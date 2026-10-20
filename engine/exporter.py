from pathlib import Path
import os
import tempfile
import numpy as np
import soundfile as sf

def export_audio(path: str, audio: np.ndarray, sample_rate: int) -> None:
    dest=Path(path).expanduser().resolve()
    if dest.suffix.lower() not in ('.wav','.wave'): raise ValueError('Cette version exporte en WAV PCM 24 bits ou 16 bits.')
    if not np.isfinite(audio).all(): raise ValueError('Le rendu contient des valeurs audio invalides.')
    subtype='PCM_24'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='.mixmaster-',suffix='.wav',dir=dest.parent,delete=False) as f: temporary=f.name
        sf.write(temporary,audio,sample_rate,subtype=subtype,format='WAV')
        check,rate=sf.read(temporary,always_2d=True,dtype='float32')
        if rate!=sample_rate or check.shape!=audio.shape or check.size==0 or not np.isfinite(check).all(): raise ValueError('La validation du fichier exporté a échoué.')
        os.replace(temporary,dest); temporary=None
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)
