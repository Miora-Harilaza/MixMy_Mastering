from pathlib import Path
import numpy as np
import soundfile as sf
from .loudness import integrated_loudness

def analyze(path: str) -> tuple[np.ndarray, int, dict]:
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('Le fichier ne contient pas de signal audio valide.')
    duration = len(audio) / rate
    segment=audio[:min(len(audio),rate*30)].astype(np.float64)
    spectrum=np.mean(np.abs(np.fft.rfft(segment,axis=0))**2,axis=1)
    freqs=np.fft.rfftfreq(len(segment),1/rate)
    total = max(float(spectrum.sum()), 1e-20)
    def share(lo: float, hi: float) -> float:
        return float(spectrum[(freqs >= lo) & (freqs < hi)].sum() / total)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    lufs = integrated_loudness(audio.astype(np.float64), rate)
    count=min(240,len(audio))
    edges=np.linspace(0,len(audio),count+1,dtype=int)
    waveform=[float(np.max(np.abs(audio[edges[i]:max(edges[i]+1,edges[i+1])])) ) for i in range(count)]
    info = {'path':str(Path(path).resolve()),'name':Path(path).name,'size_bytes':Path(path).stat().st_size,'duration':duration,'sample_rate':rate,'channels':audio.shape[1],'frames':len(audio),'peak_dbfs':_db(peak),'rms_dbfs':_db(rms),'lufs':lufs,'bands':{'bass':share(20,250),'mids':share(250,4000),'highs':share(4000, min(rate/2,20000))},'clipping':peak >= .999,'dynamic_range_db':max(0.0,_db(peak)-_db(rms)),'waveform':waveform}
    return audio, rate, info

def _db(value: float) -> float:
    return 20 * np.log10(max(value, 1e-10))
