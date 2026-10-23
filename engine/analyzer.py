from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft
from .loudness import integrated_loudness
from .true_peak import true_peak

def analyze(path: str) -> tuple[np.ndarray, int, dict]:
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('Le fichier ne contient pas de signal audio valide.')
    duration = len(audio) / rate
    segment=audio[:min(len(audio),rate*30)].astype(np.float64)
    frequencies,_,spectrum_by_frame=stft(
        segment,fs=rate,nperseg=min(8192,len(segment)),
        noverlap=min(4096,max(0,min(8192,len(segment))//2)),
        axis=0,boundary=None,padded=False,
    )
    spectrum=np.mean(np.abs(spectrum_by_frame)**2,axis=(1,2))
    freqs=frequencies
    total = max(float(spectrum.sum()), 1e-20)
    def share(lo: float, hi: float) -> float:
        return float(spectrum[(freqs >= lo) & (freqs < hi)].sum() / total)
    peak = float(np.max(np.abs(audio)))
    true_peak_value = true_peak(audio)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    lufs = integrated_loudness(audio.astype(np.float64), rate)
    count=min(240,len(audio))
    edges=np.linspace(0,len(audio),count+1,dtype=int)
    waveform=[float(np.max(np.abs(audio[edges[i]:max(edges[i]+1,edges[i+1])])) ) for i in range(count)]
    bass=share(20,min(250,rate/2))
    low_mids=share(250,min(500,rate/2))
    mids=share(500,min(4000,rate/2))
    highs=share(4000,min(rate/2,20000))
    info = {'path':str(Path(path).resolve()),'name':Path(path).name,'size_bytes':Path(path).stat().st_size,'duration':duration,'sample_rate':rate,'channels':audio.shape[1],'frames':len(audio),'peak_dbfs':_db(peak),'true_peak_dbfs':_db(true_peak_value),'rms_dbfs':_db(rms),'crest_factor_db':max(0.0,_db(peak)-_db(rms)),'lufs':lufs,'bands':{'bass':bass,'low_mids':low_mids,'mids':mids,'highs':highs},'clipping':peak >= .999,'dynamic_range_db':max(0.0,_db(peak)-_db(rms)),'waveform':waveform}
    return audio, rate, info

def _db(value: float) -> float:
    return 20 * np.log10(max(value, 1e-10))
