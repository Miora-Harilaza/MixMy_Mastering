import numpy as np
from .filters import highpass, peaking_eq
from .compressor import compress
from .dynamics import opto_compress, vocal_rider
from .limiter import limit
from .loudness import integrated_loudness

PRESETS={'Afrobeat Warm':{'bass':1.0,'air':0.9,'threshold':-15,'ratio':2.0},'Afrobeat Punchy':{'bass':0.2,'air':1.2,'threshold':-17,'ratio':2.2},'R&B Smooth':{'bass':0.5,'air':0.4,'threshold':-14,'ratio':1.8},'Vocal Clear':{'bass':-0.8,'air':1.3,'threshold':-16,'ratio':2.0},'Rap & Hip-Hop':{'bass':0.2,'air':0.9,'threshold':-17,'ratio':2.2},'Pop Bright':{'bass':-0.6,'air':1.4,'threshold':-16,'ratio':2.0},'Natural Master':{'bass':0.3,'air':0.3,'threshold':-14,'ratio':1.8},'Custom':{'bass':0.0,'air':0.0,'threshold':-14,'ratio':1.8}}

def master(audio: np.ndarray, sample_rate: int, bands: dict[str,float], preset: str, progress=None) -> tuple[np.ndarray,list[str]]:
    if preset not in PRESETS: raise ValueError('Preset inconnu.')
    params=PRESETS[preset]; warnings=[]
    input_loudness=integrated_loudness(audio.astype(np.float64),sample_rate)
    def tick(stage, percent):
        if progress: progress(stage,percent)
    tick('Analyse spectrale terminée',30)
    y=highpass(audio,sample_rate,24.0)
    bass_share=bands.get('bass',0)
    # Small adaptive attenuation only when low-frequency energy dominates.
    if bass_share>0.62:
        gain=max(-2.0,-min(2.0,(bass_share-0.62)*12))
        y=peaking_eq(y,sample_rate,145,gain,0.7)
        warnings.append(f'Basses dominantes détectées : correction prudente de {gain:.1f} dB.')
    y=peaking_eq(y,sample_rate,180,params['bass'],0.7)
    y=peaking_eq(y,sample_rate,10500,params['air'],0.7)
    tick('Égalisation tonale élégante',40)
    y=vocal_rider(y,sample_rate)
    tick('Nivellement automatique et fluide',52)
    y=opto_compress(y,sample_rate,threshold_db=-22.0,ratio=2.2)
    tick('Compression optique douce',66)
    y=compress(y,sample_rate,threshold_db=params['threshold'],
               ratio=params['ratio'],attack_ms=4.0,release_ms=100.0)
    tick('Densité et contrôle des crêtes',80)
    processed_loudness=integrated_loudness(y.astype(np.float64),sample_rate)
    if input_loudness is not None and processed_loudness is not None:
        makeup_db=float(np.clip(input_loudness-processed_loudness,-6.0,6.0))
        y=(y*10**(makeup_db/20)).astype(np.float32)
    y=limit(y,-1.0)
    tick('Limitation de sécurité',95)
    if not np.isfinite(y).all(): raise ValueError('Le traitement a produit des échantillons invalides.')
    if np.max(np.abs(y))>=1.0: warnings.append('Les crêtes approchent la limite numérique.')
    return y,warnings
