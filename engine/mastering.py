import numpy as np
from .filters import highpass, peaking_eq
from .compressor import compress
from .dynamics import opto_compress, program_leveler
from .limiter import limit
from .loudness import integrated_loudness

PRESETS={
    'AUTO':{'bass':0.2,'air':0.2,'threshold':-14,'ratio':1.6},
    'Balanced':{'bass':0.2,'air':0.2,'threshold':-14,'ratio':1.6},
    'Afrobeat':{'bass':0.3,'air':0.5,'threshold':-15,'ratio':1.8},
    'R&B':{'bass':0.2,'air':0.2,'threshold':-14,'ratio':1.7},
    'Vocal Clarity':{
        'bass':-0.2,'air':0.3,'presence':1.5,'threshold':-16,'ratio':1.9,
        'opto_threshold_db':-22,'opto_ratio':1.9,
    },
    'Warm & Smooth':{'bass':0.3,'air':-0.1,'threshold':-14,'ratio':1.6},
    'Powerful':{'bass':0.1,'air':0.3,'threshold':-16,'ratio':2.0},
    'AI Music Polish':{'bass':0.0,'air':0.2,'threshold':-14,'ratio':1.7},
    'Afrobeat Warm':{'bass':1.0,'air':0.9,'threshold':-15,'ratio':2.0},
    'Afrobeat Punchy':{'bass':0.2,'air':1.2,'threshold':-17,'ratio':2.2},
    'R&B Smooth':{'bass':0.5,'air':0.4,'threshold':-14,'ratio':1.8},
    'Vocal Clear':{'bass':-0.8,'air':1.3,'threshold':-16,'ratio':2.0},
    'Rap & Hip-Hop':{'bass':0.2,'air':0.9,'threshold':-17,'ratio':2.2},
    'Pop Bright':{'bass':-0.6,'air':1.4,'threshold':-16,'ratio':2.0},
    'Natural Master':{'bass':0.3,'air':0.3,'threshold':-14,'ratio':1.8},
    'Afrobeat Vocal Studio':{
        'bass':0.15,'air':0.3,'threshold':-13,'ratio':1.6,
        'presence':0.9,
        'rider_max_gain_db':0.7,'opto_threshold_db':-19,'opto_ratio':1.7,
        'opto_attack_ms':35,'opto_release_ms':420,'opto_knee_db':10,
        'peak_attack_ms':16,'peak_release_ms':250,
    },
    'Custom':{'bass':0.0,'air':0.0,'threshold':-14,'ratio':1.8},
}

def master(audio: np.ndarray, sample_rate: int, bands: dict[str,float], preset: str, progress=None) -> tuple[np.ndarray,list[str]]:
    if preset not in PRESETS: raise ValueError('Preset inconnu.')
    warnings=[]
    if preset == 'AUTO':
        bass_share=bands.get('bass',0.0)
        low_mid_share=bands.get('low_mids',0.0)
        high_share=bands.get('highs',0.0)
        if bass_share > 0.58:
            resolved_preset='Afrobeat'
        elif low_mid_share > 0.24 and high_share < 0.22:
            resolved_preset='R&B'
        elif high_share > 0.34:
            resolved_preset='Warm & Smooth'
        else:
            resolved_preset='Balanced'
        warnings.append(f'Profil AUTO sélectionné : {resolved_preset}.')
    else:
        resolved_preset=preset
    params=PRESETS[resolved_preset]
    input_loudness=integrated_loudness(audio.astype(np.float64),sample_rate)
    def tick(stage, percent):
        if progress: progress(stage,percent)
    tick('Analyse spectrale terminée',30)
    y=highpass(audio,sample_rate,24.0)
    bass_share=bands.get('bass',0)
    low_mid_share=bands.get('low_mids',0)
    # Small adaptive attenuation only when low-frequency energy dominates.
    if bass_share>0.62:
        gain=max(-2.0,-min(2.0,(bass_share-0.62)*12))
        y=peaking_eq(y,sample_rate,145,gain,0.7)
        warnings.append(f'Basses dominantes détectées : correction prudente de {gain:.1f} dB.')
    mud_gain=-min(1.0,max(0.0,(low_mid_share-0.18)*5.0))
    if mud_gain < -0.15:
        y=peaking_eq(y,sample_rate,350,mud_gain,0.8)
        warnings.append(f'Bas-médiums chargés : correction douce de {mud_gain:.1f} dB.')
    bass_profile_gain=params['bass']*float(np.clip((0.5-bass_share)/0.3,0.0,1.0))
    high_share=bands.get('highs',0)
    air_profile_gain=params['air']*float(np.clip((0.28-high_share)/0.2,0.0,1.0))
    y=peaking_eq(y,sample_rate,180,bass_profile_gain,0.7)
    y=peaking_eq(y,sample_rate,10500,air_profile_gain,0.7)
    presence_gain=params.get('presence',0.0)*float(np.clip(
        1.0-low_mid_share,0.45,1.0))
    y=peaking_eq(y,sample_rate,3200,presence_gain,0.9)
    tick('Égalisation tonale élégante',40)
    y=program_leveler(
        y,sample_rate,
        max_boost_db=params.get('rider_max_gain_db',1.5),
        max_reduction_db=params.get('rider_max_reduction_db',2.0),
    )
    tick('Nivellement des passages et des silences',52)
    y=opto_compress(y,sample_rate,
                    threshold_db=params.get('opto_threshold_db',-18.0),
                    ratio=params.get('opto_ratio',1.6),
                    attack_ms=params.get('opto_attack_ms',30.0),
                    release_ms=params.get('opto_release_ms',350.0),
                    knee_db=params.get('opto_knee_db',9.0))
    tick('Compression optique douce',66)
    y=compress(y,sample_rate,threshold_db=params['threshold'],
               ratio=params['ratio'],
               attack_ms=params.get('peak_attack_ms',8.0),
               release_ms=params.get('peak_release_ms',180.0))
    tick('Densité et contrôle des crêtes',80)
    processed_loudness=integrated_loudness(y.astype(np.float64),sample_rate)
    if input_loudness is not None and processed_loudness is not None:
        makeup_db=float(np.clip(input_loudness-processed_loudness,-6.0,6.0))
        y=(y*10**(makeup_db/20)).astype(np.float32)
    y=limit(y,-1.0,sample_rate=sample_rate)
    tick('Limitation de sécurité',95)
    if not np.isfinite(y).all(): raise ValueError('Le traitement a produit des échantillons invalides.')
    if np.max(np.abs(y))>=1.0: warnings.append('Les crêtes approchent la limite numérique.')
    return y,warnings
