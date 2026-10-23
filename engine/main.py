import argparse, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from .analyzer import analyze
from .mastering import master
from .exporter import export_audio
from .vocal import enhance_vocals

def emit(obj): print(json.dumps(obj,ensure_ascii=False),flush=True)
def run(payload):
    action=payload.get('action'); source=Path(payload.get('input','')).expanduser().resolve()
    if not source.is_file(): raise ValueError('Le fichier audio sélectionné est introuvable.')
    if action=='preview':
        audio,rate=sf.read(str(source),dtype='float32',always_2d=True)
        if audio.size==0 or not np.isfinite(audio).all():
            raise ValueError('Le fichier ne contient pas de signal audio valide.')
        dest=Path(payload.get('output','')).expanduser().resolve()
        if dest.suffix.lower() not in ('.wav','.wave'):
            raise ValueError('Le format de l’aperçu doit être WAV.')
        export_audio(str(dest),audio,rate)
        return {'path':str(dest)}
    emit({'type':'progress','stage':'Lecture et analyse du signal','percent':10})
    audio,rate,before=analyze(str(source))
    if action=='analyze': return before
    if action!='master': raise ValueError('Action moteur inconnue.')
    dest=Path(payload.get('output','')).expanduser().resolve()
    if dest==source: raise ValueError('Le fichier de sortie doit être différent de l’original.')
    if dest.suffix.lower() not in ('.wav','.wave'): raise ValueError('Format indisponible : choisissez WAV.')
    preset=payload.get('preset','AUTO')
    def progress(stage,percent): emit({'type':'progress','stage':stage,'percent':percent})
    if preset == 'Afrobeat Vocal Studio':
        enhanced,vocal_warnings=enhance_vocals(audio,rate,progress)
    else:
        enhanced,vocal_warnings=audio,[]
    def mastering_progress(stage,percent):
        progress(stage,64+round(percent*0.30))
    processed,warnings=master(
        enhanced,rate,before['bands'],preset,mastering_progress,
    )
    warnings=vocal_warnings+warnings
    emit({'type':'progress','stage':'Écriture et validation du fichier','percent':95})
    export_audio(str(dest),processed,rate)
    _,_,after=analyze(str(dest))
    emit({'type':'progress','stage':'Mastering terminé','percent':100})
    loudness_change=(after['lufs']-before['lufs']) if before['lufs'] is not None and after['lufs'] is not None else None
    return {
        'input':before,'output':after,'output_path':str(dest),'preset':preset,
        'warnings':warnings,
        'report':{
            'loudness_change_lu':loudness_change,
            'true_peak_change_db':after['true_peak_dbfs']-before['true_peak_dbfs'],
            'crest_factor_change_db':after['crest_factor_db']-before['crest_factor_db'],
            'corrections':warnings,
        },
    }

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--json',required=True); args=parser.parse_args()
    try: emit({'type':'result','data':run(json.loads(args.json))})
    except Exception as exc: emit({'type':'error','message':str(exc)}); sys.exit(1)
if __name__=='__main__': main()
