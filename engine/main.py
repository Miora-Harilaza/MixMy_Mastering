import argparse, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from .analyzer import analyze
from .mastering import master
from .exporter import export_audio

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
    def progress(stage,percent): emit({'type':'progress','stage':stage,'percent':percent})
    processed,warnings=master(audio,rate,before['bands'],payload.get('preset','Natural Master'),progress)
    emit({'type':'progress','stage':'Écriture et validation du fichier','percent':95})
    export_audio(str(dest),processed,rate)
    _,_,after=analyze(str(dest))
    emit({'type':'progress','stage':'Mastering terminé','percent':100})
    return {'input':before,'output':after,'output_path':str(dest),'preset':payload['preset'],'warnings':warnings}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--json',required=True); args=parser.parse_args()
    try: emit({'type':'result','data':run(json.loads(args.json))})
    except Exception as exc: emit({'type':'error','message':str(exc)}); sys.exit(1)
if __name__=='__main__': main()
