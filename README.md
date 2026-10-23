# MixMaster AI

Application Windows de mastering audio local : interface Electron, React, TypeScript et Vite, moteur DSP Python.

## Démarrage en développement

Prérequis : Node.js 20+, Python 3.10+ et les dépendances DSP Python du moteur. Le traitement vocal automatique utilise PyTorch, torchaudio, Demucs et librosa.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm install
npm run electron:dev
```

Si le lanceur `python` n’est pas dans le PATH, définissez `MIXMASTER_PYTHON` avec le chemin de l’interpréteur avant le démarrage.

Sous Windows, `powershell -ExecutionPolicy Bypass -File .\setup-and-run.ps1` automatise l’installation des dépendances. Si Python manque, le script télécharge Python 3.13.16 depuis python.org et l’installe localement dans `.runtime`.

## Moteur audio

Le mode AUTO choisit un profil équilibré à partir de la répartition spectrale mesurée ; il ne prétend pas classifier le genre musical. Les autres profils pilotent les corrections tonales et la dynamique du moteur. La séparation des stems et le traitement vocal dédié ne sont exécutés que si le profil « Afrobeat Vocal Studio » est sélectionné ; les masters ordinaires ne corrigent donc pas automatiquement la justesse et évitent les artefacts de séparation hors de ce mode. Une égalisation adaptative, un nivellement modéré, une compression de bus et une limitation true-peak complètent le traitement. Au premier lancement du profil vocal, une connexion Internet peut être nécessaire pour télécharger le modèle Demucs ; il est ensuite mis en cache, et l’analyse et le traitement restent locaux. Le moteur est un traitement original et ne prétend ni reproduire ni surpasser Ozone ou Waves. Le loudness reste une approximation sans gate de programme, pas une mesure de conformité EBU R128 ; le true peak 4x est une estimation DSP, pas une certification de conformité. L’export est WAV PCM 24 bits.

Les fichiers audio ne quittent pas la machine. Le moteur charge pour l’instant le signal complet en mémoire : les très longs fichiers peuvent nécessiter beaucoup de RAM.

## Tests DSP

```powershell
python -m pytest
```

## Créer l’installateur Windows

Sur Windows, installez Node.js 20+ et Python 3.10+, puis lancez :

```powershell
npm install
npm run electron:build
```

Le script installe les dépendances de construction, crée un moteur Python autonome, puis fabrique un installateur NSIS. Le fichier `release\MixMaster-AI-Setup-0.1.0.exe` peut être installé sur un PC Windows sans installer séparément Python ou Node.js. La construction nécessite une connexion Internet pour télécharger les dépendances.
