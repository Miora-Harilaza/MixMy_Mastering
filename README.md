# MixMaster AI

Application Windows de mastering audio local : interface Electron, React, TypeScript et Vite, moteur DSP Python.

## Démarrage en développement

Prérequis : Node.js 20+, Python 3.10+ et les dépendances audio.

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

Le moteur analyse l’amplitude, l’énergie spectrale et une mesure de loudness K pondérée fondée sur BS.1770, puis applique une chaîne DSP originale : nivellement automatique modéré, compression RMS à attaque lente, compression de crête, EQ et limitation de crête échantillon. Ces traitements s’appliquent au mix stéréo complet ; ils ne séparent pas les voix ni les instruments et ne reproduisent pas des algorithmes commerciaux. Le score LUFS est une approximation sans gate de programme ; ce n’est pas une mesure de conformité EBU R128. Le contrôle de crête n’est pas un limiteur true-peak. L’export actuel est WAV PCM 24 bits ; les exports 16 bits/MP3 et la configuration avancée sont des points d’extension.

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
