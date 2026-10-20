$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
Set-Location -LiteralPath $project

$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3 -m venv (Join-Path $project '.venv')
    } else {
        $systemPython = Get-Command 'python.exe' -ErrorAction SilentlyContinue
        if (-not $systemPython) { throw 'Python 3 est requis pour construire l’installateur.' }
        & $systemPython.Source -m venv (Join-Path $project '.venv')
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $python)) { throw 'Création de .venv impossible.' }
}

Write-Host 'Installation des dépendances du moteur et de PyInstaller...' -ForegroundColor Cyan
& $python -m pip install -r (Join-Path $project 'requirements.txt') -r (Join-Path $project 'requirements-build.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installation des dépendances Python impossible.' }

New-Item -ItemType Directory -Force -Path (Join-Path $project 'build\python') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $project '.runtime\pyinstaller') | Out-Null
Write-Host 'Construction du moteur Python autonome...' -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean --onedir --console --name mixmaster-engine --distpath (Join-Path $project 'build\python') --workpath (Join-Path $project '.runtime\pyinstaller') --specpath (Join-Path $project '.runtime\pyinstaller') --paths $project --collect-all soundfile (Join-Path $project 'engine_entry.py')
if ($LASTEXITCODE -ne 0) { throw 'Construction du moteur Python autonome impossible.' }

Write-Host 'Construction de l’application et de l’installateur Windows...' -ForegroundColor Cyan
& npm run build
if ($LASTEXITCODE -ne 0) { throw 'Construction de l’application impossible.' }

$electronDist = Join-Path $project 'node_modules\electron\dist'
if (-not (Test-Path -LiteralPath (Join-Path $electronDist 'electron.exe'))) {
    throw 'Runtime Electron introuvable. Exécutez npm install puis relancez la construction.'
}
$electronVersion = (Get-Content -LiteralPath (Join-Path $project 'node_modules\electron\package.json') -Raw | ConvertFrom-Json).version
$electronCache = Join-Path $project '.runtime\electron-dist'
New-Item -ItemType Directory -Force -Path $electronCache | Out-Null
$electronArchive = Join-Path $electronCache "electron-v$electronVersion-win32-x64.zip"
if (-not (Test-Path -LiteralPath $electronArchive)) {
    Write-Host 'Préparation du runtime Electron local...' -ForegroundColor Cyan
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($electronDist,$electronArchive,[System.IO.Compression.CompressionLevel]::Fastest,$false)
}

& npx electron-builder --win nsis --x64
if ($LASTEXITCODE -ne 0) { throw 'Création de l’installateur Windows impossible.' }

Write-Host "Installateur créé dans $project\release" -ForegroundColor Green
