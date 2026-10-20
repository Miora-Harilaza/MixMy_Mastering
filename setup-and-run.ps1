$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
Set-Location -LiteralPath $project

Write-Host 'MixMaster AI - preparation du projet' -ForegroundColor Cyan
$pythonExe = $null
$localPython = Join-Path $project '.runtime\python\python.exe'
if (Test-Path -LiteralPath $localPython) { $pythonExe = $localPython }

if (-not $pythonExe) {
    foreach ($commandName in @('py.exe', 'python.exe')) {
        $candidate = Get-Command $commandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($candidate -and $candidate.Source -notlike '*\Microsoft\WindowsApps\*') {
            try {
                & $candidate.Source --version *> $null
                if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate.Source; break }
            } catch { }
        }
    }
}

if (-not $pythonExe) {
    Write-Host 'Python introuvable. Telechargement de Python 3.13.16 depuis python.org...' -ForegroundColor Yellow
    $installer = Join-Path $env:TEMP 'mixmaster-python-3.13.16-amd64.exe'
    $pythonHome = Join-Path $project '.runtime\python'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $localPython) | Out-Null
    & curl.exe --location --fail --silent --show-error --connect-timeout 20 --max-time 600 --retry 3 --retry-delay 2 --output $installer 'https://www.python.org/ftp/python/3.13.16/python-3.13.16-amd64.exe'
    if ($LASTEXITCODE -ne 0) { throw 'Telechargement de Python impossible. Verifiez votre connexion puis relancez le script.' }
    $install = Start-Process -FilePath $installer -ArgumentList @('/quiet','InstallAllUsers=0','Include_launcher=0','Include_pip=1',"TargetDir=$pythonHome") -Wait -PassThru -WindowStyle Hidden
    if ($install.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $localPython)) { throw "Installation de Python echouee (code $($install.ExitCode))." }
    $pythonExe = $localPython
}

$node = Get-Command 'npm.cmd' -ErrorAction SilentlyContinue
if (-not $node) { throw 'Node.js est requis. Installez la version LTS depuis https://nodejs.org/ puis relancez ce script.' }

$venvPython = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Creation de l environnement Python...' -ForegroundColor Cyan
    & $pythonExe -m venv (Join-Path $project '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creation de .venv echouee.' }
}

Write-Host 'Installation des paquets audio...' -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Mise a jour de pip echouee.' }
& $venvPython -m pip install -r (Join-Path $project 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installation des paquets Python echouee.' }

Write-Host 'Installation des paquets de l application...' -ForegroundColor Cyan
& $node.Source install
if ($LASTEXITCODE -ne 0) { throw 'Installation npm echouee.' }

$electronExe = Join-Path $project 'node_modules\electron\dist\electron.exe'
if (-not (Test-Path -LiteralPath $electronExe)) {
    Write-Host 'Runtime Electron absent. Telechargement du binaire Windows...' -ForegroundColor Yellow
    & $node.Source rebuild electron
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $electronExe)) {
        throw 'Installation du runtime Electron echouee. Verifiez que GitHub est accessible, puis relancez le script.'
    }
}

$env:MIXMASTER_PYTHON = $venvPython
Write-Host 'Demarrage de MixMaster AI...' -ForegroundColor Green
& $node.Source run electron:dev
exit $LASTEXITCODE
