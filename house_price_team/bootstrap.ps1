param(
    [ValidateSet('all','extract','compare')][string]$Mode = 'all',
    [ValidateSet('all','baseline','v1','v2','v3')][string]$Method = 'all',
    [string]$PythonPath = $env:TEAM_PYTHON,
    [string]$PackageIndex = 'https://pypi.tuna.tsinghua.edu.cn/simple',
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cu128',
    [string]$RuntimeMirrorUrl = ''
)
# Windows PowerShell 5.1 and Windows 10/11 x64. This script never installs globally.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$script:stage = 'startup'
$script:exitCode = 1
$lock = $null
$transcriptStarted = $false
$runtimeName = 'python-3.11.13-windows-x86_64.tar.gz'
$runtimeSource = 'https://github.com/astral-sh/python-build-standalone/releases/download/20250918/cpython-3.11.13%2B20250918-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
$runtimeSha256 = 'a3a2f8ac71045d7bcf80366a7ec2cc5a1ba08112182a59e5df08a10242b8460a'
$runtimeExeSha256 = '068a47360cfac59cb307cfff9c9cc9daea4a2d25fa3f9ddab0c1791ca7823da8'
$bootstrap = Join-Path $root '.bootstrap'
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $root ('logs\bootstrap_' + $stamp + '_' + [Guid]::NewGuid().ToString('N').Substring(0,6) + '.log')

function Write-Stage([string]$name, [string]$message) {
    $script:stage = $name
    Write-Host ("`n[" + $name + '] ' + $message)
}

function Assert-Https([string]$value) {
    $uri = $null
    if (![Uri]::TryCreate($value, [UriKind]::Absolute, [ref]$uri) -or $uri.Scheme -ne 'https') {
        throw 'Download and package URLs must use HTTPS.'
    }
}

function Quote-Native([string]$value) {
    # Windows CommandLineToArgvW escaping, including a trailing backslash.
    $escaped = [regex]::Replace($value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    if (!(Test-Path -LiteralPath $Executable -PathType Leaf)) { throw ('Executable is missing: ' + $Executable) }
    $global:LASTEXITCODE = 9009
    # Do not treat native stderr warnings as PowerShell terminating errors.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | ForEach-Object { Write-Host ([string]$_) }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($null -eq $code -or $code -ne 0) { throw ('Command failed with exit code ' + $code + ': ' + $Executable) }
}

function Probe-Python([string]$Executable, [string[]]$Prefix = @()) {
    if (!$Executable -or !(Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    if ($Executable -like '*\Microsoft\WindowsApps\*') { return $null }
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $probe = Join-Path $bootstrap 'probe_python.py'
    $arguments = @($Prefix) + @($probe)
    $info.Arguments = (($arguments | ForEach-Object { Quote-Native ([string]$_) }) -join ' ')
    $info.WorkingDirectory = $root
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $proc = New-Object Diagnostics.Process
    $proc.StartInfo = $info
    try {
        if (!$proc.Start()) { return $null }
        if (!$proc.WaitForExit(15000)) { $proc.Kill(); $proc.WaitForExit(); return $null }
        $proc.WaitForExit()
        $stdout = $proc.StandardOutput.ReadToEnd()
        if ($proc.ExitCode -ne 0) { return $null }
        $result = $stdout | ConvertFrom-Json
        if ($result.valid) { return $result }
    } catch { return $null } finally { $proc.Dispose() }
    return $null
}

function Resolve-Python {
    if ($PythonPath) {
        $candidate = Probe-Python $PythonPath
        if (!$candidate) { throw 'TEAM_PYTHON / -PythonPath must point to a working 64-bit Python 3.11 or 3.12 executable.' }
        return $candidate.path
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        foreach ($version in @('-3.11','-3.12')) {
            $candidate = Probe-Python $launcher.Source @($version)
            if ($candidate) { return $candidate.path }
        }
    }
    foreach ($name in @('python.exe','python3.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $candidate = Probe-Python $command.Source
            if ($candidate) { return $candidate.path }
        }
    }
    $portable = Join-Path $bootstrap 'python\python.exe'
    if ((Test-Path -LiteralPath $portable -PathType Leaf) -and ((Get-FileHash -LiteralPath $portable -Algorithm SHA256).Hash.ToLower() -eq $runtimeExeSha256)) {
        $candidate = Probe-Python $portable
        if ($candidate) { return $candidate.path }
    }
    Write-Host 'No suitable Python found. Preparing a private portable Python, without administrator access.'
    $archive = Join-Path $root ('runtime\' + $runtimeName)
    if (!(Test-Path -LiteralPath $archive -PathType Leaf)) {
        $archive = Join-Path $bootstrap $runtimeName
        if (!(Test-Path -LiteralPath $archive -PathType Leaf)) {
            $url = $runtimeSource
            if ($RuntimeMirrorUrl) { Assert-Https $RuntimeMirrorUrl; $url = $RuntimeMirrorUrl }
            $temporary = $archive + '.download'
            Write-Host ('Official source: ' + $runtimeSource)
            Write-Host ('Downloading: ' + $url)
            Write-Host ('Expected SHA-256: ' + $runtimeSha256)
            $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
            if ($null -ne $curl) {
                Invoke-Checked $curl.Source @('--fail','--location','--retry','2','--connect-timeout','20','--max-time','600','--output',$temporary,$url)
            } else {
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                Invoke-WebRequest -Uri $url -OutFile $temporary -UseBasicParsing -TimeoutSec 600
            }
            if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLower() -ne $runtimeSha256) {
                throw 'Downloaded Python archive checksum mismatch. Nothing was extracted or executed.'
            }
            Move-Item -LiteralPath $temporary -Destination $archive -Force
        }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $runtimeSha256) {
        throw 'Portable Python archive checksum mismatch. Restore the verified archive or install 64-bit Python 3.11/3.12 from python.org.'
    }
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($null -eq $tar) { throw 'Windows tar.exe is missing. Install 64-bit Python 3.11/3.12 from python.org for your user, then run again.' }
    $unpack = Join-Path $bootstrap ('unpack_' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $unpack | Out-Null
    Invoke-Checked $tar.Source @('-xzf',$archive,'-C',$unpack)
    $extracted = Join-Path $unpack 'python\python.exe'
    if (!(Test-Path -LiteralPath $extracted) -or ((Get-FileHash -LiteralPath $extracted -Algorithm SHA256).Hash.ToLower() -ne $runtimeExeSha256)) {
        throw 'Extracted Python executable checksum mismatch.'
    }
    $destination = Join-Path $bootstrap 'python'
    if (Test-Path -LiteralPath $destination) {
        Move-Item -LiteralPath $destination -Destination (Join-Path $bootstrap ('python_previous_' + [Guid]::NewGuid().ToString('N')))
    }
    Move-Item -LiteralPath (Join-Path $unpack 'python') -Destination $destination
    $candidate = Probe-Python $portable
    if (!$candidate) { throw 'Portable Python could not run. The Windows runtime may be incomplete; see the bootstrap log.' }
    return $candidate.path
}

function Test-OfflineWheelhouse {
    $directory = Join-Path $root 'wheelhouse'
    $manifestPath = Join-Path $root 'WHEELHOUSE_MANIFEST.json'
    if (!(Test-Path -LiteralPath $directory -PathType Container)) { return $false }
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        Write-Host 'Offline wheelhouse has no verification manifest; using the configured package index instead.'
        return $false
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $listed = @($manifest.files.PSObject.Properties)
    if ($listed.Count -eq 0) { throw 'Offline wheelhouse manifest is empty.' }
    $verified = @{}
    foreach ($item in $listed) {
        $name = $item.Name
        if ([IO.Path]::GetFileName($name) -ne $name -or !$name.EndsWith('.whl', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Offline wheelhouse manifest contains an unsafe filename.'
        }
        $path = Join-Path $directory $name
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw ('Offline wheel is missing: ' + $name) }
        if ((Get-Item -LiteralPath $path).Length -ne $item.Value.bytes) { throw ('Offline wheel has the wrong size: ' + $name) }
        Write-Host ('Verifying offline wheel: ' + $name)
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $item.Value.sha256) { throw ('Offline wheel checksum mismatch: ' + $name) }
        $verified[$name] = $true
    }
    foreach ($file in Get-ChildItem -LiteralPath $directory -File) {
        if (!$verified.ContainsKey($file.Name)) { throw ('Offline wheelhouse contains an unverified file: ' + $file.Name) }
    }
    return $true
}

function Install-Requirements([string]$File) {
    if ($File -eq 'requirements.txt' -and (Test-OfflineWheelhouse)) {
        try {
            Write-Host 'Installing verified local base wheels without network access.'
            Invoke-Checked $python @('-m','pip','--isolated','install','--disable-pip-version-check','--no-index','--find-links',(Join-Path $root 'wheelhouse'),'-r',(Join-Path $root $File))
            return
        } catch {
            Write-Host 'Local wheels could not satisfy this interpreter (for example Python 3.12 with 3.11 wheels); trying the configured online package index.'
            Write-Host $_.Exception.Message
        }
    }
    $indexes = @($PackageIndex)
    if ($PackageIndex -ne 'https://pypi.org/simple') { $indexes += 'https://pypi.org/simple' }
    $failure = $null
    foreach ($index in $indexes) {
        try {
            Write-Host ('Package source: ' + $index)
            Invoke-Checked $python @('-m','pip','--isolated','install','--disable-pip-version-check','--only-binary=:all:','--timeout','30','--retries','1','--index-url',$index,'-r',(Join-Path $root $File))
            return
        } catch { $failure = $_; Write-Host ('Package source failed: ' + $_.Exception.Message) }
    }
    throw $failure
}

try {
    Set-Location -LiteralPath $root
    if (![Environment]::Is64BitOperatingSystem) { throw 'This package requires 64-bit Windows.' }
    if ($root -match '\.zip([\\.]|$)') { throw 'Extract the whole ZIP before starting; do not run inside the ZIP preview.' }
    New-Item -ItemType Directory -Path $bootstrap -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $root 'logs') -Force | Out-Null
    $lock = [IO.File]::Open((Join-Path $root '.team_running.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    Start-Transcript -LiteralPath $log -Force | Out-Null
    $transcriptStarted = $true
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $OutputEncoding = [Console]::OutputEncoding
    $env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONNOUSERSITE = '1'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'; $env:TOKENIZERS_PARALLELISM = 'false'
    $env:OMP_NUM_THREADS = '2'; $env:OPENBLAS_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Assert-Https $PackageIndex; Assert-Https $TorchIndex
    @'
import json, sys, struct, platform
valid = sys.version_info[:2] in [(3,11),(3,12)] and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64','x86_64')
print(json.dumps({'valid':valid,'path':sys.executable,'version':sys.version,'prefix':sys.prefix},ensure_ascii=True))
raise SystemExit(0 if valid else 3)
'@ | Set-Content -LiteralPath (Join-Path $bootstrap 'probe_python.py') -Encoding UTF8

    Write-Stage '1/5 Python' 'Checking existing Python 3.11/3.12 and private virtual environment.'
    $existing = Probe-Python $python
    if ($existing -and !([IO.Path]::GetFullPath($existing.prefix).TrimEnd('\').Equals([IO.Path]::GetFullPath($venv).TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase))) { $existing = $null }
    if (!$existing) {
        $basePython = Resolve-Python
        Write-Host ('Base Python: ' + $basePython)
        if (Test-Path -LiteralPath $venv) {
            $previousVenv = Join-Path $root ('.venv_previous_' + $stamp + '_' + [Guid]::NewGuid().ToString('N').Substring(0,6))
            Move-Item -LiteralPath $venv -Destination $previousVenv
            Write-Host ('Previous unusable environment retained: ' + $previousVenv)
        }
        Invoke-Checked $basePython @('-m','venv',$venv)
        if (!(Probe-Python $python)) { throw 'Creating the private .venv failed.' }
    }
    Write-Host ('Private environment: ' + $python)
    Write-Stage '2/5 Base packages' 'Installing only into this project .venv; no system packages are changed.'
    Install-Requirements 'requirements.txt'
    Invoke-Checked $python @('-m','pip','check')
    if ($Mode -eq 'extract') {
        Write-Stage '3/5 GPU packages' 'First setup downloads large CUDA PyTorch wheels; existing compatible downloads are cached.'
        Write-Host ('PyTorch source: ' + $TorchIndex)
        Invoke-Checked $python @('-m','pip','--isolated','install','--disable-pip-version-check','--only-binary=:all:','--timeout','60','--retries','2','--index-url',$TorchIndex,'--extra-index-url',$PackageIndex,'-r',(Join-Path $root 'requirements-torch.txt'))
        Install-Requirements 'requirements-gpu.txt'
        Invoke-Checked $python @('-m','pip','check')
        @'
import json, torch
if not torch.cuda.is_available():
    raise RuntimeError('NVIDIA CUDA unavailable. Update the NVIDIA driver and check that an NVIDIA GPU is assigned; no CPU fallback is used for extraction.')
x=torch.randn(64,64,device='cuda'); y=(x@x.T).square().mean(); torch.cuda.synchronize()
if not torch.isfinite(y): raise RuntimeError('CUDA arithmetic check failed')
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'VRAM_GiB':torch.cuda.get_device_properties(0).total_memory/2**30}))
'@ | Set-Content -LiteralPath (Join-Path $bootstrap 'check_gpu.py') -Encoding UTF8
        Invoke-Checked $python @((Join-Path $bootstrap 'check_gpu.py'))
    } else {
        Write-Stage '3/5 GPU packages' 'Skipped: cached-feature regression does not require CUDA or GPU packages.'
    }
    Write-Stage '4/5 Environment record' 'Recording installed versions and starting the selected experiment mode.'
    & $python -m pip freeze 2>&1 | Out-File -LiteralPath (Join-Path $root ('logs\environment_' + $stamp + '.txt')) -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'Could not record package versions.' }
    @{python=$python; mode=$Mode; method=$Method; package_index=$PackageIndex; torch_index=$TorchIndex; runtime_source=$runtimeSource; runtime_sha256=$runtimeSha256; log=$log; status='running'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'BOOTSTRAP_STATUS.json') -Encoding UTF8
    if (!(Test-Path -LiteralPath (Join-Path $root 'run_experiments.py'))) { throw 'run_experiments.py is missing; extract the complete package.' }
        $runMode = $Mode
    if ($Mode -eq 'compare') { $runMode = 'all' }
    $runArgs = @('-u',(Join-Path $root 'run_experiments.py'),'--mode',$runMode)
    if ($runMode -eq 'all') { $runArgs += @('--method',$Method) }
    if ($Mode -eq 'compare') { $runArgs += @('--features','generated_features') }
    Write-Stage '5/5 Run' ('mode=' + $Mode + '; method=' + $Method)
    Invoke-Checked $python $runArgs
    $script:exitCode = 0
    @{status='complete';mode=$Mode;method=$Method;python=$python;log=$log} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'BOOTSTRAP_STATUS.json') -Encoding UTF8
    Write-Host 'Selected mode completed. Read the experiment report and result files.'
} catch {
    $script:exitCode = 1
    $message = $_.Exception.Message
    Write-Host ('FAILED at ' + $script:stage + ': ' + $message) -ForegroundColor Red
    try {
        if ($null -ne $lock) {
            @{status='failed';stage=$script:stage;mode=$Mode;method=$Method;message=$message;log=$log} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'BOOTSTRAP_STATUS.json') -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $root 'BOOTSTRAP_ERROR.txt') -Value ($script:stage + ': ' + $message) -Encoding UTF8
        } else {
            Set-Content -LiteralPath ($log + '.rejected.txt') -Value $message -Encoding UTF8
        }
    } catch { Write-Host 'Could not write the error file; check folder write permissions.' }
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
    if ($null -ne $lock) { $lock.Dispose() }
}
exit $script:exitCode
