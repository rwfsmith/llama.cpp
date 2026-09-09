<#
.SYNOPSIS
Run repeatable chat tests against one resident, harness-owned Windows llama-server.
.DESCRIPTION
Run starts the server if needed, otherwise reuses its loaded weights. A changed
EXE/DLL content or metadata, model shard path/size/mtime, startup argument, or
effective child environment restarts it. Model contents are NEVER hashed: after
an in-place edit preserving model size/mtime, explicitly Stop before Run.
Prompt, Seed, Temperature, N, ExpectedOutput and CachePrompt are request-only.
Changing Mtp, context, batching, KV offload, threads, model or backend needs restart.
Only one worker (-np 1) and loopback HTTP are used; no fallback to another backend.
The Windows child survives this script. Run Stop when finished to release VRAM.
Run/Status/Stop are serialized by a named Windows mutex. Status never starts a
process; Stop checks PID, process start time and executable before terminating.
Timeouts and backend failures stop only the owned child, including recovered
compute errors. A text-expectation failure keeps an otherwise healthy server.
Logs/state default to LocalAppData, never the repository. StateDir must be private
to your account; do not manually alter state.json or share this endpoint.
A random API key persists in StateDir\api-key.txt. Requests use Bearer auth;
only its hash enters the fingerprint. Replacing/deleting the key restarts the
server on the next Run, including upgrading an existing unauthenticated server.
Status/Stop do not read or create keys; the server's health route remains public.
Never commit or share api-key.txt. Invalid key files fail closed, not regenerate.
All DLLs in the executable, HIP bin, pinned bin and GGML_BACKEND_PATH DLL's directories
are hashed. If using extra runtime locations, supply them through DllDirectories.
Environment is inherited, then HIP/known-good HRX defaults, then Env overrides.
Its fingerprint is stored, not its potentially sensitive values. SWIGLU is not
enabled by default. MTP defaults off; enable explicitly for the full sidecar.
MicroBatch > 1 enables the MOE, HC and glue small-batch flags even without MTP.
Threads defaults to 1 and KV offload is enabled; use -OffloadKV:$false to disable.
.EXAMPLE
.\scripts\run-hrx-resident.ps1 -Prompt 'The capital of France is' -N 16 -ExpectedOutput Paris
.EXAMPLE
.\scripts\run-hrx-resident.ps1 -Mtp -DraftTokens 1 -DraftThreads 4 -MicroBatch 8 -N 32
.EXAMPLE
.\scripts\run-hrx-resident.ps1 -Action Status
.\scripts\run-hrx-resident.ps1 -Action Stop
#>

#Requires -Version 7.2
[CmdletBinding()]
param(
    [ValidateSet('Run', 'Status', 'Stop')][string]$Action = 'Run',
    [string]$StateDir = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'HRX\resident'),
    [string]$Binary = 'C:\b\llama-qwen4exp-build\bin\llama-server.exe',
    [string]$Model = 'C:\models\unsloth\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf',
    [string]$DraftModel = 'C:\models\unsloth\Qwen3.8-Flash-Next-GGUF\mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf',
    [string]$HipPath = 'C:\TheRock\build',
    [string]$PinnedBin = 'C:\b\hrx-pinned-install\bin',
    [string[]]$DllDirectories = @(),
    [hashtable]$Env = @{},
    [ValidateRange(1024, 65535)][int]$Port = 8087,
    [switch]$Mtp,
    [ValidateRange(1, 8)][int]$DraftTokens = 1,
    [ValidateRange(0, 256)][int]$DraftThreads = 0,
    [ValidateRange(1, 512)][int]$MicroBatch = 8,
    [ValidateRange(512, 32768)][int]$Context = 512,
    [ValidateRange(0, 256)][int]$Threads = 1,
    [switch]$OffloadKV = $true,
    [ValidateNotNullOrEmpty()][string]$Prompt = 'The capital of France is',
    [string]$PromptFile = '',
    [int]$Seed = 1234,
    [ValidateRange(0.0, 10.0)][double]$Temperature = 1.0,
    [ValidateRange(1, 4096)][int]$N = 16,
    [string]$ExpectedOutput = '',
    [switch]$CachePrompt,
    [string]$Tag = 'test',
    [ValidateRange(1, 1800)][int]$StartupSeconds = 600,
    [ValidateRange(1, 1800)][int]$MaxSeconds = 600,
    [ValidateRange(1, 3600)][int]$LockSeconds = 30
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hrx-resident-common.ps1')
if (-not $IsWindows) { throw 'This utility requires Windows PowerShell 7.2+ (pwsh).' }
$StateDir = [IO.Path]::GetFullPath($StateDir)
for ($directory = [IO.DirectoryInfo]::new($StateDir); $null -ne $directory; $directory = $directory.Parent) {
    if (Test-Path -LiteralPath (Join-Path $directory.FullName '.git')) { throw 'StateDir must be outside a git working tree.' }
}
$mutex = [Threading.Mutex]::new($false, 'Global\HRX-Resident-Harness-v1')
$locked = $false; $client = $null; $handler = $null; $state = $null
$artifactDir = $null; $stderrOffset = 0L; $captureState = $null
$controlActive = $false; $inferenceHealthy = $false
$runResult = $null
try {
    try { $locked = $mutex.WaitOne([TimeSpan]::FromSeconds($LockSeconds)) }
    catch [Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) { throw 'Another resident operation holds the lock; no server was changed.' }
    $stateFile = Join-Path $StateDir 'state.json'
    if (Test-Path -LiteralPath $stateFile) {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        if ($state.Version -ne 1) { throw 'Unrecognized resident state; refusing process control.' }
    }
    $owned = Get-HrxOwnedProcess $state
    $running = $null -ne $owned
    if ($running) { $owned.Dispose() }
    if ($Action -eq 'Status') {
        [pscustomobject]@{
            Action = 'Status'; Running = $running
            PID = $(if ($state) { $state.ProcessId } else { $null })
            IdentityVerified = $running
            State = $(if ($running) { 'Resident' } elseif ($state) { 'StaleOrForeign' } else { 'Stopped' })
            Port = $(if ($state) { $state.Port } else { $Port })
            StateFile = $stateFile
            Stderr = $(if ($state) { $state.Stderr } else { $null })
        }
        return
    }
    if ($Action -eq 'Stop') {
        $stopped = Stop-HrxOwnedProcess $state
        if ($state -and -not $stopped) { Write-Warning 'Recorded process is absent or foreign; no process was killed.' }
        if ($state) { Remove-Item -LiteralPath $stateFile }
        [pscustomobject]@{ Action = 'Stop'; Stopped = $stopped; StateFile = $stateFile }
        return
    }
    if ($PromptFile) {
        if ($PSBoundParameters.ContainsKey('Prompt')) { throw 'Use Prompt or PromptFile, not both.' }
        $Prompt = Get-Content -LiteralPath $PromptFile -Raw
        if ([string]::IsNullOrEmpty($Prompt)) { throw 'Prompt file is empty.' }
    }
    $Binary = (Get-Item -LiteralPath $Binary).FullName
    $Model = (Get-Item -LiteralPath $Model).FullName
    if ($Mtp) { $DraftModel = (Get-Item -LiteralPath $DraftModel).FullName }
    $authentication = Get-HrxApiKey $StateDir
    $options = @{
        Binary = $Binary; Model = $Model; DraftModel = $DraftModel; Mtp = $Mtp.IsPresent
        DraftTokens = $DraftTokens; DraftThreads = $DraftThreads; MicroBatch = $MicroBatch
        Context = $Context; Threads = $Threads; OffloadKV = $OffloadKV.IsPresent
        Port = $Port; HipPath = $HipPath; PinnedBin = $PinnedBin; Env = $Env; ApiKeyFile = $authentication.Path
    }
    $configuration = New-HrxStartupConfiguration $options
    $fingerprint = Get-HrxFingerprint -Binary $Binary -DllDirectories ($configuration.DllDirectories + $DllDirectories) `
        -Models $configuration.Models -Arguments $configuration.Arguments -Environment $configuration.Environment `
        -AuthenticationHash $authentication.Hash
    $reused = $running -and $state.Fingerprint -eq $fingerprint
    $restarted = $running -and -not $reused
    if ($restarted) {
        $null = Stop-HrxOwnedProcess $state
        Remove-Item -LiteralPath $stateFile
        $state = $null
    }
    if (-not $reused) { Assert-HrxPortFree $Port }
    $stamp = '{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
    $artifactDir = Join-Path $StateDir "tests\$($Tag -replace '[^a-zA-Z0-9_-]', '_')-$stamp"
    $null = New-Item -ItemType Directory -Path $artifactDir -Force
    $controlActive = $true
    if (-not $reused) {
        $logDir = Join-Path $StateDir "servers\$stamp"
        $null = New-Item -ItemType Directory -Path $logDir -Force
        $out = Join-Path $logDir 'stdout.log'; $err = Join-Path $logDir 'stderr.log'
        $captureState = @{ Stderr = $err }
        $savedEnvironment = @{}
        foreach ($key in $configuration.Environment.Keys) {
            $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        }
        $process = $null
        try {
            foreach ($key in $configuration.Environment.Keys) {
                [Environment]::SetEnvironmentVariable($key, $configuration.Environment[$key], 'Process')
            }
            # On Windows Start-Process creates an independent child. File redirection uses
            # inherited file handles, not a pipeline requiring this script to stay alive.
            $process = Start-Process -FilePath $Binary -WorkingDirectory (Split-Path $Binary) `
                -ArgumentList (($configuration.Arguments | ForEach-Object { ConvertTo-HrxNativeArgument $_ }) -join ' ') `
                -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
            $null = $process.Handle
            $state = [pscustomobject]@{
                Version = 1; ProcessId = $process.Id; StartTimeUtcTicks = $process.StartTime.ToUniversalTime().Ticks
                Executable = $Binary; Fingerprint = $fingerprint; Port = $Port
                Stdout = $out; Stderr = $err
            }
            $state | ConvertTo-Json | Set-Content -LiteralPath "$stateFile.new" -Encoding utf8
            Move-Item -LiteralPath "$stateFile.new" -Destination $stateFile -Force
        } catch {
            # This exact Process object was just launched here, even if state persistence failed.
            if ($null -ne $process -and -not $process.HasExited) {
                $process.Kill()
                if (-not $process.WaitForExit(10000)) { throw 'New server failed to stop after startup failure.' }
            }
            throw
        } finally {
            foreach ($key in $configuration.Environment.Keys) {
                [Environment]::SetEnvironmentVariable($key, $savedEnvironment[$key], 'Process')
            }
            if ($process) { $process.Dispose() }
        }
    }
    $captureState = $state
    if ($reused) { $stderrOffset = (Get-Item -LiteralPath $state.Stderr).Length }
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [Threading.Timeout]::InfiniteTimeSpan
    Set-HrxAuthorization $client $authentication.Key
    $responsePath = Join-Path $artifactDir 'response.json'
    $stderrPath = Join-Path $artifactDir 'server.stderr.log'
    $request = @{
        messages = @(@{ role = 'user'; content = $Prompt })
        stream = $false; seed = $Seed; temperature = $Temperature; max_tokens = $N
        cache_prompt = $CachePrompt.IsPresent; chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 8
    $request | Set-Content -LiteralPath (Join-Path $artifactDir 'request.json') -Encoding utf8
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        Wait-HrxReady $client $state $StartupSeconds
        Assert-HrxNoComputeError $state
        $watch.Restart()
        $response = Invoke-HrxHttp $client $state '/v1/chat/completions' $MaxSeconds $request
        $watch.Stop()
        $response.Body | Set-Content -LiteralPath $responsePath -Encoding utf8
        Assert-HrxEndpointOwner $state
        Assert-HrxNoComputeError $state
        if ($response.StatusCode -ne 200) { throw "Inference HTTP $($response.StatusCode); see $responsePath" }
        $result = $response.Body | ConvertFrom-Json
        if ($result.PSObject.Properties['error']) { throw "Inference error: $($result.error | ConvertTo-Json -Compress)" }
        if (-not $result.PSObject.Properties['choices'] -or $result.choices.Count -ne 1 -or
            $result.choices[0].message.content -isnot [string]) {
            throw "No valid text completion; see $responsePath"
        }
    } catch {
        $null = Stop-HrxOwnedProcess $state
        if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }
        throw
    }
    $inferenceHealthy = $true
    $text = $result.choices[0].message.content
    if ($ExpectedOutput -and $text.IndexOf($ExpectedOutput, [StringComparison]::Ordinal) -lt 0) {
        throw "Expected output was not generated; see $responsePath (healthy server remains resident)."
    }
    $runResult = [pscustomobject]@{
        Action = 'Run'; PID = $state.ProcessId; Reused = $reused; Restarted = $restarted
        Text = $text
        Timings = $(if ($result.PSObject.Properties['timings']) { $result.timings } else { $null })
        ElapsedSeconds = $watch.Elapsed.TotalSeconds
        ResponsePath = $responsePath; StderrSegmentPath = $stderrPath
        ServerStderr = $state.Stderr; ServerStdout = $state.Stdout
        ArtifactDirectory = $artifactDir; StateFile = $stateFile; Fingerprint = $fingerprint
    }
} catch {
    if ($controlActive -and -not $inferenceHealthy -and $state) {
        $null = Stop-HrxOwnedProcess $state
        if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }
    }
    if ($artifactDir) { $_.ToString() | Set-Content -LiteralPath (Join-Path $artifactDir 'error.txt') -Encoding utf8 }
    throw
} finally {
    try {
        if ($captureState -and $artifactDir) {
            Read-HrxLog $captureState.Stderr $stderrOffset |
                Add-Content -LiteralPath (Join-Path $artifactDir 'server.stderr.log') -Encoding utf8
        }
    } finally {
        if ($client) { $client.Dispose() }
        if ($handler) { $handler.Dispose() }
        if ($locked) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}
if ($null -ne $runResult) { $runResult }
