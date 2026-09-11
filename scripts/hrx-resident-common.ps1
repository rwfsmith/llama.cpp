# Shared implementation; dot-sourcing this file never starts or stops a process.
Set-StrictMode -Version Latest

$script:HrxComputeError = 'process_ubatch: failed to compute graph|graph_compute: .*failed|hipErrorLaunchFailure|unsupported HRX node|HRX .*invalid expert ID'

function Get-HrxHash([string]$Text) {
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($Text)))
}

function Get-HrxApiKey([string]$StateDir) {
    $path = Join-Path $StateDir 'api-key.txt'
    if (-not (Test-Path -LiteralPath $path)) {
        $null = New-Item -ItemType Directory -Path $StateDir -Force
        $key = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)).
            TrimEnd('=').Replace('+', '-').Replace('/', '_')
        $file = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $bytes = [Text.Encoding]::UTF8.GetBytes($key)
            $file.Write($bytes, 0, $bytes.Length)
            $file.Flush($true)
        } finally { $file.Dispose() }
    }
    $contents = [IO.File]::ReadAllText($path)
    if ($contents -cnotmatch '\A[A-Za-z0-9_-]{32,256}(?:\r?\n)?\z') {
        throw "Invalid API key file: $path. Expected one 32-256 character base64url token; refusing unauthenticated startup."
    }
    [pscustomobject]@{ Path = [IO.Path]::GetFullPath($path); Key = $contents.TrimEnd("`r", "`n"); Hash = Get-HrxHash $contents }
}

function Set-HrxAuthorization([Net.Http.HttpClient]$Client, [string]$Key) {
    if ($Key -cnotmatch '\A[A-Za-z0-9_-]{32,256}\z') { throw 'Missing or invalid resident API key.' }
    $Client.DefaultRequestHeaders.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $Key)
}

function Get-HrxModelMetadata([string]$Path) {
    $first = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($first.PSIsContainer) { throw "Not a model file: $Path" }
    $paths = @($first.FullName)
    if ($first.Name -match '^(.*)-(\d{5})-of-(\d{5})\.gguf$') {
        $prefix = $Matches[1]; $count = [int]$Matches[3]
        $paths = @(1..$count | ForEach-Object {
            Join-Path $first.DirectoryName ('{0}-{1:D5}-of-{2:D5}.gguf' -f $prefix, $_, $count)
        })
    }
    foreach ($p in $paths) {
        $f = Get-Item -LiteralPath $p -ErrorAction Stop
        [ordered]@{ Path = $f.FullName.ToLowerInvariant(); Length = $f.Length; Modified = $f.LastWriteTimeUtc.Ticks }
    }
}

function Get-HrxFingerprint([string]$Binary, [string[]]$DllDirectories, [string[]]$Models,
                            [string[]]$Arguments, [System.Collections.IDictionary]$Environment,
                            [string]$AuthenticationHash = '') {
    $files = @((Get-Item -LiteralPath $Binary -ErrorAction Stop).FullName)
    foreach ($directory in $DllDirectories) {
        if (-not (Test-Path -LiteralPath $directory -PathType Container)) { throw "Missing DLL directory: $directory" }
        $files += @(Get-ChildItem -LiteralPath $directory -Filter '*.dll' -File | ForEach-Object FullName)
    }
    $binaries = @(foreach ($file in ($files | Sort-Object -Unique)) {
        $f = Get-Item -LiteralPath $file
        [ordered]@{
            Path = $f.FullName.ToLowerInvariant()
            Length = $f.Length
            Modified = $f.LastWriteTimeUtc.Ticks
            SHA256 = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
        }
    })
    $environmentRows = @(foreach ($key in ($Environment.Keys | Sort-Object)) {
        [ordered]@{ Name = $key.ToString().ToUpperInvariant(); Value = [string]$Environment[$key] }
    })
    $document = [ordered]@{
        Version = 1
        Binaries = $binaries
        Models = @(foreach ($model in $Models) { Get-HrxModelMetadata $model })
        Arguments = @($Arguments)
        Environment = $environmentRows
        AuthenticationHash = $AuthenticationHash
    }
    Get-HrxHash (ConvertTo-Json -InputObject $document -Depth 12 -Compress)
}

function ConvertTo-HrxNativeArgument([string]$Value) {
    if ($Value.Contains([char]0)) { throw 'Arguments cannot contain NUL characters.' }
    '"' + (($Value -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
}

function Test-HrxIdentity($State, $Process) {
    if ($null -eq $State -or $null -eq $Process) { return $false }
    return (-not $Process.HasExited -and
        [int]$State.ProcessId -eq $Process.Id -and
        [long]$State.StartTimeUtcTicks -eq $Process.StartTime.ToUniversalTime().Ticks -and
        [string]::Equals([IO.Path]::GetFullPath($State.Executable),
            [IO.Path]::GetFullPath($Process.MainModule.FileName), [StringComparison]::OrdinalIgnoreCase))
}

function Get-HrxOwnedProcess($State) {
    if ($null -eq $State) { return $null }
    try { $process = [Diagnostics.Process]::GetProcessById([int]$State.ProcessId) }
    catch [ArgumentException] { return $null }
    try {
        # Pin the process handle before checking identity, avoiding a PID-reuse race at Kill().
        $null = $process.Handle
        if (Test-HrxIdentity $State $process) { return $process }
    } catch { $process.Dispose(); throw "Cannot verify resident process identity: $_" }
    $process.Dispose()
    return $null
}

function Stop-HrxOwnedProcess($State) {
    $process = Get-HrxOwnedProcess $State
    if ($null -eq $process) { return $false }
    try {
        $process.Kill()
        if (-not $process.WaitForExit(10000)) { throw "Owned PID $($process.Id) did not exit. Do not start another server." }
        return $true
    } finally { $process.Dispose() }
}

function Assert-HrxPortFree([int]$Port) {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    $listener.Server.ExclusiveAddressUse = $true
    try { $listener.Start() }
    catch { throw "Loopback port $Port is occupied/unavailable; refusing to attach to or replace an unowned server." }
    finally { $listener.Stop() }
}

function Assert-HrxEndpointOwner($State) {
    $process = Get-HrxOwnedProcess $State
    if ($null -eq $process) { throw 'Resident server exited or its identity no longer matches.' }
    $process.Dispose()
    $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
        Where-Object { $_.LocalPort -eq $State.Port -and $_.LocalAddress -in @('127.0.0.1', '0.0.0.0', '::') })
    if ($listeners.Count -ne 1 -or $listeners[0].OwningProcess -ne $State.ProcessId) {
        throw "Port $($State.Port) is not exclusively owned by the recorded server."
    }
}

function Read-HrxLog([string]$Path, [long]$Offset = 0) {
    if (-not (Test-Path -LiteralPath $Path)) { return '' }
    $file = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $null = $file.Seek($Offset, [IO.SeekOrigin]::Begin)
        $reader = [IO.StreamReader]::new($file, [Text.Encoding]::UTF8)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $file.Dispose() }
}

function Assert-HrxNoComputeError($State) {
    if ((Read-HrxLog $State.Stderr) -match $script:HrxComputeError) {
        throw "Backend compute error (including recovered failures); see $($State.Stderr)"
    }
}

function Invoke-HrxHttp($Client, $State, [string]$Path, [int]$TimeoutSeconds, [string]$Body = '') {
    $request = [Net.Http.HttpRequestMessage]::new(
        $(if ($Body) { [Net.Http.HttpMethod]::Post } else { [Net.Http.HttpMethod]::Get }),
        "http://127.0.0.1:$($State.Port)$Path")
    if ($Body) { $request.Content = [Net.Http.StringContent]::new($Body, [Text.Encoding]::UTF8, 'application/json') }
    $cancel = [Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($TimeoutSeconds))
    try {
        $task = $Client.SendAsync($request, $cancel.Token)
        while (-not $task.IsCompleted) {
            $process = Get-HrxOwnedProcess $State
            if ($null -eq $process) { throw 'Resident server exited during HTTP request.' }
            $process.Dispose()
            Assert-HrxNoComputeError $State
            Start-Sleep -Milliseconds 200
        }
        $response = $task.GetAwaiter().GetResult()
        try {
            [pscustomobject]@{
                StatusCode = [int]$response.StatusCode
                Body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            }
        } finally { $response.Dispose() }
    } finally { $cancel.Cancel(); $cancel.Dispose(); $request.Dispose() }
}

function Test-HrxReadinessRetry([Exception]$Exception) {
    while ($Exception -is [Management.Automation.RuntimeException] -and $Exception.InnerException) {
        $Exception = $Exception.InnerException
    }
    if ($Exception -is [OperationCanceledException] -or $Exception -is [TimeoutException]) { return $true }
    if ($Exception -is [Net.Http.HttpRequestException]) { $Exception = $Exception.InnerException }
    return ($Exception -is [Net.Sockets.SocketException] -and
        $Exception.SocketErrorCode -in @([Net.Sockets.SocketError]::ConnectionRefused,
            [Net.Sockets.SocketError]::ConnectionReset, [Net.Sockets.SocketError]::ConnectionAborted,
            [Net.Sockets.SocketError]::TimedOut))
}

function Wait-HrxReady($Client, $State, [int]$TimeoutSeconds) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $process = Get-HrxOwnedProcess $State
        if ($null -eq $process) { throw "Server exited before readiness; see $($State.Stderr)" }
        $process.Dispose()
        Assert-HrxNoComputeError $State
        try {
            $result = Invoke-HrxHttp $Client $State '/health' 2
        } catch {
            # Connection refused/timeout is normal while loading; compute failures and exits are not.
            if (-not (Test-HrxReadinessRetry $_.Exception)) { throw }
            Assert-HrxNoComputeError $State
            $process = Get-HrxOwnedProcess $State
            if ($null -eq $process) { throw "Server exited before readiness; see $($State.Stderr)" }
            $process.Dispose()
            Start-Sleep -Milliseconds 250
            continue
        }
        if ($result.StatusCode -eq 200) {
            $health = $result.Body | ConvertFrom-Json
            if ($health.status -ne 'ok') { throw "Unexpected health response: $($result.Body)" }
            Assert-HrxEndpointOwner $State
            return
        }
        if ($result.StatusCode -ne 503) { throw "Health HTTP $($result.StatusCode): $($result.Body)" }
        Start-Sleep -Milliseconds 250
    }
    throw "Server readiness timed out after $TimeoutSeconds seconds; see $($State.Stderr)"
}

function New-HrxStartupConfiguration($Options) {
    $environment = @{}
    foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $environment[$entry.Key.ToString().ToUpperInvariant()] = [string]$entry.Value
    }
    # A build-time LLVM override can break COMGR's internal blit compilation and
    # hang HIP stream creation. Let the runtime select its compiler by default.
    $environment.Remove('LLVM_PATH')
    $environment.HRX_GPU_DRIVER = 'hip'
    $environment.HIP_PATH = $Options.HipPath
    $environment.HIP_DEVICE_LIB_PATH = Join-Path $Options.HipPath 'lib\llvm\amdgcn\bitcode'
    $environment.PATH = "$(Join-Path $Options.HipPath 'bin');$($Options.PinnedBin);$($environment.PATH)"
    foreach ($flag in @('IQ_EXPERTS', 'Q8_NARROW', 'Q8_EMBEDDING', 'SET_ROWS', 'F32_GET_ROWS',
                         'REPEAT', 'QSA_ATTN', 'QSA_ROPE', 'DENSE_F32_ACCUM', 'GDN_NORM_GATE',
                         'PLE_CONV_FUSION', 'F32_ROUTER', 'QWEN4EXP_ROUTER',
                         'RECURRENT_CONCAT', 'GDN_CONV_PREFILL', 'GDN_PREFILL', 'GDN_NORM_PREFILL',
                         'QSA_MASK', 'QSA_PROJECTIONS', 'QSA_F16_GATHER', 'TRUSTED_INDEX_VIEWS')) {
        $environment["HRX_ENABLE_$flag"] = '1'
    }
    $environment.HRX_ENABLE_Q8_GEMV = '0'
    $environment.HRX_MUL_CLAIM_MASK = '0x7F'
    if ($Options.MicroBatch -gt 1 -or $Options.Mtp) {
        foreach ($flag in @('MOE_SMALL_BATCH', 'HC_SMALL_BATCH', 'SMALL_BATCH_GLUE', 'SWIGLU')) {
            $environment["HRX_ENABLE_$flag"] = '1'
        }
    }
    if ($Options.Mtp) {
        foreach ($flag in @('Q4_EMBEDDING', 'MTP_HC_PROJECTION')) {
            $environment["HRX_ENABLE_$flag"] = '1'
        }
    }
    foreach ($key in $Options.Env.Keys) {
        if ([string]::IsNullOrEmpty($key) -or $key.Contains('=') -or $key.Contains([char]0)) { throw "Invalid environment key: $key" }
        if ($key -in @('LLAMA_API_KEY', 'LLAMA_ARG_API_KEY_FILE')) {
            throw 'Set authentication through StateDir\api-key.txt, not Env.'
        }
        $environment[$key.ToUpperInvariant()] = [string]$Options.Env[$key]
    }
    # Do not accidentally authorize extra inherited keys alongside the private key file.
    # Removing (not blanking) matters: llama.cpp treats a present-but-empty
    # LLAMA_ARG_API_KEY_FILE as "use this (empty) path", which fails startup outright
    # instead of falling back to the --api-key-file argument below.
    $environment.Remove('LLAMA_API_KEY')
    $environment.Remove('LLAMA_ARG_API_KEY_FILE')
    $arguments = @('-m', $Options.Model, '-ngl', '99', '-fit', 'off', '--no-warmup', '-fa', 'on',
        '-b', '512', '-ub', "$($Options.MicroBatch)", '-c', "$($Options.Context)",
        '-np', '1', '--host', '127.0.0.1', '--port', "$($Options.Port)", '--jinja',
        '-lv', "$($Options.LogVerbosity)")
    if ($Options.ContainsKey('ApiKeyFile')) {
        $arguments += @('--api-key-file', $Options.ApiKeyFile)
    }
    if (-not $Options.OffloadKV) { $arguments += '-nkvo' }
    if ($Options.Threads -gt 0) { $arguments += @('-t', "$($Options.Threads)", '-tb', "$($Options.Threads)") }
    $models = @($Options.Model)
    if ($Options.Mtp) {
        $models += $Options.DraftModel
        $arguments += @('-md', $Options.DraftModel, '--spec-type', 'draft-mtp', '--spec-draft-n-max', "$($Options.DraftTokens)")
        if ($Options.DraftThreads -gt 0) {
            $arguments += @('--spec-draft-threads', "$($Options.DraftThreads)", '--spec-draft-threads-batch', "$($Options.DraftThreads)")
        }
    }
    $directories = @((Split-Path $Options.Binary), (Join-Path $environment.HIP_PATH 'bin'), $Options.PinnedBin)
    if ($environment.ContainsKey('GGML_BACKEND_PATH') -and $environment.GGML_BACKEND_PATH) {
        $backend = Get-Item -LiteralPath $environment.GGML_BACKEND_PATH -ErrorAction Stop
        if ($backend.PSIsContainer -or $backend.Extension -ne '.dll') {
            throw 'GGML_BACKEND_PATH must name a backend DLL, not a directory.'
        }
        $directories += $backend.DirectoryName
    }
    [pscustomobject]@{ Environment = $environment; Arguments = $arguments; Models = $models; DllDirectories = $directories }
}
