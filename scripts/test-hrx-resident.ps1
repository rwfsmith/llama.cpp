<#
.SYNOPSIS
CPU-only resident harness checks using PowerShell/.NET; no llama process is run.
.EXAMPLE
pwsh -NoProfile -File .\scripts\test-hrx-resident.ps1
#>

#Requires -Version 7.2
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hrx-resident-common.ps1')
$passed = 0
function Assert-Test([bool]$Condition, [string]$Name) {
    if (-not $Condition) { throw "FAIL: $Name" }
    $script:passed++
    Write-Host "PASS: $Name"
}
function Assert-Throws([scriptblock]$Body, [string]$Name) {
    $threw = $false
    try { & $Body } catch { $threw = $true }
    Assert-Test $threw $Name
}
$fixture = Join-Path $PSScriptRoot ('.resident-tests-' + [guid]::NewGuid().ToString('N'))
try {
    $null = New-Item -ItemType Directory -Path $fixture
    $dllDir = Join-Path $fixture 'bin'
    $null = New-Item -ItemType Directory -Path $dllDir
    $binary = Join-Path $dllDir 'llama-server.exe'
    $dll = Join-Path $dllDir 'ggml-hrx.dll'
    $model = Join-Path $fixture 'model-00001-of-00002.gguf'
    $second = Join-Path $fixture 'model-00002-of-00002.gguf'
    $draft = Join-Path $fixture 'mtp.gguf'
    foreach ($file in @($binary, $dll, $model, $second, $draft)) { [IO.File]::WriteAllText($file, 'test-data') }
    $base = @{
        Binary = $binary; Model = $model; DraftModel = $draft; Mtp = $false
        DraftTokens = 1; DraftThreads = 0; MicroBatch = 8; Context = 512
        Threads = 1; OffloadKV = $true; Port = 8087; HipPath = $fixture; PinnedBin = $dllDir; Env = @{}
    }
    function Get-TestFingerprint($Options) {
        $configuration = New-HrxStartupConfiguration $Options
        Get-HrxFingerprint -Binary $Options.Binary -DllDirectories $configuration.DllDirectories `
            -Models $configuration.Models -Arguments $configuration.Arguments -Environment $configuration.Environment
    }
    $baseline = Get-TestFingerprint $base
    Assert-Test ($baseline -eq (Get-TestFingerprint $base)) 'unchanged configuration reuses weights'
    $configuration = New-HrxStartupConfiguration $base
    Assert-Test (($configuration.Arguments -join ' ') -match '-np 1') 'one server worker'
    Assert-Test (($configuration.Arguments -join ' ') -match '--host 127.0.0.1') 'loopback only'
    Assert-Test (($configuration.Arguments -join ' ') -match '-t 1 -tb 1') 'single CPU thread profile'
    Assert-Test ($configuration.Arguments -notcontains '-nkvo') 'KV offload profile'
    Assert-Test ($configuration.Environment.HRX_ENABLE_Q8_GEMV -eq '0') 'Q8 GEMV stays disabled'
    Assert-Test ($configuration.Environment.HRX_MUL_CLAIM_MASK -eq '0x7F') 'known-good claim mask'
    Assert-Test ($configuration.Models.Count -eq 1) 'MTP is opt-in'
    # A present-but-empty LLAMA_ARG_API_KEY_FILE makes llama.cpp try to open '' and fail
    # startup outright; these must be absent from the child environment, not blanked.
    Assert-Test (-not $configuration.Environment.ContainsKey('LLAMA_API_KEY')) 'LLAMA_API_KEY is removed, not blanked'
    Assert-Test (-not $configuration.Environment.ContainsKey('LLAMA_ARG_API_KEY_FILE')) 'LLAMA_ARG_API_KEY_FILE is removed, not blanked'
    $previousApiKey = $env:LLAMA_API_KEY; $previousApiKeyFile = $env:LLAMA_ARG_API_KEY_FILE
    try {
        $env:LLAMA_API_KEY = 'leaked-inherited-key'
        $env:LLAMA_ARG_API_KEY_FILE = 'C:\leaked\inherited\path.txt'
        $leakedConfiguration = New-HrxStartupConfiguration $base
        Assert-Test (-not $leakedConfiguration.Environment.ContainsKey('LLAMA_API_KEY')) 'inherited LLAMA_API_KEY is stripped'
        Assert-Test (-not $leakedConfiguration.Environment.ContainsKey('LLAMA_ARG_API_KEY_FILE')) 'inherited LLAMA_ARG_API_KEY_FILE is stripped'
    } finally {
        $env:LLAMA_API_KEY = $previousApiKey; $env:LLAMA_ARG_API_KEY_FILE = $previousApiKeyFile
    }
    $withKey = $base.Clone(); $withKey.ApiKeyFile = Join-Path $fixture 'api-key.txt'
    $withKeyConfiguration = New-HrxStartupConfiguration $withKey
    Assert-Test (($withKeyConfiguration.Arguments -join ' ') -match
        [regex]::Escape("--api-key-file $($withKey.ApiKeyFile)")) 'api key file is passed via argument'
    Assert-Test (-not $withKeyConfiguration.Environment.ContainsKey('LLAMA_ARG_API_KEY_FILE')) 'api key file argument does not leak into environment'
    foreach ($flag in @('MOE_SMALL_BATCH', 'HC_SMALL_BATCH', 'SMALL_BATCH_GLUE')) {
        Assert-Test ($configuration.Environment["HRX_ENABLE_$flag"] -eq '1') "$flag enabled for non-MTP microbatch > 1"
    }
    $override = $base.Clone(); $override.Env = @{ HRX_ENABLE_SMALL_BATCH_GLUE = '0'; HRX_ENABLE_SWIGLU = '1' }
    $overrideConfiguration = New-HrxStartupConfiguration $override
    Assert-Test ($overrideConfiguration.Environment.HRX_ENABLE_SMALL_BATCH_GLUE -eq '0') 'explicit Env overrides small-batch defaults'
    Assert-Test ($overrideConfiguration.Environment.HRX_ENABLE_SWIGLU -eq '1') 'SWIGLU can be explicitly enabled'
    foreach ($field in @('Prompt', 'PromptFile', 'Seed', 'Temperature', 'N', 'ExpectedOutput', 'CachePrompt', 'Tag')) {
        $changed = $base.Clone(); $changed[$field] = 'different-request'
        Assert-Test ($baseline -eq (Get-TestFingerprint $changed)) "$field does not restart"
    }
    foreach ($entry in @{
        Context = 1024; Threads = 4; MicroBatch = 4; OffloadKV = $false
        Mtp = $true; Port = 8088; Env = @{ HRX_ENABLE_Q8_GEMV = '1' }
    }.GetEnumerator()) {
        $changed = $base.Clone(); $changed[$entry.Key] = $entry.Value
        Assert-Test ($baseline -ne (Get-TestFingerprint $changed)) "$($entry.Key) change restarts"
    }
    $mtp = $base.Clone(); $mtp.Mtp = $true
    $mtpFingerprint = Get-TestFingerprint $mtp
    $mtpConfiguration = New-HrxStartupConfiguration $mtp
    Assert-Test ($mtpConfiguration.Environment.HRX_ENABLE_MTP_HC_PROJECTION -eq '1') 'MTP projection flag'
    Assert-Test ($mtpConfiguration.Environment.HRX_ENABLE_MOE_SMALL_BATCH -eq '1') 'MTP small-batch flag'
    foreach ($field in @('DraftTokens', 'DraftThreads')) {
        $changed = $mtp.Clone(); $changed[$field] = 4
        Assert-Test ($mtpFingerprint -ne (Get-TestFingerprint $changed)) "$field restarts MTP"
        $changed = $base.Clone(); $changed[$field] = 4
        Assert-Test ($baseline -eq (Get-TestFingerprint $changed)) "$field is inert without MTP"
    }
    $a = $base.Clone(); $a.Env = @{ HRX_Z_TEST = 'z'; HRX_A_TEST = 'a' }
    $b = $base.Clone(); $b.Env = @{ hrx_a_test = 'a'; hrx_z_test = 'z' }
    Assert-Test ((Get-TestFingerprint $a) -eq (Get-TestFingerprint $b)) 'environment ordering and case are canonical'
    $external = $base.Clone(); $external.Env = @{ GGML_BACKEND_PATH = $dll }
    Assert-Test ((Get-TestFingerprint $external) -ne $baseline) 'explicit backend DLL path is fingerprinted'
    $external.Env.GGML_BACKEND_PATH = $dllDir
    Assert-Throws { Get-TestFingerprint $external } 'backend path directory is rejected as server expects a DLL'
    foreach ($file in @($binary, $dll)) {
        $originalTime = (Get-Item -LiteralPath $file).LastWriteTimeUtc
        [IO.File]::WriteAllText($file, 'new-bytes')
        [IO.File]::SetLastWriteTimeUtc($file, $originalTime)
        Assert-Test ($baseline -ne (Get-TestFingerprint $base)) 'binary/DLL content change with preserved metadata restarts'
        [IO.File]::WriteAllText($file, 'test-data')
        [IO.File]::SetLastWriteTimeUtc($file, $originalTime)
    }
    $originalTime = (Get-Item -LiteralPath $second).LastWriteTimeUtc
    [IO.File]::WriteAllText($second, 'new-bytes')
    [IO.File]::SetLastWriteTimeUtc($second, $originalTime)
    Assert-Test ($baseline -eq (Get-TestFingerprint $base)) 'models use metadata, never content hashing'
    [IO.File]::SetLastWriteTimeUtc($second, $originalTime.AddSeconds(1))
    Assert-Test ($baseline -ne (Get-TestFingerprint $base)) 'non-first model shard mtime restarts'
    [IO.File]::SetLastWriteTimeUtc($second, $originalTime)
    [IO.File]::AppendAllText($second, 'x')
    Assert-Test ($baseline -ne (Get-TestFingerprint $base)) 'non-first model shard size restarts'
    Remove-Item -LiteralPath $second
    Assert-Throws { Get-TestFingerprint $base } 'missing model shard is an error, not fallback'
    $metadata = @(Get-HrxModelMetadata $draft)
    Assert-Test ($metadata.Count -eq 1) 'unsplit draft metadata supported'
    Assert-Test ((ConvertTo-HrxNativeArgument 'C:\path with space\') -eq '"C:\path with space\\"') 'native trailing slash quoting'
    Assert-Throws { ConvertTo-HrxNativeArgument "bad$([char]0)arg" } 'NUL arguments rejected'

    $self = [Diagnostics.Process]::GetCurrentProcess()
    try {
        $identity = [pscustomobject]@{
            ProcessId = $self.Id
            StartTimeUtcTicks = $self.StartTime.ToUniversalTime().Ticks
            Executable = $self.MainModule.FileName
        }
        Assert-Test (Test-HrxIdentity $identity $self) 'PID/start time/executable match'
        $handle = Get-HrxOwnedProcess $identity
        Assert-Test ($null -ne $handle) 'matching process handle acquired without launching'
        $handle.Dispose()
        $identity.StartTimeUtcTicks++
        Assert-Test (-not (Stop-HrxOwnedProcess $identity)) 'wrong start time never kills current foreign process'
        $identity.StartTimeUtcTicks--
        $identity.Executable = Join-Path $fixture 'foreign.exe'
        Assert-Test (-not (Stop-HrxOwnedProcess $identity)) 'wrong executable never kills current foreign process'
        $identity.Executable = $self.MainModule.FileName
        $identity.ProcessId = -1
        Assert-Test (-not (Stop-HrxOwnedProcess $identity)) 'missing PID is harmless'
        Assert-Throws { Test-HrxIdentity @{} $self } 'malformed identity raises a verification error'
        $identity.ProcessId = $self.Id
        $unreadable = [pscustomobject]@{ HasExited = $false; Id = $self.Id }
        $unreadable | Add-Member -MemberType ScriptProperty -Name StartTime -Value {
            throw [UnauthorizedAccessException]::new('Access denied reading process start time')
        }
        Assert-Throws { Test-HrxIdentity $identity $unreadable } 'unreadable process identity is not silently treated as absent'
        Assert-Test (-not (Stop-HrxOwnedProcess $null)) 'empty state is harmless'
    } finally { $self.Dispose() }
    $stderr = Join-Path $fixture 'stderr.log'
    [IO.File]::WriteAllText($stderr, "old log`n")
    $offset = (Get-Item -LiteralPath $stderr).Length
    [IO.File]::AppendAllText($stderr, "new log`n")
    Assert-Test ((Read-HrxLog $stderr $offset) -eq "new log`n") 'per-request stderr segment'
    foreach ($errorLine in @('process_ubatch: failed to compute graph', 'graph_compute: command failed',
        'hipErrorLaunchFailure', 'unsupported HRX node', 'HRX gather invalid expert ID')) {
        [IO.File]::WriteAllText($stderr, "$errorLine`nrecovered successfully")
        Assert-Throws { Assert-HrxNoComputeError @{ Stderr = $stderr } } "recovered error rejected: $errorLine"
    }
    & {
        function Get-HrxOwnedProcess { [Diagnostics.Process]::GetCurrentProcess() }
        function Assert-HrxNoComputeError { }
        function Assert-HrxEndpointOwner { }
        function Start-Sleep { }
        function Invoke-HrxHttp {
            $script:healthCalls++
            if ($script:healthCalls -eq 1) { throw $script:healthFailure }
            [pscustomobject]@{ StatusCode = 200; Body = '{"status":"ok"}' }
        }
        $refused = [Net.Http.HttpRequestException]::new('connection refused',
            [Net.Sockets.SocketException]::new([int][Net.Sockets.SocketError]::ConnectionRefused))
        foreach ($failure in @($refused, [Threading.Tasks.TaskCanceledException]::new('health timeout'),
            [Management.Automation.MethodInvocationException]::new('wrapped HTTP failure', $refused))) {
            $script:healthCalls = 0; $script:healthFailure = $failure
            Wait-HrxReady $null @{ Stderr = 'unused' } 3
            Assert-Test ($script:healthCalls -eq 2) "readiness retries expected $($failure.GetType().Name)"
        }
        foreach ($failure in @([UnauthorizedAccessException]::new('identity denied'),
            [IO.IOException]::new('cannot read server log'), [InvalidOperationException]::new('programming error'),
            [Net.Http.HttpRequestException]::new('invalid HTTP response'),
            [Net.Sockets.SocketException]::new([int][Net.Sockets.SocketError]::AccessDenied))) {
            $script:healthCalls = 0; $script:healthFailure = $failure
            Assert-Throws { Wait-HrxReady $null @{ Stderr = 'unused' } 3 } "readiness propagates $($failure.GetType().Name)"
            Assert-Test ($script:healthCalls -eq 1) 'unexpected readiness failure is not retried'
        }
    }
    foreach ($file in @('run-hrx-resident.ps1', 'hrx-resident-common.ps1', 'test-hrx-resident.ps1')) {
        $tokens = $null; $errors = $null
        $null = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $file), [ref]$tokens, [ref]$errors)
        Assert-Test ($errors.Count -eq 0) "$file parses"
    }
    Assert-Test ((Get-Help (Join-Path $PSScriptRoot 'run-hrx-resident.ps1')).Synopsis -like 'Run repeatable*') 'comment help is discoverable'
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'run-hrx-resident.ps1'),
        [ref]$tokens, [ref]$errors)
    $threadParameter = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq 'Threads' }
    $kvParameter = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq 'OffloadKV' }
    Assert-Test ($threadParameter.DefaultValue.SafeGetValue() -eq 1) 'entry point defaults to one thread'
    Assert-Test ($kvParameter.DefaultValue.SafeGetValue() -eq $true) 'entry point defaults KV offload on'
    Write-Host "$passed CPU-only checks passed. No server/model/GPU process was launched."
} finally {
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
