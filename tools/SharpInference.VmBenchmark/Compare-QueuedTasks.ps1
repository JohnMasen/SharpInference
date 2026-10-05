param(
    [Parameter(Mandatory = $true)][string] $MainDirectory,
    [Parameter(Mandatory = $true)][string] $ModelDirectory,
    [Parameter(Mandatory = $true)][string] $OutputDirectory,
    [int] $Rounds = 2,
    [switch] $SummarizeOnly
)

$ErrorActionPreference = 'Stop'
if ($Rounds -lt 1) { throw 'Rounds must be positive.' }
$relativeTool = 'tools\SharpInference.VmBenchmark\bin\Release\net10.0-windows10.0.19041.0\SharpInference.VmBenchmark.dll'
$tools = @{
    main = Join-Path $MainDirectory $relativeTool
    queued = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) $relativeTool
}
foreach ($tool in $tools.Values) {
    if (!(Test-Path -LiteralPath $tool)) { throw "Build the benchmark first: $tool." }
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$models = @(
    @{ Id = 'v6'; File = 'RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin' },
    @{ Id = 'v7'; File = 'rwkv7-g1j-7.2b-20260831-ctx16384-FP16.bin' }
)
$previousTiered = $env:DOTNET_TieredCompilation
function Get-Median($Values) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { throw 'Cannot summarize empty measurements.' }
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$middle] }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2
}
try {
    $env:DOTNET_TieredCompilation = '0'
    $modelPaths = ($models | ForEach-Object { Join-Path $ModelDirectory $_.File }) -join '|'
    $profileVariants = if ($SummarizeOnly) { @() } else { @('main', 'queued') }
    foreach ($variant in $profileVariants) {
        & dotnet $tools[$variant] profile --models $modelPaths --target gpu `
            --output (Join-Path $OutputDirectory "$variant-gpu-t1-profile.json")
        if ($LASTEXITCODE -ne 0) { throw "GPU profiling failed for $variant." }
    }
    $runTargets = if ($SummarizeOnly) { @() } else { @('gpu', 'cpu') }
    foreach ($target in $runTargets) {
        foreach ($model in $models) {
            $reference = Join-Path $OutputDirectory "$($model.Id)-$target-main-r0"
            for ($round = 0; $round -lt $Rounds; $round++) {
                $order = if ($round % 2 -eq 0) { @('main', 'queued') } else { @('queued', 'main') }
                foreach ($variant in $order) {
                    $prefix = Join-Path $OutputDirectory "$($model.Id)-$target-$variant-r$round"
                    $mode = if ($target -eq 'gpu') { 'questions' } else { 'run' }
                    $command = @($tools[$variant], $mode, '--model', (Join-Path $ModelDirectory $model.File),
                        '--target', $target, '--variant', $variant, '--output', $prefix)
                    if ($target -eq 'gpu') {
                        $command += @('--profile', (Join-Path $OutputDirectory "$variant-gpu-t1-profile.json"))
                    }
                    else {
                        $command += @('--samples', '7', '--tokens', '8', '--warmups', '2', '--minimum-warmup-ms', '2000')
                    }
                    if (!($variant -eq 'main' -and $round -eq 0)) { $command += @('--reference', $reference) }
                    & dotnet @command
                    if ($LASTEXITCODE -ne 0) { throw "Failed $($model.Id)/$target/$variant/round $round." }
                    $report = Get-Content -Raw -LiteralPath "$prefix.json" | ConvertFrom-Json
                    if ($target -eq 'gpu') {
                        $expected = if ($model.Id -eq 'v6') { 644 } else { 674 }
                        $expectedMatVec = if ($model.Id -eq 'v6') { 353 } else { 447 }
                        $expectedNodes = if ($model.Id -eq 'v6') { 9045 } else { 3697 }
                        if ($report.Status -ne 'completed' -or $report.Results.Count -ne 10 -or
                            $report.TierOneDecision.SelectedCandidates -ne $expected -or
                            $report.TierOneDecision.OptimizedNodeCount -ne $expectedNodes -or
                            $report.MatVecDecision.CooperativeCalls -ne $expectedMatVec -or
                            ($report.Results | Where-Object { $_.OutputTokens.Count -ne 16 }).Count -ne 0) {
                            throw "Incomplete or noncomparable GPU graph: $prefix."
                        }
                        if ($variant -eq 'queued' -and $report.TaskStatistics.CompletedTasks -eq 0) {
                            throw "Queued GPU execution was not exercised: $prefix."
                        }
                    }
                }
            }
        }
    }
    $summary = foreach ($target in @('gpu', 'cpu')) {
        foreach ($model in $models) {
            $rates = @{}
            foreach ($variant in @('main', 'queued')) {
                $reports = @(for ($round = 0; $round -lt $Rounds; $round++) {
                    Get-Content -Raw -LiteralPath (Join-Path $OutputDirectory "$($model.Id)-$target-$variant-r$round.json") |
                        ConvertFrom-Json
                })
                if ($target -eq 'gpu') {
                    if (($reports | Where-Object { $_.Status -ne 'completed' -or $_.Results.Count -ne 10 }).Count -ne 0) {
                        throw "Incomplete GPU measurements for $($model.Id)/$variant."
                    }
                    $prefillMs = ($reports.Results | Measure-Object -Property PrefillMilliseconds -Sum).Sum
                    $decodeMs = ($reports.Results | Measure-Object -Property DecodeMilliseconds -Sum).Sum
                    $rates[$variant] = @{
                        Prefill = $Rounds * 167000 / $prefillMs
                        Decode = $Rounds * 160000 / $decodeMs
                        Combined = $Rounds * 160000 / ($prefillMs + $decodeMs)
                    }
                }
                else {
                    if (($reports | Where-Object { $_.TierOneCalls -ne 0 -or $_.Samples -ne 7 }).Count -ne 0) {
                        throw "CPU protocol changed for $($model.Id)/$variant."
                    }
                    $rates[$variant] = @{
                        Prefill = 1000 / (Get-Median $reports.PrefillMillisecondsPerToken)
                        Decode = 1000 / (Get-Median $reports.DecodeMillisecondsPerToken)
                    }
                }
            }
            foreach ($metric in @('Prefill', 'Decode', 'Combined')) {
                if ($target -eq 'cpu' -and $metric -eq 'Combined') { continue }
                $baseline = $rates.main[$metric]
                $candidate = $rates.queued[$metric]
                [pscustomobject]@{
                    Model = $model.Id; Target = $target; Metric = $metric
                    MainTokensPerSecond = $baseline; QueuedTokensPerSecond = $candidate
                    Speedup = $candidate / $baseline; ChangePercent = 100 * ($candidate / $baseline - 1)
                }
            }
        }
    }
    $summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding utf8
    $summary | Format-Table -AutoSize
}
finally {
    $env:DOTNET_TieredCompilation = $previousTiered
}
