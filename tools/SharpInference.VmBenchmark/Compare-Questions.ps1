param(
    [Parameter(Mandatory = $true)][string] $ModelDirectory,
    [Parameter(Mandatory = $true)][string] $OutputDirectory,
    [int] $Rounds = 2
)

$ErrorActionPreference = 'Stop'
if ($Rounds -lt 1) { throw 'Rounds must be positive.' }
$tool = Join-Path $PSScriptRoot 'bin\Release\net10.0-windows10.0.19041.0\SharpInference.VmBenchmark.dll'
$profile = Join-Path $OutputDirectory 'gpu-t1-profile.json'
foreach ($path in @($tool, $profile)) {
    if (!(Test-Path -LiteralPath $path)) { throw "Required tool/profile is missing: $path." }
}
$previousTiered = $env:DOTNET_TieredCompilation
try {
    $env:DOTNET_TieredCompilation = '0'
    foreach ($model in @(
        @{ Id = 'v6'; File = 'RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin' },
        @{ Id = 'v7'; File = 'rwkv7-g1j-7.2b-20260831-ctx16384-FP16.bin' }
    )) {
        $reference = Join-Path $OutputDirectory "$($model.Id)-serial-r0"
        for ($round = 0; $round -lt $Rounds; $round++) {
            $order = if ($round % 2 -eq 0) { @('serial', 'default') } else { @('default', 'serial') }
            foreach ($variant in $order) {
                $prefix = Join-Path $OutputDirectory "$($model.Id)-$variant-r$round"
                $command = @($tool, 'questions', '--model', (Join-Path $ModelDirectory $model.File),
                    '--target', 'gpu', '--variant', $variant, '--output', $prefix, '--profile', $profile)
                if ($variant -eq 'serial') { $command += @('--matvec', 'serial') }
                if (!($variant -eq 'serial' -and $round -eq 0)) { $command += @('--reference', $reference) }
                & dotnet @command
                if ($LASTEXITCODE -ne 0) { throw "Failed $($model.Id)/$variant/round $round, exit $LASTEXITCODE." }
                $measurement = Get-Content -Raw -LiteralPath "$prefix.json" | ConvertFrom-Json
                if ($measurement.Status -ne 'completed' -or $measurement.Results.Count -ne 10) {
                    throw "Incomplete question benchmark: $prefix."
                }
                if ($measurement.TierOneDecision.SelectedCandidates -eq 0) {
                    throw "No T1 selected for $prefix; regenerate the profile with tiered compilation disabled."
                }
                if ($variant -eq 'default' -and $measurement.MatVecDecision.CooperativeCalls -eq 0) {
                    throw "The default did not select cooperative MatVec: $prefix."
                }
            }
        }
    }
}
finally {
    $env:DOTNET_TieredCompilation = $previousTiered
}
