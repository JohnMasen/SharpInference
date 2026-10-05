param(
    [Parameter(Mandatory = $true)][string] $ModelDirectory,
    [Parameter(Mandatory = $true)][string] $OutputDirectory,
    [int] $Rounds = 2,
    [int] $Samples = 7,
    [int] $Tokens = 8,
    [int] $Warmups = 2
)

$ErrorActionPreference = 'Stop'
if ($Rounds -lt 1 -or $Samples -lt 1 -or $Tokens -lt 1 -or $Warmups -lt 1) {
    throw 'Rounds, samples, tokens and warmups must be positive.'
}
$tool = Join-Path $PSScriptRoot 'bin\Release\net10.0-windows10.0.19041.0\SharpInference.VmBenchmark.dll'
$matVecProfile = Join-Path $OutputDirectory 'gpu-matvec-profile.json'
$tierOneProfile = Join-Path $OutputDirectory 'gpu-t1-profile.json'
foreach ($path in @($tool, $matVecProfile, $tierOneProfile)) {
    if (!(Test-Path -LiteralPath $path)) { throw "Required tool/profile is missing: $path." }
}
$models = @(
    @{ Id = 'tiny-v6'; File = 'tiny-rwkv-6v0-3m-FP32.bin' },
    @{ Id = 'tiny-v7'; File = 'tiny-rwkv-7v0-834K-FP32.bin' },
    @{ Id = 'large-v6'; File = 'RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin' },
    @{ Id = 'large-v7'; File = 'rwkv7-g1j-7.2b-20260831-ctx16384-FP16.bin' }
)
foreach ($model in $models) {
    $reference = Join-Path $OutputDirectory "$($model.Id)-serial-r0"
    for ($round = 0; $round -lt $Rounds; $round++) {
        $order = if ($round % 2 -eq 0) { @('serial', 'profile') } else { @('profile', 'serial') }
        foreach ($variant in $order) {
            $prefix = Join-Path $OutputDirectory "$($model.Id)-$variant-r$round"
            $command = @($tool, 'run', '--model', (Join-Path $ModelDirectory $model.File),
                '--target', 'gpu', '--variant', $variant, '--output', $prefix,
                '--samples', "$Samples", '--tokens', "$Tokens", '--warmups', "$Warmups",
                '--profile', $tierOneProfile, '--matvec', $variant)
            if ($variant -eq 'profile') { $command += @('--matvec-profile', $matVecProfile) }
            if (!($variant -eq 'serial' -and $round -eq 0)) { $command += @('--reference', $reference) }
            & dotnet @command
            if ($LASTEXITCODE -ne 0) { throw "Failed $($model.Id)/$variant/round $round, exit $LASTEXITCODE." }
            $measurement = Get-Content -Raw -LiteralPath "$prefix.json" | ConvertFrom-Json
            if ($measurement.TierOneCalls -eq 0) { throw "No T1 selected for $prefix; check the current profile fingerprint." }
            if ($variant -eq 'profile' -and $measurement.MatVecDecision.CooperativeCalls -eq 0) {
                throw "No cooperative MatVec selected for $prefix; check profile diagnostics."
            }
        }
    }
}
