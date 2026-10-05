param(
    [Parameter(Mandatory = $true)][string] $ModelDirectory,
    [Parameter(Mandatory = $true)][string] $OutputDirectory,
    [int] $Rounds = 3,
    [int] $Samples = 3,
    [int] $Tokens = 4,
    [int] $Warmups = 1
)

$ErrorActionPreference = 'Stop'
if ($Rounds -lt 1 -or $Samples -lt 1 -or $Tokens -lt 1 -or $Warmups -lt 1) {
    throw 'Rounds, samples, tokens and warmups must be positive.'
}
$tool = Join-Path $PSScriptRoot 'bin\Release\net10.0-windows10.0.19041.0\SharpInference.VmBenchmark.dll'
$mainTool = Join-Path $PSScriptRoot 'bin\MainBaseline\SharpInference.VmBenchmark.dll'
if (!(Test-Path -LiteralPath $tool) -or !(Test-Path -LiteralPath $mainTool)) {
    throw 'Build the current and fixed-main harnesses before running the comparison.'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$models = @(
    @{ Id = 'tiny-v6'; File = 'tiny-rwkv-6v0-3m-FP32.bin' },
    @{ Id = 'tiny-v7'; File = 'tiny-rwkv-7v0-834K-FP32.bin' },
    @{ Id = 'large-v6'; File = 'RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin' },
    @{ Id = 'large-v7'; File = 'rwkv7-g1j-7.2b-20260831-ctx16384-FP16.bin' }
)
foreach ($model in $models) {
    foreach ($target in @('cpu', 'gpu')) {
        $key = "$($model.Id)-$target"
        $reference = Join-Path $OutputDirectory "$key-main-r0"
        for ($round = 0; $round -lt $Rounds; $round++) {
            $order = switch ($round % 3) {
                0 { @('main', 'off', 'on') }
                1 { @('off', 'on', 'main') }
                2 { @('on', 'main', 'off') }
            }
            foreach ($variant in $order) {
                $runner = if ($variant -eq 'main') { $mainTool } else { $tool }
                $prefix = Join-Path $OutputDirectory "$key-$variant-r$round"
                $command = @($runner, 'run', '--model', (Join-Path $ModelDirectory $model.File),
                    '--target', $target, '--variant', $variant, '--output', $prefix,
                    '--samples', "$Samples", '--tokens', "$Tokens", '--warmups', "$Warmups")
                if ($variant -eq 'on') {
                    $command += @('--profile', (Join-Path $OutputDirectory "$target-profile.json"))
                }
                if (!($variant -eq 'main' -and $round -eq 0)) { $command += @('--reference', $reference) }
                & dotnet @command
                if ($LASTEXITCODE -ne 0) { throw "Failed $key/$variant/round $round, exit $LASTEXITCODE." }
            }
        }
    }
}
