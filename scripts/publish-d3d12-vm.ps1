param(
    [Parameter(Mandatory = $true)][string]$ArtifactDirectory,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$artifact = (Resolve-Path -LiteralPath $ArtifactDirectory).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\src\SharpInference.Vm.D3D12Template\SharpInference.Vm.D3D12Template.csproj"))
& dotnet publish $project -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true "-p:VmArtifactDirectory=$artifact" -o $output
if ($LASTEXITCODE -ne 0) {
    throw "DXIL VM publication failed with exit code $LASTEXITCODE."
}
