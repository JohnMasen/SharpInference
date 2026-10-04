param(
    [Parameter(Mandatory = $true)][string]$ArtifactDirectory,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$RuntimeIdentifier = "win-x64",
    [switch]$Managed
)

$ErrorActionPreference = "Stop"
if ($RuntimeIdentifier -notmatch '^[a-z0-9-]+$') {
    throw "RuntimeIdentifier must be a .NET runtime identifier."
}
$artifact = (Resolve-Path -LiteralPath $ArtifactDirectory).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\src\SharpInference.Vm.Template\SharpInference.Vm.Template.csproj"))
$publish = @("publish", $project, "-c", "Release", "-r", $RuntimeIdentifier,
    "--self-contained", "true", "-p:VmArtifactDirectory=$artifact", "-o", $output)
if ($Managed) {
    $publish += @("-p:PublishSingleFile=true", "-p:PublishTrimmed=true", "-p:IncludeNativeLibrariesForSelfExtract=true")
} else {
    $publish += "-p:PublishAot=true"
}

if (-not $Managed -and $RuntimeIdentifier -eq "win-x64" -and
    [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "Install Visual Studio Build Tools with the C++ toolchain to publish a Windows Native AOT VM."
    }
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installation) {
        throw "Visual Studio's C++ toolchain was not found."
    }
    $setup = Join-Path $installation "VC\Auxiliary\Build\vcvars64.bat"
    $quoted = $publish | ForEach-Object { '"' + $_.Replace('"', '""') + '"' }
    & $env:ComSpec /c ('call "' + $setup + '" >nul && dotnet ' + ($quoted -join ' '))
} else {
    & dotnet @publish
}
if ($LASTEXITCODE -ne 0) {
    throw "VM publication failed with exit code $LASTEXITCODE."
}
