param(
    [string]$PackageName = "agentmesh-runtime-linux-x86_64"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..")
$DistDir = Join-Path $RootDir "dist"
$PackageDir = Join-Path $DistDir $PackageName
$ArchivePath = Join-Path $DistDir "$PackageName.tar.gz"

Set-Location $RootDir

if (Test-Path $PackageDir) {
    Remove-Item -LiteralPath $PackageDir -Recurse -Force
}
if (Test-Path $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

$TopLevelExcludes = @(
    ".git",
    ".agents",
    ".codex",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    "data",
    "runs",
    ".env",
    "generated_code.py"
)

foreach ($Item in Get-ChildItem -LiteralPath $RootDir -Force) {
    $Skip = $false
    if ($TopLevelExcludes -contains $Item.Name) {
        $Skip = $true
    }
    if ($Item.Name -like ".env.*" -and $Item.Name -ne ".env.example") {
        $Skip = $true
    }
    if (-not $Skip) {
        Copy-Item -LiteralPath $Item.FullName -Destination $PackageDir -Recurse -Force
    }
}

foreach ($Item in Get-ChildItem -LiteralPath $RootDir -Force -File) {
    $Skip = $false
    if ($TopLevelExcludes -contains $Item.Name) {
        $Skip = $true
    }
    if ($Item.Name -like ".env.*" -and $Item.Name -ne ".env.example") {
        $Skip = $true
    }
    if (-not $Skip) {
        Copy-Item -LiteralPath $Item.FullName -Destination (Join-Path $PackageDir $Item.Name) -Force
    }
}

$TopLevelFiles = @(
    ".dockerignore",
    ".env.example",
    ".gitignore",
    ".python-version",
    "Cargo.lock",
    "Cargo.toml",
    "docker-compose.embedding.yml",
    "docker-compose.yml",
    "Dockerfile",
    "Makefile",
    "pyproject.toml",
    "README.md",
    "uv.lock"
)

foreach ($FileName in $TopLevelFiles) {
    $SourcePath = Join-Path $RootDir $FileName
    if (Test-Path $SourcePath) {
        Copy-Item -LiteralPath $SourcePath -Destination (Join-Path $PackageDir $FileName) -Force
    }
}

Get-ChildItem -LiteralPath $PackageDir -Recurse -Force -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $PackageDir -Recurse -Force -File |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

Copy-Item -LiteralPath (Join-Path $RootDir "release\install.sh") -Destination (Join-Path $PackageDir "install.sh") -Force
Copy-Item -LiteralPath (Join-Path $RootDir "release\start-shell.sh") -Destination (Join-Path $PackageDir "start-shell.sh") -Force
Copy-Item -LiteralPath (Join-Path $RootDir "release\run-benchmark.sh") -Destination (Join-Path $PackageDir "run-benchmark.sh") -Force
Copy-Item -LiteralPath (Join-Path $RootDir "release\stop-services.sh") -Destination (Join-Path $PackageDir "stop-services.sh") -Force
Copy-Item -LiteralPath (Join-Path $RootDir "release\README_RELEASE.md") -Destination (Join-Path $PackageDir "README_RELEASE.md") -Force

$BenchmarkSource = Join-Path $RootDir "runs\latest\benchmarks"
if (Test-Path $BenchmarkSource) {
    $BenchmarkDest = Join-Path $PackageDir "benchmark-results"
    New-Item -ItemType Directory -Force -Path $BenchmarkDest | Out-Null
    Copy-Item -Path (Join-Path $BenchmarkSource "*") -Destination $BenchmarkDest -Recurse -Force
}

tar -C $DistDir -czf $ArchivePath $PackageName

Write-Host "Release archive written:"
Write-Host "  $ArchivePath"
