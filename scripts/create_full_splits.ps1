Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$IndexCsv = Join-Path $ProjectRoot "data\processed\xbd_train_index.csv"
$OutputDir = Join-Path $ProjectRoot "data\processed\splits_full"
$SummaryCsv = Join-Path $OutputDir "split_summary.csv"
$SplitScript = "src\crisismap\data\create_xbd_splits.py"

$Disasters = @(
    "guatemala-volcano",
    "hurricane-florence",
    "hurricane-harvey",
    "hurricane-matthew",
    "hurricane-michael",
    "mexico-earthquake",
    "midwest-flooding",
    "palu-tsunami",
    "santa-rosa-wildfire",
    "socal-fire"
)

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}

function Invoke-Python {
    param([string[]]$Arguments)

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "create_xbd_splits.py failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Aftermath / CrisisMap AI full-disaster split helper"
Write-Host "Project root: $ProjectRoot"
Write-Host "Output directory: $OutputDir"
Write-Host ""
Write-Host "Useful-sample filter: min nonzero ratio = 0.01"
Write-Host "Seed: 42"
Write-Host "Split ratios: train/val/test with val-size=0.15 and test-size=0.15"

Require-File $PythonExe
Require-File $IndexCsv

$Args = @(
    $SplitScript,
    "--index", $IndexCsv,
    "--output-dir", $OutputDir,
    "--disasters"
) + $Disasters + @(
    "--val-size", "0.15",
    "--test-size", "0.15",
    "--min-nonzero-ratio", "0.01",
    "--seed", "42"
)

Write-Step "Create full-disaster train/val/test splits"
Invoke-Python $Args

if (Test-Path -LiteralPath $SummaryCsv -PathType Leaf) {
    Write-Step "Generated split summary"
    Write-Host "Summary CSV: $SummaryCsv"
    Import-Csv -LiteralPath $SummaryCsv | Format-Table -AutoSize
}
else {
    Write-Warning "Expected split summary was not found: $SummaryCsv"
}
