param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$ArchiveDir = Join-Path $ProjectRoot "data\raw\archives"
$XbdRoot = Join-Path $ProjectRoot "data\raw\xbd"
$TrainRoot = Join-Path $XbdRoot "train"
$GeotransformsRoot = Join-Path $ProjectRoot "data\raw\geotransforms"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"
$SplitsDir = Join-Path $ProcessedDir "splits"
$CheckpointsDir = Join-Path $ProjectRoot "outputs\checkpoints"
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

$TrainArchive = Join-Path $ArchiveDir "train_images_labels_targets.tar"
$GeotransformsArchive = Join-Path $ArchiveDir "xview_geotransforms.json.tgz"
$GeotransformsJson = Join-Path $GeotransformsRoot "xview_geotransforms.json"
$IndexCsv = Join-Path $ProcessedDir "xbd_train_index.csv"
$TrainCsv = Join-Path $SplitsDir "train_pairs.csv"
$ValCsv = Join-Path $SplitsDir "val_pairs.csv"
$TestCsv = Join-Path $SplitsDir "test_pairs.csv"
$SplitSummaryCsv = Join-Path $SplitsDir "split_summary.csv"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

$Disasters = @(
    "hurricane-harvey",
    "hurricane-michael",
    "santa-rosa-wildfire",
    "palu-tsunami"
)

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [string]$Description,
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Step $Description
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
        Write-Host "Created directory: $Path"
    }
}

function Require-File {
    param(
        [string]$Path,
        [string]$HelpMessage
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing required file: $Path`n$HelpMessage"
    }
}

function Test-DatasetFolders {
    $RequiredFolders = @(
        (Join-Path $TrainRoot "images"),
        (Join-Path $TrainRoot "labels"),
        (Join-Path $TrainRoot "targets")
    )

    foreach ($Folder in $RequiredFolders) {
        if (-not (Test-Path -LiteralPath $Folder -PathType Container)) {
            return $false
        }
    }
    return $true
}

function Ensure-PythonVenv {
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        Write-Host "Using existing virtual environment: $VenvDir"
        return
    }

    $Created = $false
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $PythonCommand) {
        try {
            Invoke-Native "Create Python virtual environment" $PythonCommand.Source @("-m", "venv", $VenvDir)
            $Created = $true
        }
        catch {
            Write-Warning "Could not create virtual environment with 'python': $($_.Exception.Message)"
        }
    }

    if (-not $Created -and -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $PyLauncher) {
            throw "Python 3 was not found. Install Python 3, then rerun this script."
        }
        Invoke-Native "Create Python virtual environment" $PyLauncher.Source @("-3", "-m", "venv", $VenvDir)
    }

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw "Virtual environment was not created at expected path: $VenvPython"
    }
}

function Ensure-GeotransformsJson {
    if (Test-Path -LiteralPath $GeotransformsJson -PathType Leaf) {
        return
    }

    $FoundJson = Get-ChildItem -LiteralPath $GeotransformsRoot -Recurse -Filter "xview_geotransforms.json" -File |
        Select-Object -First 1
    if ($null -eq $FoundJson) {
        throw "Could not find xview_geotransforms.json after extraction under: $GeotransformsRoot"
    }

    Copy-Item -LiteralPath $FoundJson.FullName -Destination $GeotransformsJson
    Write-Host "Copied geotransforms JSON to expected path: $GeotransformsJson"
}

function Test-SplitFiles {
    $Files = @($TrainCsv, $ValCsv, $TestCsv, $SplitSummaryCsv)
    foreach ($File in $Files) {
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

Write-Host "Aftermath / CrisisMap AI setup"
Write-Host "Project root: $ProjectRoot"
if ($Force) {
    Write-Host "Force mode: processed files will be rebuilt; raw archives will not be deleted."
}

Write-Step "Create required folders"
Ensure-Directory $ArchiveDir
Ensure-Directory $XbdRoot
Ensure-Directory $GeotransformsRoot
Ensure-Directory $ProcessedDir
Ensure-Directory $SplitsDir
Ensure-Directory $CheckpointsDir

Write-Step "Check required local archives"
Require-File $TrainArchive "Place train_images_labels_targets.tar in data\raw\archives."
Require-File $GeotransformsArchive "Place xview_geotransforms.json.tgz in data\raw\archives."

Ensure-PythonVenv
Invoke-Native "Upgrade pip" $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Require-File $Requirements "requirements.txt must exist at the project root."
Invoke-Native "Install Python dependencies" $VenvPython @("-m", "pip", "install", "-r", $Requirements)

$TarCommand = Get-Command tar -ErrorAction SilentlyContinue
if ($null -eq $TarCommand) {
    throw "The 'tar' command was not found. Install a Windows tar tool or use a PowerShell version that includes tar."
}

if ($Force -or -not (Test-DatasetFolders)) {
    Invoke-Native "Extract xBD training archive" $TarCommand.Source @("-xf", $TrainArchive, "-C", $XbdRoot)
}
else {
    Write-Host "Dataset folders already exist; skipping xBD archive extraction."
}

if ($Force -or -not (Test-Path -LiteralPath $GeotransformsJson -PathType Leaf)) {
    Invoke-Native "Extract xView geotransforms archive" $TarCommand.Source @("-xzf", $GeotransformsArchive, "-C", $GeotransformsRoot)
    Ensure-GeotransformsJson
}
else {
    Write-Host "Geotransforms JSON already exists; skipping geotransforms extraction."
}

Write-Step "Verify extracted dataset folders"
$ExpectedDatasetFolders = @(
    (Join-Path $TrainRoot "images"),
    (Join-Path $TrainRoot "labels"),
    (Join-Path $TrainRoot "targets")
)
foreach ($Folder in $ExpectedDatasetFolders) {
    if (-not (Test-Path -LiteralPath $Folder -PathType Container)) {
        throw "Expected dataset folder is missing after extraction: $Folder"
    }
    Write-Host "Found: $Folder"
}
Require-File $GeotransformsJson "The geotransforms archive should extract xview_geotransforms.json."

Invoke-Native "Inspect xBD dataset" $VenvPython @("src\crisismap\data\inspect_xbd.py", "--root", $TrainRoot)

if ($Force -or -not (Test-Path -LiteralPath $IndexCsv -PathType Leaf)) {
    Invoke-Native "Build xBD index CSV" $VenvPython @(
        "src\crisismap\data\build_xbd_index.py",
        "--root", $TrainRoot,
        "--output", $IndexCsv
    )
}
else {
    Write-Host "Index already exists; skipping index rebuild: $IndexCsv"
}

Invoke-Native "Summarize xBD index" $VenvPython @(
    "src\crisismap\data\summarize_xbd_index.py",
    "--index", $IndexCsv
)

if ($Force -or -not (Test-SplitFiles)) {
    $SplitArgs = @(
        "src\crisismap\data\create_xbd_splits.py",
        "--index", $IndexCsv,
        "--output-dir", $SplitsDir,
        "--disasters"
    ) + $Disasters + @(
        "--min-nonzero-ratio", "0.01",
        "--seed", "42"
    )
    Invoke-Native "Create train/val/test splits" $VenvPython $SplitArgs
}
else {
    Write-Host "Split CSVs already exist; skipping split rebuild: $SplitsDir"
}

Write-Step "Setup complete"
Write-Host "Processed index: $IndexCsv"
Write-Host "Split CSVs: $SplitsDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  Train a baseline model:"
Write-Host "    .\.venv\Scripts\python.exe src\crisismap\training\train_unet.py --root data\raw\xbd\train --train-csv data\processed\splits\train_pairs.csv --val-csv data\processed\splits\val_pairs.csv --output-dir outputs\checkpoints\unet_baseline_512_v2_30epochs --image-size 512 --batch-size 2 --epochs 30 --target-mode 3-class"
Write-Host ""
Write-Host "  Run the Streamlit app:"
Write-Host "    .\.venv\Scripts\streamlit.exe run app\streamlit_app.py"
Write-Host ""
Write-Host "  If you received a checkpoint, place it here:"
Write-Host "    outputs\checkpoints\unet_baseline_512_v2_30epochs\best_unet.pt"
