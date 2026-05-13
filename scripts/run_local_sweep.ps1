param(
    [switch]$SkipExisting,
    [switch]$Force,
    [int]$MaxExperiments = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DataRoot = Join-Path $ProjectRoot "data\raw\xbd\train"
$TrainCsv = Join-Path $ProjectRoot "data\processed\splits\train_pairs.csv"
$ValCsv = Join-Path $ProjectRoot "data\processed\splits\val_pairs.csv"
$TestCsv = Join-Path $ProjectRoot "data\processed\splits\test_pairs.csv"
$CheckpointsRoot = Join-Path $ProjectRoot "outputs\checkpoints"
$PredictionsRoot = Join-Path $ProjectRoot "outputs\predictions"
$SummaryCsv = Join-Path $PredictionsRoot "local_sweep_summary.csv"

$TrainScript = "src\crisismap\training\train_unet.py"
$EvalScript = "src\crisismap\evaluation\evaluate_unet.py"

$Experiments = @(
    [pscustomobject]@{
        Name = "unet_512_ce_dice_w01_1_4_50epochs"
        Loss = "ce-dice"
        ClassWeights = @("0.10", "1.0", "4.0")
        Lr = "1e-4"
        Epochs = 50
    },
    [pscustomobject]@{
        Name = "unet_512_ce_dice_w005_1_3_50epochs"
        Loss = "ce-dice"
        ClassWeights = @("0.05", "1.0", "3.0")
        Lr = "1e-4"
        Epochs = 50
    },
    [pscustomobject]@{
        Name = "unet_512_ce_dice_w005_1_5_50epochs"
        Loss = "ce-dice"
        ClassWeights = @("0.05", "1.0", "5.0")
        Lr = "1e-4"
        Epochs = 50
    },
    [pscustomobject]@{
        Name = "unet_512_weighted_ce_w005_1_4_50epochs"
        Loss = "weighted-ce"
        ClassWeights = @("0.05", "1.0", "4.0")
        Lr = "1e-4"
        Epochs = 50
    },
    [pscustomobject]@{
        Name = "unet_512_ce_dice_w005_1_4_lr5e5_50epochs"
        Loss = "ce-dice"
        ClassWeights = @("0.05", "1.0", "4.0")
        Lr = "5e-5"
        Epochs = 50
    },
    [pscustomobject]@{
        Name = "unet_512_ce_dice_w01_1_5_lr5e5_50epochs"
        Loss = "ce-dice"
        ClassWeights = @("0.10", "1.0", "5.0")
        Lr = "5e-5"
        Epochs = 50
    }
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

function Require-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory not found: $Path"
    }
}

function Invoke-CheckedPython {
    param(
        [string]$Description,
        [string[]]$Arguments
    )

    Write-Step $Description
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Get-ExperimentPaths {
    param([object]$Experiment)

    $OutputDir = Join-Path $CheckpointsRoot $Experiment.Name
    $Checkpoint = Join-Path $OutputDir "best_unet.pt"
    $MetricsJson = Join-Path $PredictionsRoot "$($Experiment.Name)_test_metrics.json"

    return [pscustomobject]@{
        OutputDir = $OutputDir
        Checkpoint = $Checkpoint
        MetricsJson = $MetricsJson
    }
}

function Invoke-Training {
    param(
        [object]$Experiment,
        [object]$Paths
    )

    $Args = @(
        $TrainScript,
        "--root", $DataRoot,
        "--train-csv", $TrainCsv,
        "--val-csv", $ValCsv,
        "--output-dir", $Paths.OutputDir,
        "--image-size", "512",
        "--batch-size", "2",
        "--epochs", [string]$Experiment.Epochs,
        "--target-mode", "3-class",
        "--loss", $Experiment.Loss,
        "--class-weights"
    ) + $Experiment.ClassWeights + @(
        "--lr", $Experiment.Lr
    )

    Invoke-CheckedPython "Train $($Experiment.Name)" $Args
}

function Invoke-Evaluation {
    param(
        [object]$Experiment,
        [object]$Paths
    )

    Require-File $Paths.Checkpoint

    $Args = @(
        $EvalScript,
        "--root", $DataRoot,
        "--split-csv", $TestCsv,
        "--checkpoint", $Paths.Checkpoint,
        "--output", $Paths.MetricsJson,
        "--image-size", "512",
        "--batch-size", "2",
        "--target-mode", "3-class"
    )

    Invoke-CheckedPython "Evaluate $($Experiment.Name)" $Args
}

function Get-MetricValue {
    param(
        [object]$Values,
        [int]$Index
    )

    if ($null -eq $Values) {
        return $null
    }
    if ($Values.Count -le $Index) {
        return $null
    }
    return $Values[$Index]
}

function New-SummaryRows {
    $Rows = @()
    foreach ($Experiment in $ExperimentsToRun) {
        $Paths = Get-ExperimentPaths $Experiment
        if (-not (Test-Path -LiteralPath $Paths.MetricsJson -PathType Leaf)) {
            Write-Warning "Skipping summary row; metrics JSON not found: $($Paths.MetricsJson)"
            continue
        }

        $Metrics = Get-Content -LiteralPath $Paths.MetricsJson -Raw | ConvertFrom-Json
        $Iou = $Metrics.iou_per_class
        $Precision = $Metrics.precision_per_class
        $Recall = $Metrics.recall_per_class
        $F1 = $Metrics.f1_per_class

        $Rows += [pscustomobject]@{
            experiment = $Experiment.Name
            loss = $Experiment.Loss
            class_weights = ($Experiment.ClassWeights -join " ")
            lr = $Experiment.Lr
            epochs = $Experiment.Epochs
            pixel_accuracy = $Metrics.pixel_accuracy
            mean_iou = $Metrics.mean_iou
            iou_background = Get-MetricValue $Iou 0
            iou_no_damage = Get-MetricValue $Iou 1
            iou_damaged = Get-MetricValue $Iou 2
            precision_damaged = Get-MetricValue $Precision 2
            recall_damaged = Get-MetricValue $Recall 2
            f1_background = Get-MetricValue $F1 0
            f1_no_damage = Get-MetricValue $F1 1
            f1_damaged = Get-MetricValue $F1 2
            metrics_json = $Paths.MetricsJson
            checkpoint = $Paths.Checkpoint
        }
    }
    return $Rows
}

function Write-SortedSummary {
    param([object[]]$Rows)

    if (-not $Rows -or $Rows.Count -eq 0) {
        Write-Warning "No completed metrics were found for the summary."
        return
    }

    $Sorted = $Rows | Sort-Object `
        @{Expression = "iou_damaged"; Descending = $true},
        @{Expression = "f1_damaged"; Descending = $true},
        @{Expression = "mean_iou"; Descending = $true}

    Write-Step "Final sorted summary"
    $Sorted |
        Select-Object experiment, mean_iou, iou_damaged, f1_damaged, pixel_accuracy |
        Format-Table -AutoSize
}

if ($MaxExperiments -lt 0) {
    throw "-MaxExperiments must be zero or a positive integer."
}

if ($MaxExperiments -gt 0) {
    $ExperimentsToRun = @($Experiments | Select-Object -First $MaxExperiments)
}
else {
    $ExperimentsToRun = $Experiments
}

Write-Host "Aftermath / CrisisMap AI local U-Net sweep"
Write-Host "Project root: $ProjectRoot"
Write-Host ""
Write-Host "Current best to beat:"
Write-Host "  mean_iou = 0.6363"
Write-Host "  iou_damaged = 0.4159"
Write-Host "  f1_damaged = 0.5875"
Write-Host ""
Write-Host "Estimated runtime:"
Write-Host "  about 2h10 per 50-epoch run locally"
Write-Host "  about 13h total for 6 runs"
Write-Host ""
Write-Host "Experiments selected: $($ExperimentsToRun.Count)"
if ($Force) {
    Write-Host "Force mode: training and evaluation will rerun even when outputs exist."
}
elseif ($SkipExisting) {
    Write-Host "SkipExisting mode: experiments with existing metrics JSON will be skipped."
}

Require-File $PythonExe
Require-Directory $DataRoot
Require-File $TrainCsv
Require-File $ValCsv
Require-File $TestCsv
New-Item -ItemType Directory -Path $CheckpointsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $PredictionsRoot -Force | Out-Null

foreach ($Experiment in $ExperimentsToRun) {
    $Paths = Get-ExperimentPaths $Experiment

    Write-Step "Experiment: $($Experiment.Name)"
    Write-Host "Loss: $($Experiment.Loss)"
    Write-Host "Class weights: $($Experiment.ClassWeights -join ' ')"
    Write-Host "Learning rate: $($Experiment.Lr)"
    Write-Host "Epochs: $($Experiment.Epochs)"

    $MetricsExists = Test-Path -LiteralPath $Paths.MetricsJson -PathType Leaf
    $CheckpointExists = Test-Path -LiteralPath $Paths.Checkpoint -PathType Leaf
    $OutputDirExists = Test-Path -LiteralPath $Paths.OutputDir -PathType Container

    if (-not $Force -and $MetricsExists) {
        if ($SkipExisting) {
            Write-Host "Metrics JSON already exists; skipping experiment entirely."
        }
        else {
            Write-Host "Metrics JSON already exists; skipping to avoid overwriting. Use -Force to rerun."
        }
        continue
    }

    if (-not $Force -and $CheckpointExists -and -not $MetricsExists) {
        Write-Host "Best checkpoint exists but metrics JSON is missing; running evaluation only."
        Invoke-Evaluation $Experiment $Paths
        continue
    }

    if (-not $Force -and $OutputDirExists -and -not $CheckpointExists) {
        Write-Warning "Output directory exists but best_unet.pt is missing: $($Paths.OutputDir)"
        Write-Warning "The current training script does not support resume. Use -Force to start a fresh run that overwrites outputs in this directory."
        continue
    }

    Invoke-Training $Experiment $Paths
    Invoke-Evaluation $Experiment $Paths
}

Write-Step "Build summary CSV"
$SummaryRows = @(New-SummaryRows)
if ($SummaryRows.Count -gt 0) {
    $SummaryRows | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation
    Write-Host "Saved summary CSV: $SummaryCsv"
}
else {
    Write-Warning "No summary CSV was written because no metrics JSON files were found."
}

Write-SortedSummary $SummaryRows
