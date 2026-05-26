$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null

$commonArgs = @(
  "--root", "data\raw\xbd\train",
  "--train-csv", "data\processed\splits_noleak_full_train\train_pairs.csv",
  "--val-csv", "data\processed\splits_noleak_full_train\val_pairs.csv",
  "--test-csv", "data\processed\splits_noleak_full_train\test_pairs.csv",
  "--model", "unetplusplus_effb3",
  "--image-size", "1024",
  "--batch-size", "2",
  "--epochs", "100",
  "--lr", "1e-4",
  "--augment-mode", "safe",
  "--target-mode", "building-binary",
  "--device", "cuda",
  "--amp",
  "--num-workers", "0"
)

$runs = @(
  @{
    Name = "building_pre_unetplusplus_effb3_focaltversky_1024_bs2_100epochs"
    Args = @("--input-mode", "pre", "--loss", "focal-tversky")
  },
  @{
    Name = "building_pre_unetplusplus_effb3_bcedice_1024_bs2_100epochs"
    Args = @("--input-mode", "pre", "--loss", "bce-dice")
  },
  @{
    Name = "building_prepost_unetplusplus_effb3_focaltversky_1024_bs2_100epochs"
    Args = @("--input-mode", "pre-post", "--loss", "focal-tversky")
  }
)

foreach ($run in $runs) {
  $name = $run.Name
  $outDir = "outputs\checkpoints\$name"
  $logFile = "outputs\logs\$name.log"

  Write-Host ""
  Write-Host "============================================="
  Write-Host "Starting run: $name"
  Write-Host "Output: $outDir"
  Write-Host "Log: $logFile"
  Write-Host "============================================="
  Write-Host ""

  python -u scripts\train_building_segmentation.py `
    @commonArgs `
    @($run.Args) `
    --output-dir $outDir 2>&1 | Tee-Object -FilePath $logFile

  if ($LASTEXITCODE -ne 0) {
    Write-Host "Run failed: $name"
    Write-Host "Continuing to next run if any."
  } else {
    Write-Host "Run completed: $name"
  }
}
