param(
    [string]$ProjectRoot = "D:\111\Desktop\ctp-fish",
    [string]$OutputRoot = "outputs\cxt_fish\phase1a_screen"
)

$ErrorActionPreference = "Stop"

# This queue is intentionally validation-only. It never accepts a test partition.
$runs = @(
    "A2_s2_ce_seed407",
    "A3_s3_ce_seed407",
    "A4I_s0_balanced_image_seed407",
    "A4T_s0_balanced_track_seed407",
    "A5I_s1_balanced_image_seed407",
    "A5T_s1_balanced_track_seed407"
)

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$config = Join-Path $ProjectRoot "configs\resnet18_gate0.yaml"
$evaluator = Join-Path $ProjectRoot "scripts\evaluate_gate0.py"

foreach ($run in $runs) {
    $runDirectory = Join-Path $ProjectRoot "$OutputRoot\$run"
    $evaluationDirectory = Join-Path $runDirectory "evaluation_phase1a_val"
    $requiredFiles = @(
        (Join-Path $evaluationDirectory "metrics_val.json"),
        (Join-Path $evaluationDirectory "per_image_val.csv"),
        (Join-Path $evaluationDirectory "per_track_val.csv")
    )

    if (($requiredFiles | Where-Object { -not (Test-Path $_) }).Count -eq 0) {
        Write-Host "SKIP_COMPLETE $run"
        continue
    }
    if (Test-Path $evaluationDirectory) {
        throw "Refusing to overwrite an incomplete evaluation: $evaluationDirectory"
    }

    $checkpoint = Join-Path $runDirectory "best.pt"
    if (-not (Test-Path $checkpoint)) {
        throw "Checkpoint not found: $checkpoint"
    }

    Write-Host "START $run"
    & $python $evaluator --config $config --checkpoint $checkpoint --split track --partition val --evaluation-name phase1a_val
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $run with exit code $LASTEXITCODE"
    }
    if (($requiredFiles | Where-Object { -not (Test-Path $_) }).Count -ne 0) {
        throw "Evaluation outputs missing for $run"
    }
    Write-Host "COMPLETE $run"
}

Write-Host "ALL_COMPLETE"
