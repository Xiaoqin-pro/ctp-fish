param(
    [string]$ProjectRoot = "D:\111\Desktop\ctp-fish",
    [string]$OutputRoot = "outputs\cxt_fish\phase1b_replication"
)

$ErrorActionPreference = "Stop"
$runs = @("C0_seed2026", "C1_seed2026", "C2_seed2026", "C0_seed17", "C1_seed17", "C2_seed17")
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$evaluator = Join-Path $ProjectRoot "scripts\evaluate_cxt_phase1b.py"
$config = Join-Path $ProjectRoot "configs\cxt_fish_phase1b.yaml"

foreach ($run in $runs) {
    $variant = $run.Substring(0, 2).ToLower()
    $directory = Join-Path $ProjectRoot "$OutputRoot\$run"
    $evaluation = Join-Path $directory "evaluation_phase1b_val"
    $required = @((Join-Path $evaluation "metrics_val.json"), (Join-Path $evaluation "per_image_val.csv"), (Join-Path $evaluation "per_track_val.csv"))
    if (($required | Where-Object { -not (Test-Path $_) }).Count -eq 0) { Write-Host "SKIP_COMPLETE $run"; continue }
    if (Test-Path $evaluation) { throw "Refusing to overwrite incomplete evaluation: $evaluation" }
    $checkpoint = Join-Path $directory "best.pt"
    Write-Host "START $run"
    & $python $evaluator --config $config --checkpoint $checkpoint --variant $variant --evaluation-name phase1b_val
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $run with exit code $LASTEXITCODE" }
    if (($required | Where-Object { -not (Test-Path $_) }).Count -ne 0) { throw "Evaluation outputs missing for $run" }
    Write-Host "COMPLETE $run"
}
Write-Host "ALL_COMPLETE"
