param([string]$ProjectRoot = "D:\111\Desktop\ctp-fish")

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$evaluator = Join-Path $ProjectRoot "scripts\evaluate_cxt_phase1c.py"
$config = Join-Path $ProjectRoot "configs\cxt_fish_phase1c.yaml"

foreach ($seed in @(2026, 17)) {
    $checkpoint = Join-Path $ProjectRoot "outputs\cxt_fish\phase1a_replication\A1_s1_ce_seed$seed\best.pt"
    $evaluation = "phase1c_f0_seed$seed"
    $metric = Join-Path $ProjectRoot "outputs\cxt_fish\phase1a_replication\A1_s1_ce_seed$seed\evaluation_$evaluation\metrics_val.json"
    if (Test-Path $metric) { Write-Host "SKIP_COMPLETE F0_seed$seed"; continue }
    if (-not (Test-Path $checkpoint)) { throw "Missing frozen A1 checkpoint: $checkpoint" }
    Write-Host "START F0_seed$seed"
    & $python $evaluator --config $config --checkpoint $checkpoint --evaluation-name $evaluation
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for F0_seed$seed with exit code $LASTEXITCODE" }
    if (-not (Test-Path $metric)) { throw "Missing F0 validation metric output" }
    Write-Host "COMPLETE F0_seed$seed"
}
Write-Host "ALL_COMPLETE"
