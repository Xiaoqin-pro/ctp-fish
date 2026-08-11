param(
    [string]$ProjectRoot = "D:\111\Desktop\ctp-fish",
    [string]$OutputRoot = "outputs\cxt_fish\phase1c_replication"
)

$ErrorActionPreference = "Stop"
$seeds = @(2026, 17)
$variants = @("f1", "f2", "f3")
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$evaluator = Join-Path $ProjectRoot "scripts\evaluate_cxt_phase1c.py"
$config = Join-Path $ProjectRoot "configs\cxt_fish_phase1c.yaml"

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $run = "$($variant.ToUpper())_seed$seed"
        $checkpoint = Join-Path $ProjectRoot "$OutputRoot\$run\best.pt"
        $evaluation = "phase1c_$variant`_seed$seed"
        $metric = Join-Path $ProjectRoot "$OutputRoot\$run\evaluation_$evaluation\metrics_val.json"
        if (Test-Path $metric) { Write-Host "SKIP_COMPLETE $run"; continue }
        if (-not (Test-Path $checkpoint)) { throw "Missing frozen checkpoint: $checkpoint" }
        Write-Host "START $run"
        & $python $evaluator --config $config --checkpoint $checkpoint --evaluation-name $evaluation
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $run with exit code $LASTEXITCODE" }
        if (-not (Test-Path $metric)) { throw "Missing validation metric output for $run" }
        Write-Host "COMPLETE $run"
    }
}
Write-Host "ALL_COMPLETE"
