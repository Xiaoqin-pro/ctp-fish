param(
    [string]$ProjectRoot = "D:\111\Desktop\ctp-fish",
    [string]$OutputRoot = "outputs\cxt_fish\phase1b_replication"
)

$ErrorActionPreference = "Stop"

# Reserved v1.1 replication seeds after the predeclared seed-3407 pilot.
$seeds = @(2026, 17)
$variants = @("c0", "c1", "c2")
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $ProjectRoot "scripts\train_cxt_phase1b.py"
$config = Join-Path $ProjectRoot "configs\cxt_fish_phase1b.yaml"

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $run = "$($variant.ToUpper())_seed$seed"
        $directory = Join-Path $ProjectRoot "$OutputRoot\$run"
        $complete = (Test-Path (Join-Path $directory "best.pt")) -and (Test-Path (Join-Path $directory "last.pt")) -and (Test-Path (Join-Path $directory "training_curve.csv"))
        if ($complete) { Write-Host "SKIP_COMPLETE $run"; continue }
        if (Test-Path $directory) { throw "Refusing to overwrite incomplete Phase 1B run: $directory" }
        Write-Host "START $run"
        & $python $trainer --config $config --variant $variant --seed $seed --run-name $run --output-root $OutputRoot
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $run with exit code $LASTEXITCODE" }
        if (-not ((Test-Path (Join-Path $directory "best.pt")) -and (Test-Path (Join-Path $directory "last.pt")))) { throw "Required checkpoints missing for $run" }
        Write-Host "COMPLETE $run"
    }
}
Write-Host "ALL_COMPLETE"
