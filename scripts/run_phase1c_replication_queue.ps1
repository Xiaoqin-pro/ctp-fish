param(
    [string]$ProjectRoot = "D:\111\Desktop\ctp-fish",
    [string]$OutputRoot = "outputs\cxt_fish\phase1c_replication"
)

$ErrorActionPreference = "Stop"

# Seeds were registered before reading the seed-3407 Phase 1C pilot result.
$seeds = @(2026, 17)
$variants = @("f1", "f2", "f3")
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $ProjectRoot "scripts\train_cxt_phase1c.py"
$config = Join-Path $ProjectRoot "configs\cxt_fish_phase1c.yaml"

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $run = "$($variant.ToUpper())_seed$seed"
        $directory = Join-Path $ProjectRoot "$OutputRoot\$run"
        $complete = (Test-Path (Join-Path $directory "best.pt")) -and (Test-Path (Join-Path $directory "last.pt")) -and (Test-Path (Join-Path $directory "training_curve.csv")) -and (Test-Path (Join-Path $directory "run_metadata.json"))
        if ($complete) { Write-Host "SKIP_COMPLETE $run"; continue }
        if (Test-Path $directory) { throw "Refusing to overwrite incomplete Phase 1C run: $directory" }
        Write-Host "START $run"
        & $python $trainer --config $config --variant $variant --seed $seed --run-name $run --output-root $OutputRoot
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $run with exit code $LASTEXITCODE" }
        if (-not ((Test-Path (Join-Path $directory "best.pt")) -and (Test-Path (Join-Path $directory "last.pt")) -and (Test-Path (Join-Path $directory "run_metadata.json")))) { throw "Required Phase 1C artifacts missing for $run" }
        Write-Host "COMPLETE $run"
    }
}
Write-Host "ALL_COMPLETE"
