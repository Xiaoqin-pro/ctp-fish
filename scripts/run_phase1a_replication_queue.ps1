param(
    [string]$ProjectRoot = "D:\111\Desktop\ctp-fish",
    [string]$OutputRoot = "outputs\cxt_fish\phase1a_replication"
)

$ErrorActionPreference = "Stop"

# Frozen after validation-only screening: A1, A4-I, A2; each uses the three
# configured replication seeds. No internal test or outer folds are passed.
$jobs = @(
    @{ Name = "A1_s1_ce"; Sampler = "s1_track_uniform"; Loss = "ce"; Prior = $null },
    @{ Name = "A4I_s0_balanced_image"; Sampler = "s0_frame_uniform"; Loss = "balanced_softmax"; Prior = "image" },
    @{ Name = "A2_s2_ce"; Sampler = "s2_class_uniform"; Loss = "ce"; Prior = $null }
)
$seeds = @(3407, 2026, 17)
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $ProjectRoot "scripts\train_gate0_baseline.py"
$config = Join-Path $ProjectRoot "configs\resnet18_gate0.yaml"

foreach ($job in $jobs) {
    foreach ($seed in $seeds) {
        $run = "$($job.Name)_seed$seed"
        $directory = Join-Path $ProjectRoot "$OutputRoot\$run"
        $complete = (Test-Path (Join-Path $directory "best.pt")) -and (Test-Path (Join-Path $directory "last.pt")) -and (Test-Path (Join-Path $directory "training_curve.csv"))
        if ($complete) {
            Write-Host "SKIP_COMPLETE $run"
            continue
        }
        if (Test-Path $directory) {
            throw "Refusing to overwrite incomplete replication run: $directory"
        }
        $arguments = @(
            $trainer, "--config", $config, "--split", "track", "--seed", "$seed",
            "--phase1a-sampler", $job.Sampler, "--classification-loss", $job.Loss,
            "--run-name", $run, "--output-root", $OutputRoot
        )
        if ($null -ne $job.Prior) {
            $arguments += @("--prior", $job.Prior)
        }
        Write-Host "START $run"
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $run with exit code $LASTEXITCODE"
        }
        if (-not ((Test-Path (Join-Path $directory "best.pt")) -and (Test-Path (Join-Path $directory "last.pt")))) {
            throw "Required checkpoints missing for $run"
        }
        Write-Host "COMPLETE $run"
    }
}

Write-Host "ALL_COMPLETE"
