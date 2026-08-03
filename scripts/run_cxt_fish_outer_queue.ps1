param(
    [int]$Seed = 3407
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logRoot = Join-Path $root 'logs\final_outer'
New-Item -ItemType Directory -Force $logRoot | Out-Null

$queue = @(
    @{ Fold = '1'; Method = 'F1' },
    @{ Fold = '2'; Method = 'F0' },
    @{ Fold = '2'; Method = 'F1' },
    @{ Fold = '3'; Method = 'F0' },
    @{ Fold = '3'; Method = 'F1' }
)

foreach ($item in $queue) {
    $name = "fold$($item.Fold)_$($item.Method)_seed$Seed"
    $output = Join-Path $root "outputs\cxt_fish\final_outer\fold_$($item.Fold)\$($item.Method)_seed$Seed"
    $log = Join-Path $logRoot "$name.queue.log"
    $err = Join-Path $logRoot "$name.queue.err.log"
    if (Test-Path (Join-Path $output 'run_metadata.json')) {
        Write-Host "[$name] already complete; skipping."
        continue
    }
    if (Test-Path $output) {
        $items = Get-ChildItem $output -Force
        if ($items.Count -gt 0) { throw "[$name] output exists but is incomplete; refusing overwrite." }
    }
    Write-Host "[$name] starting"
    & $python (Join-Path $root 'scripts\train_cxt_fish_outer.py') --fold $item.Fold --method $item.Method --seed $Seed 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "[$name] failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path (Join-Path $output 'run_metadata.json'))) { throw "[$name] ended without run_metadata.json" }
    Write-Host "[$name] complete"
}

Write-Host 'Five-cell outer queue complete.'
