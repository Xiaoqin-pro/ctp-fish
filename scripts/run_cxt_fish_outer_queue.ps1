param(
    [int]$Seed = 3407,
    [ValidateSet('first', 'second', 'all')]
    [string]$Batch = 'all'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logRoot = Join-Path $root 'logs\final_outer'
New-Item -ItemType Directory -Force $logRoot | Out-Null

$queue = if ($Batch -eq 'all') {
    @(
        # Fixed order: all three registered seeds, six cells per seed.
        @{ Fold = '1'; Method = 'F0'; Seed = 3407 },
        @{ Fold = '1'; Method = 'F1'; Seed = 3407 },
        @{ Fold = '2'; Method = 'F0'; Seed = 3407 },
        @{ Fold = '2'; Method = 'F1'; Seed = 3407 },
        @{ Fold = '3'; Method = 'F0'; Seed = 3407 },
        @{ Fold = '3'; Method = 'F1'; Seed = 3407 },
        @{ Fold = '1'; Method = 'F0'; Seed = 2026 },
        @{ Fold = '1'; Method = 'F1'; Seed = 2026 },
        @{ Fold = '2'; Method = 'F0'; Seed = 2026 },
        @{ Fold = '2'; Method = 'F1'; Seed = 2026 },
        @{ Fold = '3'; Method = 'F0'; Seed = 2026 },
        @{ Fold = '3'; Method = 'F1'; Seed = 2026 },
        @{ Fold = '1'; Method = 'F0'; Seed = 17 },
        @{ Fold = '1'; Method = 'F1'; Seed = 17 },
        @{ Fold = '2'; Method = 'F0'; Seed = 17 },
        @{ Fold = '2'; Method = 'F1'; Seed = 17 },
        @{ Fold = '3'; Method = 'F0'; Seed = 17 },
        @{ Fold = '3'; Method = 'F1'; Seed = 17 }
    )
} elseif ($Batch -eq 'first') {
    @(
        @{ Fold = '1'; Method = 'F1'; Seed = $Seed },
        @{ Fold = '2'; Method = 'F0'; Seed = $Seed },
        @{ Fold = '2'; Method = 'F1'; Seed = $Seed },
        @{ Fold = '3'; Method = 'F0'; Seed = $Seed },
        @{ Fold = '3'; Method = 'F1'; Seed = $Seed }
    )
} else {
    @(
        @{ Fold = '1'; Method = 'F0'; Seed = $Seed },
        @{ Fold = '1'; Method = 'F1'; Seed = $Seed },
        @{ Fold = '2'; Method = 'F0'; Seed = $Seed },
        @{ Fold = '2'; Method = 'F1'; Seed = $Seed },
        @{ Fold = '3'; Method = 'F0'; Seed = $Seed },
        @{ Fold = '3'; Method = 'F1'; Seed = $Seed }
    )
}

foreach ($item in $queue) {
    $itemSeed = [int]$item.Seed
    $name = "fold$($item.Fold)_$($item.Method)_seed$itemSeed"
    $output = Join-Path $root "outputs\cxt_fish\final_outer\fold_$($item.Fold)\$($item.Method)_seed$itemSeed"
    $log = Join-Path $logRoot "$name.queue.log"
    if (Test-Path (Join-Path $output 'run_metadata.json')) {
        Write-Host "[$name] already complete; skipping."
        continue
    }
    $resume = $null
    if (Test-Path $output) {
        $items = Get-ChildItem $output -Force
        $last = Join-Path $output 'last.pt'
        if (Test-Path $last) {
            $resume = $last
            Write-Host "[$name] resuming from last.pt"
        } elseif ($items.Count -gt 0) {
            throw "[$name] output exists but has no resumable last.pt; refusing overwrite."
        }
    }
    Write-Host "[$name] starting"
    $args = @(
        (Join-Path $root 'scripts\train_cxt_fish_outer.py'),
        '--fold', $item.Fold,
        '--method', $item.Method,
        '--seed', $itemSeed
    )
    if ($resume) { $args += @('--resume', $resume) }
    & $python @args 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "[$name] failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path (Join-Path $output 'run_metadata.json'))) { throw "[$name] ended without run_metadata.json" }
    Write-Host "[$name] complete"
}

Write-Host "$Batch outer queue complete."
