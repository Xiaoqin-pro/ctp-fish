param(
    [ValidateSet('train', 'evaluate')]
    [string]$Mode = 'train'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$seeds = @(3407, 2026, 17)
$logDir = Join-Path $root 'logs\clib_style_outer'
New-Item -ItemType Directory -Force $logDir | Out-Null

foreach ($fold in @(1, 2, 3)) {
    foreach ($seed in $seeds) {
        $stamp = "fold_${fold}_seed${seed}"
        if ($Mode -eq 'train') {
            $script = Join-Path $root 'scripts\train_cxt_fish_clib_style_outer.py'
            $args = @('--config', 'configs/cxt_fish_clib_style_outer_v1.yaml', '--fold', "$fold", '--seed', "$seed")
        } else {
            $script = Join-Path $root 'scripts\evaluate_cxt_fish_clib_style_outer.py'
            $args = @('--config', 'configs/cxt_fish_clib_style_outer_v1.yaml', '--fold', "$fold", '--seed', "$seed")
        }
        $log = Join-Path $logDir "${Mode}_${stamp}.log"
        Write-Host "[$Mode] $stamp"
        & $python $script @args 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            throw "CLIB-style $Mode failed for $stamp; see $log"
        }
    }
}
Write-Host "CLIB-style $Mode queue complete."
