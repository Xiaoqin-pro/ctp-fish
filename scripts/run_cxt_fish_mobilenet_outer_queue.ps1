param(
    [ValidateSet('train', 'evaluate')]
    [string]$Mode = 'train'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs\mobilenet_outer'; New-Item -ItemType Directory -Force $logDir | Out-Null
foreach ($fold in @(1, 2, 3)) {
    foreach ($method in @('MV0', 'MV1')) {
        $stamp = "fold_${fold}_${method}_seed3407"
        if ($Mode -eq 'train') {
            $script = Join-Path $root 'scripts\train_cxt_fish_mobilenet_outer.py'
            $args = @('--config', 'configs/cxt_fish_mobilenet_outer_v1.yaml', '--fold', "$fold", '--method', $method, '--seed', '3407')
            $cell = Join-Path $root "outputs\cxt_fish\mobilenet_outer\fold_${fold}\${method}_seed3407"
            if ((Test-Path (Join-Path $cell 'run_metadata.json')) -and (Test-Path (Join-Path $cell 'best.pt'))) { Write-Host "[train] $stamp already complete; skipping"; continue }
            if (Test-Path (Join-Path $cell 'last.pt')) { $args += @('--resume', (Join-Path $cell 'last.pt')); Write-Host "[train] resuming $stamp" }
        } else {
            $script = Join-Path $root 'scripts\evaluate_cxt_fish_mobilenet_outer.py'
            $args = @('--config', 'configs/cxt_fish_mobilenet_outer_v1.yaml', '--fold', "$fold", '--method', $method, '--seed', '3407', '--output-root', 'outputs/cxt_fish/mobilenet_outer_evaluation', '--unlock-outer-test')
        }
        $log = Join-Path $logDir "${Mode}_${stamp}.log"; Write-Host "[$Mode] $stamp"
        & $python $script @args 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "MobileNetV3 $Mode failed for $stamp; see $log" }
    }
}
Write-Host "MobileNetV3 $Mode queue complete."
