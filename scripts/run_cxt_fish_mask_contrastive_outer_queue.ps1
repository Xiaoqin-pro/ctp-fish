param(
    [ValidateSet('train', 'evaluate')]
    [string]$Mode = 'train'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs\mask_contrastive_outer_v1'; New-Item -ItemType Directory -Force $logDir | Out-Null
foreach ($fold in @(1, 2, 3)) {
    foreach ($seed in @(3407, 2026, 17)) {
        if ($Mode -eq 'train') {
            $script = Join-Path $root 'scripts\train_cxt_fish_mask_contrastive_outer.py'
            $args = @('--config', 'configs/cxt_fish_mask_contrastive_outer_v1.yaml', '--fold', "$fold", '--seed', "$seed")
            $cell = Join-Path $root "outputs\cxt_fish\mask_contrastive_outer_v1\fold_${fold}\seed${seed}"
            if ((Test-Path (Join-Path $cell 'run_metadata.json')) -and (Test-Path (Join-Path $cell 'best.pt')) -and (Test-Path (Join-Path $cell 'last.pt'))) {
                Write-Host "[train] fold_${fold}_seed${seed} already complete; skipping"
                continue
            }
            $resume = $null
            if (Test-Path (Join-Path $cell 'last.pt')) { $resume = Join-Path $cell 'last.pt' }
            elseif (Test-Path (Join-Path $cell 'pretrain_last.pt')) { $resume = Join-Path $cell 'pretrain_last.pt' }
            if ($resume) { $args += @('--resume', $resume); Write-Host "[train] resuming $resume" }
        } else {
            $script = Join-Path $root 'scripts\evaluate_cxt_fish_mask_contrastive_outer.py'
            $args = @('--config', 'configs/cxt_fish_mask_contrastive_outer_v1.yaml', '--fold', "$fold", '--seed', "$seed", '--output-root', 'outputs/cxt_fish/mask_contrastive_outer_v1_evaluation', '--unlock-outer-test')
        }
        $stamp = "fold_${fold}_seed${seed}"; $log = Join-Path $logDir "${Mode}_${stamp}.log"
        Write-Host "[$Mode] $stamp"; & $python $script @args 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "route-C $Mode failed for $stamp; see $log" }
    }
}
Write-Host "route-C $Mode queue complete."
