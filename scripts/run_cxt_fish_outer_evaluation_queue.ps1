param(
    [string]$OutputRoot = 'outputs/cxt_fish/final_outer_evaluation'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logRoot = Join-Path $root 'logs\final_outer_evaluation'
New-Item -ItemType Directory -Force $logRoot | Out-Null
$queue = foreach($seed in 3407,2026,17){ foreach($fold in 1,2,3){ foreach($method in 'F0','F1'){
    [pscustomobject]@{ Seed=$seed; Fold=[string]$fold; Method=$method }
}}}
foreach($item in $queue){
    $name = "fold$($item.Fold)_$($item.Method)_seed$($item.Seed)"
    $log = Join-Path $logRoot "$name.log"
    Write-Host "[$name] starting"
    & $python (Join-Path $root 'scripts\evaluate_cxt_fish_outer.py') `
        --fold $item.Fold --method $item.Method --seed $item.Seed --output-root $OutputRoot `
        2>&1 | Tee-Object -FilePath $log -Append
    if($LASTEXITCODE -ne 0){ throw "[$name] evaluation failed with exit code $LASTEXITCODE" }
    Write-Host "[$name] complete"
}
Write-Host 'outer evaluation queue complete.'
