param([string]$Python = ".\.venv\Scripts\python.exe", [switch]$DryRun)
foreach ($fold in @(1, 2, 3)) {
  foreach ($seed in @(3407, 2026, 17)) {
    $args = @("scripts\evaluate_cxt_fish_rgb2_outer.py", "--fold", $fold, "--seed", $seed)
    if ($DryRun) { Write-Host ($Python + " " + ($args -join " ")) }
    else { & $Python @args; if ($LASTEXITCODE -ne 0) { throw "RGB2 evaluation failed: fold=$fold seed=$seed" } }
  }
}
