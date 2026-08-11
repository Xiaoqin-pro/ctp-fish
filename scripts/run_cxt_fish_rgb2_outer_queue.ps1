param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [switch]$DryRun
)
$seeds = @(3407, 2026, 17)
foreach ($fold in @(1, 2, 3)) {
  foreach ($seed in $seeds) {
    $args = @("scripts\train_cxt_fish_rgb2_outer.py", "--fold", $fold, "--seed", $seed)
    if ($DryRun) { Write-Host ($Python + " " + ($args -join " ")) }
    else {
      & $Python @args
      if ($LASTEXITCODE -ne 0) { throw "F0-2RGB training failed: fold=$fold seed=$seed" }
    }
  }
}
