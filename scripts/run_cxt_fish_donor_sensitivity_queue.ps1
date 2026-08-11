param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [switch]$DryRun
)
$seeds = @(4101, 4102, 4103, 4104, 4105)
$modelSeeds = @(3407, 2026, 17)
foreach ($donorSeed in $seeds) {
  foreach ($fold in @(1, 2, 3)) {
    foreach ($method in @("F0", "F1")) {
      foreach ($modelSeed in $modelSeeds) {
        $args = @("scripts\evaluate_cxt_fish_donor_sensitivity.py", "--donor-seed", $donorSeed, "--fold", $fold, "--method", $method, "--model-seed", $modelSeed)
        if ($DryRun) { Write-Host ($Python + " " + ($args -join " ")) }
        else {
          & $Python @args
          if ($LASTEXITCODE -ne 0) { throw "Donor sensitivity failed: seed=$donorSeed fold=$fold method=$method model_seed=$modelSeed" }
        }
      }
    }
  }
}
