# Learning-free QW-QPSO (branch feat/prior-qw): the quantum-well swarm is centred on a closed-form
# physics prior (common/prior.py) instead of a GNN prediction. No labels, no training.
#
#   1. cases for Set A (paper layouts, K <= 6) and Set B (K_max = 10) -- deterministic, same as 4090
#   2. validate prior_qwqpso grid (s x L x beta = 30 cells) on Set B validation, K_val = 2..6
#   3. test Set B (K = 2..6, 8, 10) and Set A (K = 2..6): prior_qwqpso + prior_direct
#      (Set A re-uses the configuration selected on Set B validation)
#
#   powershell -ExecutionPolicy Bypass -File run_prior.ps1 [-Python <exe>] [-Workers 12]
# Status: logs\prior_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 12
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\prior_status.txt"

function Write-Status([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line -Encoding ascii
    Write-Host $line
}

function Invoke-Step([string]$name, [string[]]$cmd) {
    $t0 = Get-Date
    Write-Status "START $name :: python $($cmd -join ' ')"
    & $Python @cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Status "FAILED $name (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    $dt = (Get-Date) - $t0
    Write-Status ("DONE {0} in {1:hh\:mm\:ss}" -f $name, $dt)
}

$BLA = "prior_blA"
$BLB = "prior_blB"
$OA = "prior_A_01"
$OB = "prior_B_01"
$kA = @("--k-test", "2,3,4,5,6")
$kB = @("--k-test", "2,3,4,5,6,8,10")
$run = @("--preset", "paper", "--budget-short", "150", "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== prior run start (python=$Python workers=$Workers) ==="

Invoke-Step "1/5 cases Set A" (@("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BLA) + $kA + $run)
Invoke-Step "2/5 cases Set B" (@("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BLB) + $kB + $run)

Invoke-Step "3/5 validate prior_qwqpso grid (Set B, K_val 2..6)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "prior_qwqpso", "--k-val", "2,3,4,5,6", "--budgets-val", "150",
    "--baseline-dir", $BLB, "--out-dir", $OB) + $kB + $run)

Invoke-Step "4/5 test Set B (K 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "prior_qwqpso,prior_direct", "--k-val", "2,3,4,5,6",
    "--budgets-val", "150", "--baseline-dir", $BLB, "--out-dir", $OB) + $kB + $run)

New-Item -ItemType Directory -Force -Path "result\$OA\validation" | Out-Null
Copy-Item -Force "result\$OB\validation\selected_prior.json" "result\$OA\validation\selected_prior.json"
Invoke-Step "5/5 test Set A (K 2..6, config from Set B validation)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "prior_qwqpso,prior_direct", "--k-val", "2,3,4,5,6",
    "--budgets-val", "150", "--baseline-dir", $BLA, "--out-dir", $OA) + $kA + $run)

$zip = "prior_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\$OB\validation\prior_grid.csv", "result\$OB\validation\selected_prior.json", `
    "result\$OB\tables", "result\$OA\tables", "result\$OB\test\trials.csv", "result\$OA\test\trials.csv", `
    "result\$OB\run.log", "result\$OA\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
