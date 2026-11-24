# Fair PSO counterparts of Prior + QW-QPSO (branch feat/prior-dep), Set B layouts (prior_blB cases).
# Same physics prior, evaluator, ranking, budget and validation as prior_qwqpso (run prior_B_01):
#
#   1. validate prior_pso grid on Set B validation, K_val = 2..6 (as prior_qwqpso):
#        s (5) x well L (3) x PSO inertia (2) = 30 cells, P = 1  (QPSO beta <-> PSO inertia)
#   2. test K = 2..6, 8, 10 -> ppso_B_01: prior_pso, prior_qwqpso (config of prior_B_01), prior_direct
#   3. test K = 2..6, 8, 10 -> ppso_full_B_01: prior_fullpso (Full PSO 1800 + prior in particle 0)
#   4. zip: grid, selected configs, tables, test trials.csv
#
#   powershell -ExecutionPolicy Bypass -File run_ppso.ps1 [-Python <exe>] [-Workers 18] [-Baseline prior_blB]
# Status: logs\ppso_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 18,
    [string]$Baseline = "prior_blB"
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\ppso_status.txt"

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

$BL = $Baseline
$OUT = "ppso_B_01"
$OUTF = "ppso_full_B_01"
$common = @("--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10",
            "--k-val", "2,3,4,5,6", "--budgets-val", "150", "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== prior-PSO run start (python=$Python workers=$Workers baseline=$BL) ==="

if (-not (Test-Path "result\$BL\cases\cases_val.csv")) {
    Invoke-Step "0/3 cases Set B" @("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BL,
        "--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10", "--workers", "$Workers",
        "--quiet", "--no-plots")
}

Invoke-Step "1/3 validate prior_pso grid (Set B val, K_val 2..6)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "prior_pso", "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

# prior_qwqpso: configuration selected on the same Set B validation in run prior_B_01 (3060)
$selp = "result\$OUT\validation\selected_prior.json"
if (-not (Test-Path $selp)) {
    $json = '{"prior_s": 0.2, "well_scale": 0.1, "well_frac": 1.0, "beta_start": 0.5, "beta_end": 0.25, "polish_pops": 1, "source": "prior_B_01 validation (3060)"}'
    Set-Content -Path $selp -Value $json -Encoding ascii
}

Invoke-Step "2/3 test prior_pso, prior_qwqpso, prior_direct (K 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "prior_pso,prior_qwqpso,prior_direct",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

New-Item -ItemType Directory -Force -Path "result\$OUTF\validation" | Out-Null
Copy-Item -Force "result\$OUT\validation\selected_ppso.json" "result\$OUTF\validation\selected_ppso.json"
Invoke-Step "3/3 test prior_fullpso (Full PSO 1800 + prior, K 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "prior_fullpso",
    "--baseline-dir", $BL, "--out-dir", $OUTF) + $common)

$zip = "ppso_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\$OUT\validation\ppso_grid.csv", "result\$OUT\validation\selected_ppso.json", `
    "result\$OUT\tables", "result\$OUT\test\trials.csv", "result\$OUT\run.log", `
    "result\$OUTF\test\trials.csv", "result\$OUTF\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
