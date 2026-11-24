# Pure-swarm QW-QPSO from a physics prior, NO boundary polish (branch feat/prior-dep).
# Question: can the QPSO swarm itself optimise from the prior (all 150 evaluations = swarm)?
# Set B layouts (prior_blB = same cases as bl_kscale_01), learning-free.
#
#   1. validate (--prior-polish 0), Set B validation, K_val = 2,3,4,5,6,8,10:
#        prior_qwqpso (K-only prior, m_b on the QoS bound)  24 cells: s x L x beta
#        prior_dqpso  (DEP-aware prior + box z_b <= cap)    24 cells
#   2. test K = 2..6, 8, 10: prior_qwqpso, prior_dqpso, prior_dqpso_noW, prior_direct, prior_ddirect
#   3. zip: grids, selected configs, tables, test trials.csv (for pairing with gnn_pso / full_pso
#      of bl_kscale_01 on the code machine)
#
#   powershell -ExecutionPolicy Bypass -File run_nopolish.ps1 [-Python <exe>] [-Workers 12] [-Baseline prior_blB]
# Status: logs\nopol_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 12,
    [string]$Baseline = "prior_blB"
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\nopol_status.txt"

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
$OUT = "nopol_B_01"
$common = @("--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10",
            "--k-val", "2,3,4,5,6,8,10", "--budgets-val", "150", "--prior-polish", "0",
            "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== no-polish run start (python=$Python workers=$Workers baseline=$BL) ==="

if (-not (Test-Path "result\$BL\cases\cases_val.csv")) {
    Invoke-Step "0/2 cases Set B" @("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BL,
        "--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10", "--workers", "$Workers",
        "--quiet", "--no-plots")
}

Invoke-Step "1/2 validate prior_qwqpso + prior_dqpso grids, P=0 (K_val 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "prior_qwqpso,prior_dqpso",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

Invoke-Step "2/2 test (K 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "prior_qwqpso,prior_dqpso,prior_dqpso_noW,prior_direct,prior_ddirect",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

$zip = "nopol_result.zip"
$vd = "result\$OUT\validation"
Compress-Archive -Force -DestinationPath $zip -Path `
    "$vd\prior_grid.csv", "$vd\dprior_grid.csv", "$vd\selected_prior.json", "$vd\selected_dprior.json", `
    "result\$OUT\tables", "result\$OUT\test\trials.csv", "result\$OUT\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
