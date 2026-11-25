# Fair-prior run (branch feat/prior-fair): every method obeys the SAME search targets.
# Why: dep_prior sits on the QoS bound built from the rate CEILINGS c_j; actual public rates are
# below c_j, so at K >= 6 the prior is search-infeasible (R_j < R_qos + 0.01) and only passes the
# looser assessment target. fair_prior bisects m_b under the search targets (<= 14 evaluations,
# counted); the swarm methods get B - 30 = 120 evaluations so n_evals <= 150.
#
#   test K = 2..6, 8, 10 (Set B, prior_blB):
#     prior_fdirect  fair prior only
#     prior_fqpso    fair prior + boxed QW-QPSO (dprior config selected on validation, P = 0)
#     prior_fpso     fair prior + boxed PSO, same W / P copied, Algorithm 1 inertia
#     prior_ddirect, prior_dqpso   old (unfair) prior, for pairing in the same file
#     full_pso       Full PSO 1800 (reference, same seeds as bl_kscale_01)
#   No new validation: the swarm config is the one nopol_B_01 selected on validation
#   (validation\selected_dprior.json, copied below). Test is never used for selection.
#
#   powershell -ExecutionPolicy Bypass -File run_fair.ps1 [-Python <exe>] [-Workers 12] [-Baseline prior_blB] [-ConfigFrom nopol_B_01]
# Status: logs\fair_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 12,
    [string]$Baseline = "prior_blB",
    [string]$ConfigFrom = "nopol_B_01"
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\fair_status.txt"

function Write-Status([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line -Encoding ascii
    Write-Host $line
}

function Invoke-Step([string]$name, [string[]]$cmd, [bool]$fatal = $true) {
    $t0 = Get-Date
    Write-Status "START $name :: python $($cmd -join ' ')"
    $log = Join-Path $PSScriptRoot ("logs\fair_" + ($name -replace '[^A-Za-z0-9]', '_') + ".txt")
    & $Python @cmd *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Status "FAILED $name (exit $LASTEXITCODE), output: $log"
        if ($fatal) { exit $LASTEXITCODE }
        return
    }
    $dt = (Get-Date) - $t0
    Write-Status ("DONE {0} in {1:hh\:mm\:ss}" -f $name, $dt)
}

$BL = $Baseline
$OUT = "fair_B_01"
$common = @("--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10",
            "--k-val", "2,3,4,5,6,8,10", "--budgets-val", "150", "--prior-polish", "0",
            "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== fair-prior run start (python=$Python workers=$Workers baseline=$BL config=$ConfigFrom) ==="

$src = "result\$ConfigFrom\validation\selected_dprior.json"
if (-not (Test-Path $src)) { Write-Status "FAILED: missing $src"; exit 1 }
New-Item -ItemType Directory -Force -Path "result\$OUT\validation" | Out-Null
Copy-Item -Force $src "result\$OUT\validation\selected_dprior.json"
Write-Status ("config: " + ((Get-Content $src) -join ' '))

Invoke-Step "1/2 test" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test", "--methods", "prior_fdirect,prior_fqpso,prior_fpso,prior_ddirect,prior_dqpso,full_pso",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

# tables are not fatal: trials.csv is the result (4060 Ti runs failed at this stage on 2026-11-25)
Invoke-Step "2/2 tables" (@("Quantummodel_code/run_quantum.py",
    "--stage", "tables", "--methods", "prior_fdirect,prior_fqpso,prior_fpso,prior_ddirect,prior_dqpso,full_pso",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common) $false

$zip = "fair_result.zip"
$paths = @("result\$OUT\validation\selected_dprior.json", "result\$OUT\test\trials.csv", "result\$OUT\run.log")
if (Test-Path "result\$OUT\tables") { $paths += "result\$OUT\tables" }
Compress-Archive -Force -DestinationPath $zip -Path $paths
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
