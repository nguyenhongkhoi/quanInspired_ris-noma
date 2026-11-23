# K-scaling run (branch feat/k-scale): full pipeline FROM SCRATCH with test K = 2,3,4,5,6,8,10.
# Networks still train on K = 2..5 only; K = 6, 8, 10 are unseen. QW-QPSO search parameters are
# re-selected on validation with K_val = 2..6 (K = 8, 10 never seen by any tuning step).
# Methods on test: full_pso, gnn_pso (baseline) + gnn_qwqpso, gnn_pso_p (quantum run).
#
#   powershell -ExecutionPolicy Bypass -File run_kscale_paper.ps1 [-Python <python.exe>] [-Workers 19]
#
# Stops at the first failing step (status file logs\kscale_status.txt). Re-running resumes.
# K_max = 10 changes the public-user positions of every layout, so this run is NOT paired with
# bl_paper_01 / qw_paper_01; all four methods are re-run here on the new layouts.

param(
    [string]$Python = "",
    [int]$Workers = 19
)

$ErrorActionPreference = "Continue"   # failures are detected via $LASTEXITCODE below
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\kscale_status.txt"

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

$BL = "bl_kscale_01"
$QW = "qw_kscale_01"
$CMP = "cmp_kscale_01"
$common = @("--preset", "paper", "--k-test", "2,3,4,5,6,8,10", "--workers", "$Workers", "--quiet")

Write-Status "=== kscale run start (python=$Python workers=$Workers) ==="

Invoke-Step "1/6 baseline cases,labels,train,validate" (@("baseline_code/run_baseline.py",
    "--stage", "cases,labels,train,validate", "--out-dir", $BL) + $common)

Invoke-Step "2/6 baseline test,tables (full_pso, gnn_pso)" (@("baseline_code/run_baseline.py",
    "--stage", "test,tables", "--methods", "full_pso,gnn_pso", "--out-dir", $BL) + $common)

Invoke-Step "3/6 quantum validate (gnn_qwqpso grid, K_val 2..6)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "gnn_qwqpso", "--k-val", "2,3,4,5,6", "--budgets-val", "150",
    "--baseline-dir", $BL, "--out-dir", $QW) + $common)

Invoke-Step "4/6 quantum test,tables (gnn_qwqpso, gnn_pso_p)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "gnn_qwqpso,gnn_pso_p",
    "--qq-mbest", "mean", "--qq-beta", "linear", "--qq-polish", "2",
    "--k-val", "2,3,4,5,6", "--budgets-val", "150",
    "--baseline-dir", $BL, "--out-dir", $QW) + $common)

Invoke-Step "5/6 compare report" @("make_report.py", "--baseline-dir", $BL, "--quantum-dir", $QW,
    "--out-dir", $CMP, "--pairs", "gnn_qwqpso:gnn_pso,gnn_qwqpso:gnn_pso_p,gnn_pso_p:gnn_pso,gnn_qwqpso:full_pso")

Invoke-Step "6/6 comparison figure" @("plot_qw_compare.py", "--baseline-dir", $BL, "--quantum-dir", $QW,
    "--out-dir", $CMP, "--k-train", "2,3,4,5")

$zip = "kscale_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\$CMP\tables", `
    "result\$CMP\figures\fig_qw_compare.png", `
    "result\$CMP\figures\fig_qw_compare.pdf", `
    "result\$QW\validation\qw_grid.csv", `
    "result\$QW\validation\selected_qw.json", `
    "result\$BL\validation\selected_budget.json", `
    "result\$QW\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
