# Run the whole QW-QPSO paper pipeline unattended (branch feat/qpso-qwell).
#   powershell -ExecutionPolicy Bypass -File run_qw_paper.ps1
# Steps: validate (42-config grid) -> test (+ ablations, gnn_qqpso, gnn_pso_p) -> report -> zip.
# Stops at the first failing step. Re-running resumes finished (layout, K) groups.
# Uses the baseline run bl_paper_01 of THIS machine (baseline is not re-run).

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Invoke-Step([string]$name, [string[]]$cmd) {
    $t0 = Get-Date
    Write-Host ""
    Write-Host "==== [$((Get-Date).ToString('HH:mm:ss'))] $name ====" -ForegroundColor Cyan
    Write-Host ("python " + ($cmd -join " "))
    & python @cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==== FAILED: $name (exit $LASTEXITCODE) -> send the error above ====" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $dt = (Get-Date) - $t0
    Write-Host ("==== done: {0} in {1:hh\:mm\:ss} ====" -f $name, $dt) -ForegroundColor Green
}

$common = @("--preset", "paper", "--baseline-dir", "bl_paper_01", "--out-dir", "qw_paper_01",
            "--k-val", "2,3,4,5,6", "--budgets-val", "150")

Invoke-Step "1/3 validate (gnn_qwqpso grid)" (@("Quantummodel_code/run_quantum.py", "--stage", "validate",
    "--methods", "gnn_qwqpso") + $common)

Invoke-Step "2/3 test + tables" (@("Quantummodel_code/run_quantum.py", "--stage", "test,tables",
    "--methods", "gnn_qwqpso,gnn_qwqpso_noW,gnn_qwqpso_noB,gnn_qqpso,gnn_pso_p",
    "--qq-mbest", "mean", "--qq-beta", "linear", "--qq-polish", "2") + $common)

Invoke-Step "3/3 compare report" @("make_report.py", "--baseline-dir", "bl_paper_01",
    "--quantum-dir", "qw_paper_01", "--out-dir", "cmp_qw_paper_01")

$zip = "qw_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\cmp_qw_paper_01\tables", `
    "result\qw_paper_01\validation\qw_grid.csv", `
    "result\qw_paper_01\validation\selected_qw.json", `
    "result\qw_paper_01\run.log"
Write-Host ""
Write-Host "==== ALL DONE. Send $((Resolve-Path $zip).Path) ====" -ForegroundColor Green
