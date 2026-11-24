# Control experiment "does the quantum-well initialisation (W) need QPSO?" (branch feat/pso-well).
# Re-uses bl_kscale_01 (cases, GNN models, budget) so every trial is paired with qw_kscale_01.
# Test only K = 2..6 (--k-subset); networks train on K = 2..5, K = 6 unseen.
#
#   gnn_pso_wp : PSO (Algorithm 1 inertia) + W + P copied from the selected gnn_qwqpso config
#                -> same initial swarm as gnn_qwqpso, only the update rule differs.
#   gnn_pso_wpt: PSO + W + P with W x inertia x P tuned on validation (42 cells, same size and
#                same selection rule as the gnn_qwqpso grid) -> balanced tuning effort.
#   gnn_qwqpso, gnn_pso_p are re-run in this folder (determinism check vs qw_kscale_01).
#
#   powershell -ExecutionPolicy Bypass -File run_pw_paper.ps1 [-Python <python.exe>] [-Workers 19]
#
# Status: logs\pw_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 19
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\pw_status.txt"

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
$PW = "pw_kscale_01"
$CMP = "cmp_pw_01"
$common = @("--preset", "paper", "--k-test", "2,3,4,5,6,8,10", "--k-val", "2,3,4,5,6",
            "--budgets-val", "150", "--workers", "$Workers", "--quiet")
$qw = @("--qw-scale", "0.05", "--qw-frac", "1.0", "--qw-beta-start", "0.5", "--qw-beta-end", "0.25",
        "--qw-polish", "1")
$qq = @("--qq-mbest", "mean", "--qq-beta", "linear", "--qq-polish", "2")

Write-Status "=== pw run start (python=$Python workers=$Workers) ==="

Invoke-Step "1/3 validate gnn_pso_wpt grid (42 cells, K_val 2..6)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "gnn_pso_wpt",
    "--baseline-dir", $BL, "--out-dir", $PW) + $common)

Invoke-Step "2/3 test,tables K 2..6 (gnn_qwqpso, gnn_pso_p, gnn_pso_wp, gnn_pso_wpt)" (@(
    "Quantummodel_code/run_quantum.py", "--stage", "test,tables",
    "--methods", "gnn_qwqpso,gnn_pso_p,gnn_pso_wp,gnn_pso_wpt", "--k-subset", "2,3,4,5,6",
    "--baseline-dir", $BL, "--out-dir", $PW) + $common + $qw + $qq)

Invoke-Step "3/3 compare report" @("make_report.py", "--baseline-dir", $BL, "--quantum-dir", $PW,
    "--out-dir", $CMP, "--pairs",
    "gnn_qwqpso:gnn_pso_wp,gnn_qwqpso:gnn_pso_wpt,gnn_pso_wp:gnn_pso_p,gnn_pso_wpt:gnn_pso_p,gnn_qwqpso:gnn_pso_p,gnn_pso_wpt:gnn_pso")

$zip = "pw_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\$CMP\tables", `
    "result\$PW\validation\pw_grid.csv", `
    "result\$PW\validation\selected_pw.json", `
    "result\$PW\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
