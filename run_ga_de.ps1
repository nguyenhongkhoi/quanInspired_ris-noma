# GA / DE comparison against GNN + QW-QPSO (branch feat/ga-de), Set B layouts, test K = 2..6.
# Re-uses bl_kscale_01 (cases, GNN models, selected budget). Two parts, one per machine:
#
#   -Part heavy : full_ga, full_de (B = 1800, uniform init) + gnn_qwqpso   -> result\ga_full_01
#                 + make_report against full_pso / gnn_pso of bl_kscale_01 -> result\cmp_ga_full_01
#   -Part light : gnn_ga, gnn_de (B = 150, GNN init) + gnn_qwqpso + gnn_pso -> result\ga_short_01
#
# gnn_qwqpso is re-run in each part so every comparison is made on ONE machine.
#   powershell -ExecutionPolicy Bypass -File run_ga_de.ps1 -Part heavy [-Python <exe>] [-Workers 19]
# Status: logs\ga_<part>_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [Parameter(Mandatory = $true)][ValidateSet("heavy", "light")][string]$Part,
    [string]$Python = "",
    [int]$Workers = 19
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\ga_$($Part)_status.txt"

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
$common = @("--preset", "paper", "--k-test", "2,3,4,5,6,8,10", "--k-val", "2,3,4,5,6",
            "--budgets-val", "150", "--k-subset", "2,3,4,5,6", "--workers", "$Workers", "--quiet", "--no-plots")
$qw = @("--qw-scale", "0.05", "--qw-frac", "1.0", "--qw-beta-start", "0.5", "--qw-beta-end", "0.25",
        "--qw-polish", "1")

Write-Status "=== ga/de $Part start (python=$Python workers=$Workers) ==="

if ($Part -eq "heavy") {
    $OUT = "ga_full_01"
    Invoke-Step "1/2 test full_ga, full_de, gnn_qwqpso (K 2..6)" (@("Quantummodel_code/run_quantum.py",
        "--stage", "test,tables", "--methods", "gnn_qwqpso,full_ga,full_de",
        "--baseline-dir", $BL, "--out-dir", $OUT) + $common + $qw)
    Invoke-Step "2/2 compare with full_pso / gnn_pso" @("make_report.py", "--baseline-dir", $BL,
        "--quantum-dir", $OUT, "--out-dir", "cmp_ga_full_01", "--no-plots", "--pairs",
        "gnn_qwqpso:full_ga,gnn_qwqpso:full_de,gnn_qwqpso:full_pso,full_ga:full_pso,full_de:full_pso,gnn_qwqpso:gnn_pso")
    $zip = "ga_heavy_result.zip"
    Compress-Archive -Force -DestinationPath $zip -Path "result\cmp_ga_full_01\tables", "result\$OUT\tables", "result\$OUT\run.log"
} else {
    $OUT = "ga_short_01"
    Invoke-Step "1/1 test gnn_ga, gnn_de, gnn_qwqpso, gnn_pso (K 2..6)" (@("Quantummodel_code/run_quantum.py",
        "--stage", "test,tables", "--methods", "gnn_qwqpso,gnn_pso,gnn_ga,gnn_de",
        "--baseline-dir", $BL, "--out-dir", $OUT) + $common + $qw)
    $zip = "ga_light_result.zip"
    Compress-Archive -Force -DestinationPath $zip -Path "result\$OUT\tables", "result\$OUT\run.log"
}
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
