# Standard-setting baselines on Set B (branch feat/prior-dep): no tuning, paper / literature settings.
#   rand_pso  : PSO, B = 150, uniform init (Algorithm 1: c1 = c2 = 1.5, w 0.9 -> 0.4)
#   rand_qpso : QPSO, B = 150, uniform init (Sun et al. 2004: beta 1.0 -> 0.5)
#   full_qpso : QPSO, B = 1800, uniform init
# Test K = 2..6, 8, 10 on Set B cases -> result\std_B_01 -> std_result.zip (tables + trials.csv).
#
#   powershell -ExecutionPolicy Bypass -File run_std.ps1 [-Python <exe>] [-Workers 18] [-Baseline prior_blB]
# Status: logs\std_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 18,
    [string]$Baseline = "prior_blB"
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\std_status.txt"

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
$OUT = "std_B_01"
$OUTF = "std_full_B_01"
$common = @("--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10",
            "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== standard baselines start (python=$Python workers=$Workers baseline=$BL) ==="

if (-not (Test-Path "result\$BL\cases\cases_test.csv")) {
    Invoke-Step "0/2 cases Set B" @("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BL,
        "--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10", "--workers", "$Workers",
        "--quiet", "--no-plots")
}

Invoke-Step "1/2 test rand_pso, rand_qpso (B = 150)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "rand_pso,rand_qpso",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

Invoke-Step "2/2 test full_qpso (B = 1800)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables", "--methods", "full_qpso",
    "--baseline-dir", $BL, "--out-dir", $OUTF) + $common)

$zip = "std_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\$OUT\tables", "result\$OUT\test\trials.csv", "result\$OUT\run.log", `
    "result\$OUTF\tables", "result\$OUTF\test\trials.csv", "result\$OUTF\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
