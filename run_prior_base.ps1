# Swarm-free references for the learning-free study (branch feat/prior-qw), VALIDATION only:
#   prior_direct : the closed-form physics prior, no evaluation
#   prior_polish : prior + structural line search on m_b (150 evaluations, no swarm)
# Each picks its own RIS share s on the same 5-value grid and rule as prior_qwqpso.
# Runs in its own clone/out-dir, so it does not touch run_prior.ps1 outputs.
#
#   powershell -ExecutionPolicy Bypass -File run_prior_base.ps1 -Python <exe> [-Workers 8]
# Status: logs\prior_base_status.txt.

param(
    [string]$Python = "",
    [int]$Workers = 8
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\prior_base_status.txt"

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

$BLB = "prior_blB"
$OUT = "prior_B_base_01"
$kB = @("--k-test", "2,3,4,5,6,8,10")
$run = @("--preset", "paper", "--budget-short", "150", "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== prior base run start (python=$Python workers=$Workers) ==="
Invoke-Step "1/2 cases Set B" (@("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BLB) + $kB + $run)
Invoke-Step "2/2 validate prior_direct, prior_polish (Set B, K_val 2..6)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "prior_direct,prior_polish", "--k-val", "2,3,4,5,6", "--budgets-val", "150",
    "--baseline-dir", $BLB, "--out-dir", $OUT) + $kB + $run)

$zip = "prior_base_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path `
    "result\$OUT\validation\prior_direct_grid.csv", "result\$OUT\validation\prior_polish_grid.csv", `
    "result\$OUT\validation\selected_prior_base.json", "result\$OUT\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
