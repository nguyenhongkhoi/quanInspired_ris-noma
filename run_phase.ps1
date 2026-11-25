# RIS-phase extension (branch feat/ris-phase): fair prior + QW-QPSO vs Full PSO 1800, Set B test.
# One shard per machine:
#   4090: powershell -ExecutionPolicy Bypass -File run_phase.ps1 -Shard 0/2 -Workers 19
#   3060: powershell -ExecutionPolicy Bypass -File run_phase.ps1 -Shard 1/2 -Workers 12
# Output: result\phase_B_s<i>\test\trials.csv -> phase_s<i>_result.zip
# Status: logs\phase_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 12,
    [string]$Shard = "0/2",
    [string]$Baseline = "prior_blB",
    [string]$ConfigFrom = "nopol_B_01",
    [string]$Methods = "phase_fdirect,phase_fqpso,phase_fullpso",
    [string]$KTest = "2,3,4,5,6,8,10",
    [int]$Replicates = 1
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\phase_status.txt"
$i = $Shard.Split("/")[0]
$OUT = "phase_B_s$i"

function Write-Status([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line -Encoding ascii
    Write-Host $line
}

Write-Status "=== phase run start shard=$Shard out=$OUT workers=$Workers methods=$Methods K=$KTest reps=$Replicates ==="
$cfg = "result\$ConfigFrom\validation\selected_dprior.json"
if (-not (Test-Path $cfg)) { Write-Status "FAILED: missing $cfg"; exit 1 }
if (-not (Test-Path "result\$Baseline\cases\cases_test.csv")) { Write-Status "FAILED: missing result\$Baseline\cases"; exit 1 }

$t0 = Get-Date
$log = Join-Path $PSScriptRoot "logs\phase_s$i.txt"
& $Python Quantummodel_code/run_phase.py --baseline-dir $Baseline --config-from $ConfigFrom --out-dir $OUT `
    --shard $Shard --workers $Workers --methods $Methods --k-test $KTest --replicates $Replicates *> $log
if ($LASTEXITCODE -ne 0) { Write-Status "FAILED run_phase (exit $LASTEXITCODE), output: $log"; exit $LASTEXITCODE }
Write-Status ("DONE run_phase in {0:hh\:mm\:ss}" -f ((Get-Date) - $t0))

$zip = "phase_s$($i)_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path "result\$OUT\test\trials.csv", "result\$OUT\run.log", "result\$OUT\config_phase.json"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
