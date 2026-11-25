# Uniform-split prior test (branch feat/prior-uni): can QPSO find the public split by itself?
# Init: m_b on the DEP/QoS bound (0 evals), public power split EQUALLY (no equal-ceiling formula).
#   prior_uqpso    boxed QW-QPSO (W, beta from the validation choice), no polish, B = 150
#   prior_uqpso_p  same + polish (1 population)
# Same cases / seeds as fair_B_01 -> paired with prior_fdirect / full_pso there.
# Two machines, split by K:
#   3060: powershell -ExecutionPolicy Bypass -File run_uni.ps1 -KTest 6,8,10 -Out uni_B_hi -Workers 12
#   4090: powershell -ExecutionPolicy Bypass -File run_uni.ps1 -KTest 2,3,4,5 -Out uni_B_lo -Workers 19 -Baseline bl_kscale_01 -ConfigFrom dprior_B_01
# Status: logs\uni_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 12,
    [string]$KTest = "2,3,4,5,6,8,10",
    [string]$Out = "uni_B_01",
    [string]$Baseline = "prior_blB",
    [string]$ConfigFrom = "nopol_B_01",
    [string]$Methods = "prior_uqpso,prior_uqpso_p"
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\uni_status.txt"

function Write-Status([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line -Encoding ascii
    Write-Host $line
}

Write-Status "=== uniform-prior run start out=$Out K=$KTest workers=$Workers baseline=$Baseline config=$ConfigFrom ==="
$src = "result\$ConfigFrom\validation\selected_dprior.json"
if (-not (Test-Path $src)) { Write-Status "FAILED: missing $src"; exit 1 }
New-Item -ItemType Directory -Force -Path "result\$Out\validation" | Out-Null
Copy-Item -Force $src "result\$Out\validation\selected_dprior.json"
Write-Status ("config: " + ((Get-Content $src) -join ' '))

$t0 = Get-Date
$log = Join-Path $PSScriptRoot "logs\uni_$Out.txt"
& $Python Quantummodel_code/run_quantum.py --stage test --methods $Methods `
    --baseline-dir $Baseline --out-dir $Out --preset paper --budget-short 150 --k-test 2,3,4,5,6,8,10 --k-subset $KTest `
    --workers $Workers --quiet --no-plots *> $log
if ($LASTEXITCODE -ne 0) { Write-Status "FAILED test (exit $LASTEXITCODE), output: $log"; exit $LASTEXITCODE }
Write-Status ("DONE test in {0:hh\:mm\:ss}" -f ((Get-Date) - $t0))
$zip = "uni_$($Out)_result.zip"
Compress-Archive -Force -DestinationPath $zip -Path "result\$Out\test\trials.csv", "result\$Out\run.log", "result\$Out\validation\selected_dprior.json"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
