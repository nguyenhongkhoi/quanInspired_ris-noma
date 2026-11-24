# DEP-aware prior + boxed QW-QPSO (branch feat/prior-dep). Learning-free; Set B layouts (bl_kscale_01).
# Goal: give the QPSO swarm a real job. The covertness (DEP) bound on m_b is solved from the Willie
# moments (0 objective evaluations) and becomes (i) the prior's m_b and (ii) the upper edge of the
# z_b search box; the swarm then searches the public split / RIS share / m_b below the cap.
#
#   0. smoke_dprior.py (assumption checks, ~1 min)
#   1. validate on Set B validation, K_val = 2,3,4,5,6,8,10:
#        prior_dqpso grid (s x L x beta = 12 cells), prior_ddirect / prior_dpolish / prior_direct /
#        prior_polish (s only)
#   2. test K = 2..6, 8, 10: prior_dqpso (+ ablations noW, noBox), prior_ddirect, prior_dpolish,
#        prior_direct, prior_polish, prior_qwqpso (config selected earlier on Set B val, 3060 run)
#   3. make_report against gnn_pso (original method) and full_pso of bl_kscale_01
#
#   powershell -ExecutionPolicy Bypass -File run_dprior.ps1 [-Python <exe>] [-Workers 19]
# Status: logs\dprior_status.txt. Re-running resumes finished (layout, K) groups.

param(
    [string]$Python = "",
    [int]$Workers = 19
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\dprior_status.txt"

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
$OUT = "dprior_B_01"
$CMP = "cmp_dprior_01"
$common = @("--preset", "paper", "--k-test", "2,3,4,5,6,8,10", "--k-val", "2,3,4,5,6,8,10",
            "--budgets-val", "150", "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== dprior run start (python=$Python workers=$Workers) ==="

Invoke-Step "0/4 smoke_dprior" @("smoke_dprior.py")

Invoke-Step "1/4 validate (Set B val, K_val 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "validate", "--methods", "prior_dqpso,prior_ddirect,prior_dpolish,prior_direct,prior_polish",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

# prior_qwqpso: configuration selected on Set B validation (K_val 2..6) in run prior_B_01 (3060)
$selp = "result\$OUT\validation\selected_prior.json"
if (-not (Test-Path $selp)) {
    $json = '{"prior_s": 0.2, "well_scale": 0.1, "well_frac": 1.0, "beta_start": 0.5, "beta_end": 0.25, "polish_pops": 1, "source": "prior_B_01 validation (3060)"}'
    Set-Content -Path $selp -Value $json -Encoding ascii
}

Invoke-Step "2/4 test (K 2..6,8,10)" (@("Quantummodel_code/run_quantum.py",
    "--stage", "test,tables",
    "--methods", "prior_dqpso,prior_dqpso_noW,prior_dqpso_noBox,prior_ddirect,prior_dpolish,prior_direct,prior_polish,prior_qwqpso",
    "--baseline-dir", $BL, "--out-dir", $OUT) + $common)

Invoke-Step "3/4 compare with gnn_pso (original) / full_pso" @("make_report.py", "--baseline-dir", $BL,
    "--quantum-dir", $OUT, "--out-dir", $CMP, "--no-plots", "--pairs",
    "prior_dqpso:gnn_pso,prior_dqpso:full_pso,prior_dpolish:gnn_pso,prior_dpolish:full_pso,prior_qwqpso:gnn_pso,prior_dqpso:prior_dpolish,prior_dqpso:prior_dqpso_noW,prior_dqpso:prior_dqpso_noBox,prior_dqpso:prior_qwqpso")

$zip = "dprior_result.zip"
$vd = "result\$OUT\validation"
Compress-Archive -Force -DestinationPath $zip -Path `
    "$vd\dprior_grid.csv", "$vd\selected_dprior.json", "$vd\selected_prior_base.json", `
    "$vd\prior_ddirect_grid.csv", "$vd\prior_dpolish_grid.csv", "$vd\prior_direct_grid.csv", "$vd\prior_polish_grid.csv", `
    "result\$OUT\tables", "result\$CMP\tables", "result\$OUT\run.log"
Write-Status "=== ALL DONE. Result: $((Resolve-Path $zip).Path) ==="
