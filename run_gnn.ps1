# GNN-only solver on Set B (branch feat/gnn-direct): how far does a learned model get WITHOUT the
# closed-form prior? Same cases / seeds as fair_B_01, fullmeta_B_01, fullde_B_01 -> paired comparison.
#
#   1. gnn_blB    : cases (must be byte-identical to prior_blB, checked by SHA-256), PSO-reference labels
#                   (2 x Full PSO 1800 per train/val case, K = 2..5, as in the source paper), train GNN + MLP
#                   (seeds 7, 17, 27; checkpoint = lowest validation loss)
#   2. gnn_B_01   : test direct_gnn (GNN output + delta adjustment, 0 objective evaluations), K = 2..6, 8
#   3. gnnpso_B_01: test gnn_pso (GNN prediction as particle 0 + short PSO, B = 150), K = 2..6, 8
# K = 6, 8 are unseen by the networks (trained on K = 2..5).
#
#   powershell -ExecutionPolicy Bypass -File run_gnn.ps1 [-Python <exe>] [-Workers 12] [-SkipGnnPso]
# Status: logs\gnn_status.txt. Re-running resumes (finished label groups, checkpoints, test groups).

param(
    [string]$Python = "",
    [int]$Workers = 12,
    [string]$KTest = "2,3,4,5,6,8",
    [switch]$SkipGnnPso
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
if ($Python -eq "") { $Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$status = Join-Path $PSScriptRoot "logs\gnn_status.txt"

function Write-Status([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line -Encoding ascii
    Write-Host $line
}

function Invoke-Step([string]$name, [string]$logName, [string[]]$cmd) {
    $t0 = Get-Date
    $log = Join-Path $PSScriptRoot "logs\$logName"
    Write-Status "START $name :: python $($cmd -join ' ')"
    & $Python @cmd *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Status "FAILED $name (exit $LASTEXITCODE), output: $log"
        exit $LASTEXITCODE
    }
    Write-Status ("DONE {0} in {1:hh\:mm\:ss}" -f $name, ((Get-Date) - $t0))
}

$BL = "gnn_blB"
$REF = "prior_blB"
# Same settings as prior_blB (k-test 2..10 -> same layouts with K_max = 10), K restricted by --k-subset.
$common = @("--preset", "paper", "--budget-short", "150", "--k-test", "2,3,4,5,6,8,10",
            "--workers", "$Workers", "--quiet", "--no-plots")

Write-Status "=== GNN-only run start (python=$Python workers=$Workers K=$KTest) ==="

Invoke-Step "1/4 cases $BL" "gnn_cases.txt" (@("baseline_code/run_baseline.py", "--stage", "cases", "--out-dir", $BL) + $common)
foreach ($sp in "train", "val", "test") {
    $a = Get-FileHash -Algorithm SHA256 "result\$BL\cases\cases_$sp.csv"
    $b = Get-FileHash -Algorithm SHA256 "result\$REF\cases\cases_$sp.csv"
    if ($a.Hash -ne $b.Hash) { Write-Status "FAILED: cases_$sp.csv of $BL differs from $REF"; exit 2 }
}
Write-Status "cases identical to $REF (SHA-256 train/val/test)"

Invoke-Step "2/4 labels + train $BL" "gnn_labels_train.txt" (@("baseline_code/run_baseline.py", "--stage", "labels,train",
    "--out-dir", $BL) + $common)

Invoke-Step "3/4 test direct_gnn -> gnn_B_01" "gnn_test_direct.txt" (@("Quantummodel_code/run_quantum.py", "--stage", "test",
    "--methods", "direct_gnn", "--baseline-dir", $BL, "--out-dir", "gnn_B_01", "--k-subset", $KTest) + $common)
Compress-Archive -Force -DestinationPath "gnn_B_01_result.zip" -Path "result\gnn_B_01\test\trials.csv", "result\gnn_B_01\run.log", `
    "result\$BL\labels\labels_train.csv", "result\$BL\labels\labels_val.csv", "result\$BL\models\checkpoints.json", "result\$BL\run.log"
Write-Status "direct_gnn result: $((Resolve-Path gnn_B_01_result.zip).Path)"

if (-not $SkipGnnPso) {
    Invoke-Step "4/4 test gnn_pso -> gnnpso_B_01" "gnn_test_pso.txt" (@("Quantummodel_code/run_quantum.py", "--stage", "test",
        "--methods", "gnn_pso", "--baseline-dir", $BL, "--out-dir", "gnnpso_B_01", "--k-subset", $KTest) + $common)
    Compress-Archive -Force -DestinationPath "gnnpso_B_01_result.zip" -Path "result\gnnpso_B_01\test\trials.csv", "result\gnnpso_B_01\run.log"
    Write-Status "gnn_pso result: $((Resolve-Path gnnpso_B_01_result.zip).Path)"
}
Write-Status "=== ALL DONE ==="
