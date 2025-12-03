# ============================================================
# Run All Experiments Script (Parallel Execution)
# ============================================================
# 세 가지 학습 모드를 동시에 실행하고 결과를 비교
# 1. TS2Vec Only (순수 시계열)
# 2. Cross-Modal (시계열 + LLM 프롬프트 임베딩)
# 3. Cross-Modal with News (시계열 + 뉴스 임베딩)
# 
# 각 실험의 로그는 해당 폴더의 output.log에 실시간으로 저장됨
# ============================================================

param(
    [string]$Dataset = "tech",
    [int]$Epochs = 200,
    [int]$GPU = 0
)

$ErrorActionPreference = "Continue"
$StartTime = Get-Date
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  TS-CVA All Experiments Runner (Parallel)" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "Dataset: $Dataset"
Write-Host "Epochs per experiment: $Epochs"
Write-Host "GPU: $GPU"
Write-Host "Started at: $StartTime"
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

# Create output directories
$BaseDir = "training"
$Exp1Dir = "$BaseDir/${Dataset}__ts2vec_only_${Epochs}ep_${Timestamp}"
$Exp2Dir = "$BaseDir/${Dataset}__crossmodal_${Epochs}ep_${Timestamp}"
$Exp3Dir = "$BaseDir/${Dataset}__crossmodal_news_${Epochs}ep_${Timestamp}"

New-Item -ItemType Directory -Force -Path $Exp1Dir | Out-Null
New-Item -ItemType Directory -Force -Path $Exp2Dir | Out-Null
New-Item -ItemType Directory -Force -Path $Exp3Dir | Out-Null

Write-Host "Output directories created:" -ForegroundColor Cyan
Write-Host "  [1] $Exp1Dir/output.log"
Write-Host "  [2] $Exp2Dir/output.log"
Write-Host "  [3] $Exp3Dir/output.log"
Write-Host ""

# ============================================================
# Start all experiments in parallel using Start-Process
# ============================================================

Write-Host "Starting all experiments in parallel..." -ForegroundColor Yellow
Write-Host ""

# Experiment 1: TS2Vec Only
Write-Host "[1/3] Starting TS2Vec Only..." -ForegroundColor Yellow
$Proc1 = Start-Process -FilePath "python" -ArgumentList @(
    "train_forecasting.py", $Dataset,
    "--run-name", "ts2vec_only_${Epochs}ep_${Timestamp}",
    "--repr-epochs", $Epochs,
    "--forecast-epochs", "100",
    "--gpu", $GPU,
    "--lr", "0.001",
    "--eval"
) -RedirectStandardOutput "$Exp1Dir/output.log" -RedirectStandardError "$Exp1Dir/error.log" -PassThru -NoNewWindow

# Experiment 2: Cross-Modal (lower learning rate for stability)
Write-Host "[2/3] Starting Cross-Modal..." -ForegroundColor Yellow
$Proc2 = Start-Process -FilePath "python" -ArgumentList @(
    "train_forecasting.py", $Dataset,
    "--run-name", "crossmodal_${Epochs}ep_${Timestamp}",
    "--repr-epochs", $Epochs,
    "--forecast-epochs", "100",
    "--gpu", $GPU,
    "--lr", "0.0005",
    "--use-cross-modal",
    "--eval"
) -RedirectStandardOutput "$Exp2Dir/output.log" -RedirectStandardError "$Exp2Dir/error.log" -PassThru -NoNewWindow

# Experiment 3: Cross-Modal with News (lower learning rate for stability)
$NewsEmbPath = "datasets/yahoo_finance/$Dataset/news_embeddings.npz"
if (Test-Path $NewsEmbPath) {
    Write-Host "[3/3] Starting Cross-Modal with News..." -ForegroundColor Yellow
    $Proc3 = Start-Process -FilePath "python" -ArgumentList @(
        "train_forecasting.py", $Dataset,
        "--run-name", "crossmodal_news_${Epochs}ep_${Timestamp}",
        "--repr-epochs", $Epochs,
        "--forecast-epochs", "100",
        "--gpu", $GPU,
        "--lr", "0.0005",
        "--use-cross-modal",
        "--llm-embeddings", $NewsEmbPath,
        "--eval"
    ) -RedirectStandardOutput "$Exp3Dir/output.log" -RedirectStandardError "$Exp3Dir/error.log" -PassThru -NoNewWindow
} else {
    Write-Host "[3/3] Skipping: News embeddings not found at $NewsEmbPath" -ForegroundColor Red
    $Proc3 = $null
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "All experiments started!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor logs in real-time with:" -ForegroundColor Green
Write-Host "  Get-Content $Exp1Dir/output.log -Wait -Tail 20"
Write-Host "  Get-Content $Exp2Dir/output.log -Wait -Tail 20"
Write-Host "  Get-Content $Exp3Dir/output.log -Wait -Tail 20"
Write-Host ""
Write-Host "Or monitor all at once:" -ForegroundColor Green
Write-Host "  Get-Content $Exp1Dir/output.log, $Exp2Dir/output.log, $Exp3Dir/output.log -Wait -Tail 5"
Write-Host ""

# ============================================================
# Wait for all processes to complete
# ============================================================
Write-Host "Waiting for experiments to complete..." -ForegroundColor Yellow
Write-Host "(Press Ctrl+C to cancel)" -ForegroundColor DarkGray
Write-Host ""

# Show progress while waiting
$Procs = @($Proc1, $Proc2)
if ($Proc3) { $Procs += $Proc3 }

while ($true) {
    $Running = $Procs | Where-Object { -not $_.HasExited }
    if ($Running.Count -eq 0) { break }
    
    $Status1 = if ($Proc1.HasExited) { "Done (Exit: $($Proc1.ExitCode))" } else { "Running..." }
    $Status2 = if ($Proc2.HasExited) { "Done (Exit: $($Proc2.ExitCode))" } else { "Running..." }
    $Status3 = if ($Proc3) { if ($Proc3.HasExited) { "Done (Exit: $($Proc3.ExitCode))" } else { "Running..." } } else { "Skipped" }
    
    Write-Host "`r[1] TS2Vec: $Status1 | [2] CrossModal: $Status2 | [3] WithNews: $Status3    " -NoNewline
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host ""

# ============================================================
# Summary
# ============================================================
$EndTime = Get-Date
$TotalTime = $EndTime - $StartTime

Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  Experiment Summary" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "Total time: $($TotalTime.TotalMinutes.ToString('F2')) minutes"
Write-Host ""
Write-Host "Exit codes:"
Write-Host "  [1] TS2Vec Only:        $($Proc1.ExitCode)"
Write-Host "  [2] Cross-Modal:        $($Proc2.ExitCode)"
if ($Proc3) {
    Write-Host "  [3] Cross-Modal+News:   $($Proc3.ExitCode)"
}
Write-Host ""
Write-Host "Results saved in:" -ForegroundColor Green
Write-Host "  $Exp1Dir/"
Write-Host "  $Exp2Dir/"
if ($Proc3) { Write-Host "  $Exp3Dir/" }
Write-Host ""
Write-Host "Check output.log and summary.txt in each directory."
Write-Host "============================================================" -ForegroundColor Magenta
