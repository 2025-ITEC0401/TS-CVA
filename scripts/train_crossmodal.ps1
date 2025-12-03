# ============================================================
# Cross-Modal Training Script (TS2Vec + LLM Embeddings)
# ============================================================
# 시계열 데이터 + LLM 임베딩을 결합하여 학습
# Cross-Modal Alignment 활성화 (뉴스 데이터 없이 시계열 프롬프트 임베딩만 사용)
# ============================================================

# Configuration
$DATASET = "tech"           # tech or indices
$RUN_NAME = "crossmodal"
$EPOCHS = 200
$BATCH_SIZE = 32
$LR = 0.001
$REPR_DIMS = 320
$HIDDEN_DIMS = 64
$DEPTH = 10
$GPU = 0
$SEED = 42

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Cross-Modal Training (TS2Vec + LLM Embeddings)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET"
Write-Host "Run Name: $RUN_NAME"
Write-Host "Epochs: $EPOCHS"
Write-Host "Cross-Modal: ENABLED"
Write-Host "============================================================" -ForegroundColor Cyan

# Check if LLM embeddings exist
$LLM_PATH = "datasets/yahoo_finance/$DATASET/llm_embeddings.npz"
if (-not (Test-Path $LLM_PATH)) {
    Write-Host ""
    Write-Host "Warning: LLM embeddings not found at $LLM_PATH" -ForegroundColor Yellow
    Write-Host "Generating LLM embeddings first..." -ForegroundColor Yellow
    Write-Host ""
    python storage/store_emb_yahoo.py --dataset $DATASET
}

# Run training with cross-modal
python train_forecasting.py $DATASET `
    --run-name $RUN_NAME `
    --repr-epochs $EPOCHS `
    --forecast-epochs 100 `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --repr-dims $REPR_DIMS `
    --hidden-dims $HIDDEN_DIMS `
    --depth $DEPTH `
    --gpu $GPU `
    --seed $SEED `
    --use-cross-modal `
    --eval

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
