# ============================================================
# TS2Vec Only Training Script (No Cross-Modal Alignment)
# ============================================================
# 순수 시계열 데이터만 사용하여 TS2Vec 인코더로 학습
# Cross-Modal Alignment 비활성화
# ============================================================

# Configuration
$DATASET = "tech"           # tech or indices
$RUN_NAME = "ts2vec_only"
$EPOCHS = 200
$BATCH_SIZE = 32
$LR = 0.001
$REPR_DIMS = 320
$HIDDEN_DIMS = 64
$DEPTH = 10
$GPU = 0
$SEED = 42

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "TS2Vec Only Training (No Cross-Modal)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET"
Write-Host "Run Name: $RUN_NAME"
Write-Host "Epochs: $EPOCHS"
Write-Host "Cross-Modal: DISABLED"
Write-Host "============================================================" -ForegroundColor Cyan

# Run training
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
    --eval

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
