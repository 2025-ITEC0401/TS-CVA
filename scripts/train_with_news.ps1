# ============================================================
# Cross-Modal Training with News Data
# ============================================================
# 시계열 데이터 + 뉴스 기반 LLM 임베딩을 결합하여 학습
# 외부 뉴스 정보를 활용한 Cross-Modal Alignment
# ============================================================

# Configuration
$DATASET = "tech"           # tech or indices
$RUN_NAME = "crossmodal_with_news"
$EPOCHS = 200
$BATCH_SIZE = 32
$LR = 0.001
$REPR_DIMS = 320
$HIDDEN_DIMS = 64
$DEPTH = 10
$GPU = 0
$SEED = 42

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Cross-Modal Training with News Data" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET"
Write-Host "Run Name: $RUN_NAME"
Write-Host "Epochs: $EPOCHS"
Write-Host "Cross-Modal: ENABLED"
Write-Host "News Data: ENABLED"
Write-Host "============================================================" -ForegroundColor Cyan

# Check if news embeddings exist
$NEWS_EMB_PATH = "datasets/yahoo_finance/$DATASET/news_embeddings.npz"
if (-not (Test-Path $NEWS_EMB_PATH)) {
    Write-Host ""
    Write-Host "Warning: News embeddings not found at $NEWS_EMB_PATH" -ForegroundColor Yellow
    Write-Host "Attempting to generate news embeddings..." -ForegroundColor Yellow
    Write-Host ""
    
    # First, check if news data exists
    $NEWS_DATA_PATH = "datasets/yahoo_finance/$DATASET/news_data.csv"
    if (-not (Test-Path $NEWS_DATA_PATH)) {
        Write-Host "News data not found. Crawling news data first..." -ForegroundColor Yellow
        python datasets/crawl_yahoo_news.py --dataset $DATASET
    }
    
    # Generate news embeddings
    python storage/gen_news_embeddings.py --dataset $DATASET
}

# Run training with cross-modal and news embeddings
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
    --llm-embeddings "datasets/yahoo_finance/$DATASET/news_embeddings.npz" `
    --eval

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
