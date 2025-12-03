#!/bin/bash
# ============================================================
# Run All Experiments Script (Linux/Mac)
# ============================================================
# 세 가지 학습 모드를 순차적으로 실행하고 결과를 비교
# 1. TS2Vec Only (순수 시계열)
# 2. Cross-Modal (시계열 + LLM 프롬프트 임베딩)
# 3. Cross-Modal with News (시계열 + 뉴스 임베딩)
# ============================================================

# Configuration
DATASET="${1:-tech}"
EPOCHS="${2:-200}"
GPU="${3:-0}"

echo ""
echo "============================================================"
echo "  TS-CVA All Experiments Runner"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Epochs per experiment: $EPOCHS"
echo "GPU: $GPU"
echo "Started at: $(date)"
echo "============================================================"
echo ""

# ============================================================
# Experiment 1: TS2Vec Only
# ============================================================
echo ""
echo "[1/3] Running TS2Vec Only Experiment..."
echo "============================================================"

python train_forecasting.py $DATASET \
    --run-name "ts2vec_only_${EPOCHS}ep" \
    --repr-epochs $EPOCHS \
    --forecast-epochs 100 \
    --gpu $GPU \
    --eval

echo "TS2Vec Only completed!"

# ============================================================
# Experiment 2: Cross-Modal (without news)
# ============================================================
echo ""
echo "[2/3] Running Cross-Modal Experiment..."
echo "============================================================"

python train_forecasting.py $DATASET \
    --run-name "crossmodal_${EPOCHS}ep" \
    --repr-epochs $EPOCHS \
    --forecast-epochs 100 \
    --gpu $GPU \
    --use-cross-modal \
    --eval

echo "Cross-Modal completed!"

# ============================================================
# Experiment 3: Cross-Modal with News
# ============================================================
echo ""
echo "[3/3] Running Cross-Modal with News Experiment..."
echo "============================================================"

NEWS_EMB_PATH="datasets/yahoo_finance/$DATASET/news_embeddings.npz"

if [ -f "$NEWS_EMB_PATH" ]; then
    python train_forecasting.py $DATASET \
        --run-name "crossmodal_news_${EPOCHS}ep" \
        --repr-epochs $EPOCHS \
        --forecast-epochs 100 \
        --gpu $GPU \
        --use-cross-modal \
        --llm-embeddings $NEWS_EMB_PATH \
        --eval
    echo "Cross-Modal with News completed!"
else
    echo "Skipping: News embeddings not found at $NEWS_EMB_PATH"
    echo "Run 'python datasets/crawl_yahoo_news.py' first to get news data"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Experiment Summary"
echo "============================================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved in 'training/' directory:"
echo "  - training/${DATASET}__ts2vec_only_${EPOCHS}ep_*"
echo "  - training/${DATASET}__crossmodal_${EPOCHS}ep_*"
echo "  - training/${DATASET}__crossmodal_news_${EPOCHS}ep_*"
echo ""
echo "Check summary.txt in each directory for detailed metrics."
echo "============================================================"
