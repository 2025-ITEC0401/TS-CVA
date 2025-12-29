# TS-CVA 코드 구조 문서

## 📌 프로젝트 개요

**TS-CVA (TimeSeries-Context Vector Modality Alignment)**는 시계열 데이터와 LLM 기반 텍스트 임베딩을 결합하여 시계열 분석 성능을 향상시키는 딥러닝 프레임워크입니다.

### 핵심 특징
- **TS2Vec 기반 시계열 인코더**: Dilated Convolution을 사용한 시계열 표현 학습
- **Cross-Modal Attention**: 시계열과 LLM 임베딩 간의 정렬
- **멀티 태스크 지원**: 분류, 예측, 이상 탐지, 클러스터링

---

## 🏗️ 디렉토리 구조

```
TS-CVA/
├── models/              # 핵심 모델 구현
├── layers/              # 커스텀 레이어 (Cross-Modal, Normalization 등)
├── tasks/               # 다운스트림 태스크 (분류, 예측, 이상탐지 등)
├── storage/             # LLM 임베딩 생성 및 저장
├── datasets/            # 데이터 수집 및 전처리
├── data_provider/       # 데이터 로더
├── scripts/             # 학습 스크립트
├── training/            # 학습 결과 저장
├── plots/               # 시각화 결과
└── Embeddings/          # 생성된 LLM 임베딩
```

---

## 🔧 주요 모듈 설명

### 1. **모델 (models/)**

#### `ts_cva.py` - TSCVAEncoder
- **역할**: TS-CVA의 핵심 인코더
- **구조**:
  ```
  입력 시계열 [B, T, N]
    ↓
  TS2Vec Dilated Conv Encoder → [B, T, D]
    ↓
  Cross-Modal Attention (with LLM embeddings) → [B, T, D]
    ↓
  최종 표현
  ```
- **주요 기능**:
  - `forward()`: 시계열 인코딩 + Cross-Modal 정렬
  - `encode()`: 시계열을 표현 벡터로 변환
  - Hierarchical Contrastive Loss 계산

#### `TimeCMA.py` - Dual
- **역할**: Transformer 기반 대안 모델
- **구조**: TS Encoder + Prompt Encoder + Cross-Modal 정렬

#### `dilated_conv.py` - DilatedConvEncoder
- **역할**: TS2Vec의 Dilated Convolution 백본
- **특징**: 계층적 시간 패턴 캡처

#### `losses.py`
- Hierarchical Contrastive Loss
- Instance-wise & Temporal Contrastive Loss

---

### 2. **레이어 (layers/)**

#### `Cross_Modal_Align.py`
- **CrossModal 클래스**: Transformer 기반 Cross-Attention
- Query: 시계열 임베딩
- Key/Value: LLM 텍스트 임베딩
- 멀티헤드 어텐션으로 양방향 정렬

#### `StandardNorm.py`
- 시계열 정규화 레이어

#### `TS_Pos_Enc.py`
- Positional Encoding for Time Series

---

### 3. **태스크 (tasks/)**

#### `classification.py`
- **eval_classification()**: 분류 성능 평가
- Linear/SVM/KNN 프로토콜 지원
- 메트릭: Accuracy, AUPRC

#### `forecasting.py`
- **eval_forecasting()**: 시계열 예측 평가
- 메트릭: MSE, MAE, RMSE, MAPE
- Direction Accuracy (상승/하락 예측)

#### `anomaly_detection.py`
- 이상 탐지 태스크

#### `clustering.py`
- 클러스터링 태스크

---

### 4. **스토리지 (storage/)**

#### `store_emb_yahoo.py`
- Yahoo Finance 데이터용 LLM 임베딩 생성
- GPT-2 기반 뉴스 헤드라인 임베딩

#### `store_emb_uea.py`
- UEA 데이터셋용 임베딩 생성

#### `gen_prompt_emb_extended.py`
- GenPromptEmbExtended 클래스
- 프롬프트 생성 및 LLM 인코딩

---

### 5. **데이터셋 (datasets/)**

#### `download_yahoo_finance.py`
- Yahoo Finance API를 통한 주가 데이터 다운로드
- 시퀀스 생성 (`--create-sequences`)

#### `crawl_yahoo_news.py`
- Yahoo Finance 뉴스 크롤링

#### `preprocess_*.py`
- 데이터 전처리 스크립트들

---

### 6. **학습 스크립트 (scripts/)**

#### 세 가지 학습 모드:

1. **`train_ts2vec_only.ps1`**
   - 순수 시계열만 사용 (Cross-Modal ❌)
   
2. **`train_crossmodal.ps1`**
   - 시계열 + LLM 프롬프트 임베딩 (Cross-Modal ✅)
   
3. **`train_with_news.ps1`**
   - 시계열 + 뉴스 기반 LLM 임베딩

#### `run_all_experiments.ps1`
- 전체 실험 일괄 실행

---

## 🔄 학습 파이프라인

### 1. 데이터 준비
```powershell
# Yahoo Finance 데이터 다운로드
python datasets/download_yahoo_finance.py --preset tech --create-sequences
```

### 2. LLM 임베딩 생성
```powershell
# 프롬프트 기반 임베딩
python storage/store_emb_yahoo.py --dataset tech

# 뉴스 기반 임베딩
python datasets/crawl_yahoo_news.py --dataset tech
```

### 3. 모델 학습
```powershell
# 기본 학습
python train.py BasicMotions exp_name --loader UEA --epochs 100 --eval

# Cross-Modal 학습
.\scripts\train_crossmodal.ps1
```

### 4. 평가
- 학습 중 자동으로 downstream task 평가
- 결과는 `training/` 디렉토리에 저장

---

## 📊 주요 파일

### 학습 관련
- **`train.py`**: 메인 학습 스크립트
- **`train_forecasting.py`**: 예측 태스크 전용 학습
- **`ts_cva.py`**: TSCVAWrapper 클래스 (모델 래퍼)

### 유틸리티
- **`datautils.py`**: 데이터 로딩 함수들
  - `load_UCR()`, `load_UEA()`, `load_yahoo_data()`
- **`utils.py`**: 일반 유틸리티
- **`visualization.py`**: 학습 곡선 시각화

### 설정 파일
- **`TimeCMA.yaml`**: TimeCMA 모델 설정
- **`TS-CVA2.yaml`**: TS-CVA 모델 설정
- **`TS2Vec.yaml`**: TS2Vec 베이스라인 설정

---

## 🧪 실험 워크플로우

### Cross-Modal 학습 예시
```python
# 1. 데이터 로드
train_data = datautils.load_yahoo_data('tech')

# 2. LLM 임베딩 로드
llm_embeddings = torch.load('Embeddings/tech_llm.pt')

# 3. 모델 초기화
model = TSCVAWrapper(
    input_dims=7,
    output_dims=320,
    use_cross_modal=True,
    llm_embeddings=llm_embeddings
)

# 4. 학습
loss_log = model.fit(
    train_data,
    n_epochs=200,
    verbose=True
)

# 5. 예측 태스크 평가
out = tasks.eval_forecasting(
    model, data, train_slice, valid_slice, test_slice
)
```

---

## 🔑 핵심 클래스

### TSCVAWrapper (`ts_cva.py`)
```python
class TSCVAWrapper:
    def __init__(
        self,
        input_dims,
        output_dims=320,
        use_cross_modal=False,
        llm_embeddings=None,
        ...
    )
    
    def fit(train_data, n_epochs, ...)
    def encode(data, ...)
    def save(fn)
    def load(fn)
```

### TSCVAEncoder (`models/ts_cva.py`)
```python
class TSCVAEncoder(nn.Module):
    def forward(x, llm_emb=None, mask='all_true')
    def encode(x, encoding_window, llm_emb)
```

---

## 📈 출력 결과

### 학습 디렉토리 구조
```
training/tech__crossmodal_200ep_20231203_120000/
├── model.pkl              # 학습된 모델
├── model_best.pkl         # 최적 모델
├── forecast_head.pt       # 예측 헤드
├── summary.txt            # 성능 요약
├── predictions.npz        # 예측 결과
└── loss_log.pkl           # 학습 손실 로그
```

### 성능 메트릭
- **예측**: MSE, MAE, RMSE, MAPE, Direction Accuracy
- **분류**: Accuracy, AUPRC
- **이상탐지**: Precision, Recall, F1

---

## 🛠️ 커스터마이징

### 새로운 데이터셋 추가
1. `datautils.py`에 로더 함수 추가
2. `storage/`에 임베딩 생성 스크립트 추가
3. `scripts/`에 학습 스크립트 추가

### 새로운 태스크 추가
1. `tasks/`에 태스크 모듈 추가
2. `train.py`에 평가 로직 통합
