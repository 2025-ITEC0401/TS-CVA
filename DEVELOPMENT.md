# TS-CVA - Development Documentation

> 이 문서는 TS-CVA 프로젝트의 환경 설정, 접속 방법, 주요 명령어 등을 정리한 참조 문서입니다.
> **새로운 세션에서 작업을 시작할 때 이 문서를 먼저 읽어주세요.**

---

## 📋 프로젝트 개요

**TS-CVA** (Time Series - Cross-modal Variable Alignment)
예측에 특화된 새로운 임베딩 기술을 개발하는 연구 프로젝트

### 프로젝트 목표
- **베이스라인**: TimeCMA (AAAI 2025) - LLM 기반 다변량 시계열 예측
- **목표**: TimeCMA의 내부 메커니즘을 개선하여 예측 성능을 향상시키는 새로운 임베딩 방법론 개발
- **핵심 아이디어**: Cross-modal Variable Alignment를 통한 예측 특화 임베딩

### 참고 자료
- **TimeCMA 논문**: [arXiv:2406.01638](https://arxiv.org/abs/2406.01638)
- **TimeCMA GitHub**: https://github.com/ChenxiLiu-HNU/TimeCMA

---

## 🖥️ 서버 환경 정보

### Lab Server 접속 정보
- **호스트**: `lab-server` (intern@155.230.36.16)
- **비밀번호**: `intern`
- **프로젝트 경로**: `~/TS-CVA`

### 하드웨어
- **GPU**: NVIDIA RTX 3090 x2 (각 24GB VRAM)
- **CUDA 드라이버**: 535.183.01
- **Python**: 3.10

### Conda 환경

#### TS-CVA 환경 (새로운 프로젝트)
- **환경 이름**: `TS-CVA`
- **환경 경로**: `/hdd/conda_envs/envs/TS-CVA`
- **Python 경로**: `/hdd/conda_envs/envs/TS-CVA/bin/python3`
- **용도**: TS-CVA 개발 및 실험

#### TimeCMA 환경 (베이스라인)
- **환경 이름**: `TimeCMA`
- **환경 경로**: `/hdd/conda_envs/envs/TimeCMA`
- **용도**: 베이스라인 비교 실험

**공통**:
- **Conda 경로**: `/opt/anaconda3/bin/conda`

---

## 🚀 빠른 시작 가이드

### 1. 서버 접속

#### 방법 1: 스크립트 사용 (권장)
```bash
cd /Users/isangmin/Desktop/종합설계프로젝트/TS-CVA
./ssh_connect.sh
```

#### 방법 2: 수동 접속
```bash
ssh intern@lab-server
cd ~/TS-CVA
```

### 2. Conda 환경 활성화

#### TS-CVA 환경 (기본)
```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate TS-CVA
```

#### TimeCMA 환경 (베이스라인 비교용)
```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate TimeCMA
```

### 3. 환경 확인
```bash
python --version  # Python 3.10
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # 2.1.2
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"  # 4.36.2
```

---

## 📦 설치된 패키지 정보

### 핵심 패키지 (TS-CVA 환경)
| 패키지 | 버전 | 용도 |
|--------|------|------|
| Python | 3.10 | 런타임 |
| PyTorch | 2.1.2 | 딥러닝 프레임워크 |
| Transformers | 4.36.2 | LLM (GPT-2) |
| tokenizers | 0.15.0 | 토크나이저 |
| sentencepiece | 0.2.0 | 텍스트 처리 |
| einops | 0.7.0 | 텐서 연산 |
| h5py | 3.7.0 | 임베딩 저장 |
| pandas | 1.3.5 | 데이터 처리 |
| numpy | 1.22.4 | 수치 연산 |
| scikit-learn | 1.0.2 | 머신러닝 유틸 |

전체 패키지 목록은 `env.yaml` 참조

---

## 🔧 주요 스크립트 및 명령어

### 로컬 머신 스크립트

#### 1. ssh_connect.sh
서버 자동 접속 및 프로젝트 디렉토리 이동
```bash
./ssh_connect.sh
```

#### 2. sync_to_server.sh
로컬 변경사항을 서버로 동기화
```bash
./sync_to_server.sh
```

**제외 파일**: `__pycache__/`, `.git/`, `*.log`, `.DS_Store` 등

### 서버 스크립트

#### 3. setup_env.sh
Conda 환경 자동 생성
```bash
bash setup_env.sh
```

---

## 📚 개발 워크플로우

### 기본 개발 흐름

```
1. 로컬에서 코드 수정
   ↓
2. ./sync_to_server.sh (서버로 동기화)
   ↓
3. ./ssh_connect.sh (서버 접속)
   ↓
4. conda activate TS-CVA
   ↓
5. 실험 실행 및 결과 확인
```

### Step 1: 임베딩 생성 및 저장 (선행 작업)

프롬프트 임베딩을 미리 생성하여 학습 속도 향상
```bash
# ETTm1 데이터셋 예시
python storage/store_emb.py \
  --data_path ETTm1 \
  --divide train \
  --input_len 96
```

**생성 위치**: `Embeddings/{dataset_name}/{train|val|test}/`

### Step 2: 모델 학습

```bash
# 기본 학습
python train.py \
  --data_path ETTm1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --channel 64 \
  --e_layer 2 \
  --d_layer 2 \
  --dropout_n 0.5 \
  --epochs 100
```

### Step 3: 베이스라인(TimeCMA)과 비교

```bash
# TimeCMA 환경으로 전환
conda activate TimeCMA

# 동일한 설정으로 베이스라인 실행
python train.py \
  --data_path ETTm1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 16

# 결과 비교
```

### 주요 파라미터
- `--data_path`: 데이터셋 이름 (ETTm1, ETTh1, ECL, FRED, ILI, Weather 등)
- `--seq_len`: 입력 시퀀스 길이 (기본: 96)
- `--pred_len`: 예측 길이 (96, 192, 336, 720)
- `--channel`: 임베딩 차원 (기본: 32)
- `--e_layer`: 인코더 레이어 수
- `--d_layer`: 디코더 레이어 수

---

## 📁 프로젝트 구조

```
TS-CVA/
├── models/
│   └── TimeCMA.py          # 베이스 모델 (개선 예정)
├── layers/
│   ├── Cross_Modal_Align.py  # 교차 모달리티 정렬 (개선 대상)
│   ├── Embed.py
│   └── ...
├── storage/
│   ├── gen_prompt_emb.py   # GPT-2 프롬프트 임베딩 생성
│   └── store_emb.py        # 임베딩 저장
├── data_provider/
│   ├── data_loader_emb.py  # 임베딩 포함 데이터 로더
│   └── data_loader_save.py
├── scripts/
│   ├── Store_ETT.sh        # 임베딩 생성 스크립트
│   ├── ETTm1.sh           # ETTm1 학습 스크립트
│   └── ...
├── train.py               # 학습 메인 스크립트
├── env.yaml              # Conda 환경 설정 (TS-CVA)
├── ssh_connect.sh        # 서버 접속 스크립트
├── sync_to_server.sh     # 동기화 스크립트
├── setup_env.sh          # 환경 설정 스크립트
└── DEVELOPMENT.md       # 이 문서
```

---

## 🗂️ 데이터셋 정보

### 현재 상태
- **있음**: `dataset/Epilepsy/`
- **없음**: ETTm1, ETTm2, ETTh1, ETTh2, ECL, FRED, ILI, Weather

### 데이터셋 다운로드 방법

**다운로드 링크**:
1. [TimesNet 데이터셋](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)
2. [TFB 벤치마크](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view)

**서버 데이터셋 공유 경로**: `/hdd/dataset/`

다운로드 후 `~/TS-CVA/dataset/` 또는 `/hdd/dataset/`에 배치

---

## 🐛 트러블슈팅

### 1. Conda 환경 활성화 실패
```bash
# Conda 초기화
source /opt/anaconda3/etc/profile.d/conda.sh

# 환경 목록 확인
conda env list

# TS-CVA 환경 활성화
conda activate TS-CVA
```

### 2. 환경 전환 (TS-CVA ↔ TimeCMA)
```bash
# 현재 환경 비활성화
conda deactivate

# 원하는 환경 활성화
conda activate TS-CVA  # 또는 TimeCMA
```

### 3. 패키지 ImportError
```bash
# 환경 내에서 패키지 재설치
conda activate TS-CVA
pip install {package_name}
```

### 4. GPU 사용 불가
```bash
# GPU 상태 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. 데이터셋 파일 없음
```bash
# 공유 데이터셋 확인
ls /hdd/dataset/

# 심볼릭 링크 생성
ln -s /hdd/dataset/{dataset_name} ~/TS-CVA/dataset/
```

---

## 📝 코드 수정 후 동기화

### 로컬에서 코드 수정
```bash
# 로컬에서 코드 편집
vim /Users/isangmin/Desktop/종합설계프로젝트/TS-CVA/models/TimeCMA.py

# 서버로 동기화
./sync_to_server.sh
```

### 서버에서 바로 수정
```bash
# 서버 접속
./ssh_connect.sh

# 환경 활성화 후 편집
conda activate TS-CVA
vim ~/TS-CVA/models/TimeCMA.py

# 학습 실행
python train.py --data_path ETTm1 ...
```

---

## 🎯 주요 실험 실행 예시

### 1. TS-CVA 개발 실험
```bash
conda activate TS-CVA
cd ~/TS-CVA

# TS-CVA 모델로 학습
python train.py \
  --data_path ETTm1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 16 \
  --epochs 100
```

### 2. TimeCMA 베이스라인 실험
```bash
conda activate TimeCMA
cd ~/TS-CVA

# 동일 설정으로 베이스라인 실행
python train.py \
  --data_path ETTm1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 16 \
  --epochs 100
```

### 3. 여러 pred_len 실험
```bash
# scripts/ETTm1.sh 실행
bash scripts/ETTm1.sh

# 백그라운드 실행 (nohup 사용)
# pred_len: 96, 192, 336, 720 모두 자동 실행
```

### 4. 로그 확인
```bash
# 실시간 로그 확인
tail -f Results/ETTm1/*.log

# 학습 결과 확인
ls logs/$(date +%Y-%m-%d)*
```

---

## 💡 알아야 할 중요 사항

### 1. 프로젝트 구분
- **TS-CVA 환경**: 새로운 임베딩 방법론 개발 (메인)
- **TimeCMA 환경**: 베이스라인 비교 실험용
- 두 환경은 독립적으로 관리됨

### 2. LLM 다운로드
- **첫 실행 시** GPT-2 모델(~500MB)이 자동 다운로드됨
- 인터넷 연결 필요
- Hugging Face에서 자동으로 캐시 (`~/.cache/huggingface/`)

### 3. 임베딩 저장 방식
- **Offline embedding storage** 사용
- 학습 전 임베딩을 미리 생성하여 `Embeddings/` 디렉토리에 저장
- 학습 시 저장된 임베딩을 로드하여 속도 향상

### 4. 메모리 관리
- RTX 3090 24GB x2 사용
- batch_size 조절로 메모리 관리
- CUDA OOM 발생 시 batch_size 감소

### 5. 실험 재현성
- `--seed` 파라미터로 랜덤 시드 고정 (기본: 2024)
- 동일한 하이퍼파라미터로 재현 가능

---

## 🔬 연구 방향

### 현재 상태
- TimeCMA 코드베이스 기반
- 베이스라인 환경 구축 완료

### 다음 단계 (예정)
1. **분석 단계**: TimeCMA의 cross-modality alignment 메커니즘 분석
2. **개선 단계**: 예측 특화 임베딩 방법론 설계
3. **실험 단계**: 다양한 데이터셋에서 성능 비교
4. **평가 단계**: TimeCMA 대비 성능 향상 검증

---

## 🔗 유용한 링크

- **TimeCMA 논문**: https://arxiv.org/abs/2406.01638
- **TimeCMA GitHub**: https://github.com/ChenxiLiu-HNU/TimeCMA
- **Hugging Face (GPT-2)**: https://huggingface.co/gpt2
- **데이터셋**: [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)

---

## ✅ 체크리스트

새로운 세션에서 작업 시작 전 확인:

- [ ] 서버 접속: `./ssh_connect.sh`
- [ ] Conda 환경 활성화: `conda activate TS-CVA`
- [ ] Python 버전 확인: `python --version` (3.10)
- [ ] PyTorch 작동 확인: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 프로젝트 경로 확인: `cd ~/TS-CVA`
- [ ] 데이터셋 존재 확인: `ls dataset/` 또는 `ls /hdd/dataset/`
- [ ] 환경 확인: `echo $CONDA_DEFAULT_ENV` → TS-CVA 확인

---

## 📞 문의 및 참고

**베이스라인 (TimeCMA) 저자**: Chenxi Liu (chenxi.liu@ntu.edu.sg)

**로컬 환경**:
- macOS (Darwin 24.6.0)
- 로컬 프로젝트 경로: `/Users/isangmin/Desktop/종합설계프로젝트/TS-CVA`

**서버 환경**:
- Lab Server: intern@lab-server (155.230.36.16)
- 프로젝트 경로: `~/TS-CVA`
- TS-CVA 환경: `/hdd/conda_envs/envs/TS-CVA`
- TimeCMA 환경: `/hdd/conda_envs/envs/TimeCMA`

---

**마지막 업데이트**: 2025-11-09
**작성자**: Development Team

**변경 이력**:
- 2025-11-09: 프로젝트명 TimeCMA → TS-CVA 변경, 환경 분리
