<div align="center">
  <h2><b> TS-CVA: Time Series Forecasting via Cross-modal Variable Alignment </b></h2>
</div>

This repository contains **TS-CVA** (Time Series - Cross-modal Variable Alignment), an advanced framework for multivariate time series forecasting that extends the TimeCMA approach with dual-modality learning.

## Overview

TS-CVA enhances time series forecasting by integrating two complementary modalities:

- **Vector Modality**: Pure time series feature extraction using contrastive learning methods (TS2Vec)
- **Context Modality**: LLM-empowered encoding with external knowledge and real-time information

This dual-modality approach is particularly effective for **stock market prediction**, where both historical price patterns and external events (news, sentiment, macroeconomic indicators) play crucial roles.

### Key Features

- Cross-modal alignment between vector and context representations
- Integration of external real-time information sources
- Pre-computed embedding storage for efficient inference
- Extensible architecture for various forecasting domains
- Built upon TimeCMA (AAAI 2025) foundation

### Based on TimeCMA

This project extends the TimeCMA framework:

```bibtex
@inproceedings{liu2024timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

## Architecture

TS-CVA implements a triple-modality architecture:

1. **Time Series Branch**: Extracts temporal patterns from raw time series data
2. **Vector Modality Branch**: Learns robust representations via contrastive learning
3. **Context Modality Branch**: Incorporates LLM-encoded external knowledge

The cross-modal alignment mechanism retrieves complementary information from both modalities, combining the strengths of pure time series learning and context-aware prediction.

## Dependencies

* Python 3.11
* PyTorch 2.1.2
* CUDA 12.1
* torchvision 0.8.0

```bash
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) and [TFB](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view).

## Usages
* ### Last token embedding storage

```bash
bash Store_{data_name}.sh
```

* ### Train and inference
   
```bash
bash {data_name}.sh
```

## License

This project is based on TimeCMA, which uses the S-Lab License 1.0. See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

We acknowledge the original TimeCMA authors for their foundational work in LLM-empowered time series forecasting. This project extends their framework with additional modalities for enhanced prediction capabilities.
