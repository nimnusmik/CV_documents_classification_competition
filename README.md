# 🏆 Computer Vision Competition
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![F1 Score](https://img.shields.io/badge/F1_Score-0.9689+-brightgreen.svg)](https://github.com/your-repo/issues)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/Optimization-Optuna-purple.svg)](https://optuna.org/)
[![Pipeline](https://img.shields.io/badge/Pipeline-Full_Automation-green.svg)](#)

## Team

| 프로필 | 이름 (깃허브) | MBTI | 전공/학과 | 담당 역할 |
|:------:|:-------------:|:----:|:---------:|:----------|
| <img src="https://github.com/user-attachments/assets/a24cf78c-2c8f-47b9-b53b-867557872d88" width="200" height="200"> | [김선민](https://github.com/nimnusmik) | ENFJ | 경영&AI 융합 학부 | 팀 리드, EDA/모델링, 성능 고도화 |
| <img src="https://github.com/user-attachments/assets/489d401e-f5f5-4998-91a0-3b0f37f4490f" width="200" height="200"> | [김병현](https://github.com/Bkankim) | ENFP | 정보보안 | 전처리, 시스템/모델 개발, 성능 고도화 |
| <img src="https://github.com/user-attachments/assets/55180131-9401-457e-a600-312eda87ded9" width="200" height="200"> | [임예슬](https://github.com/joy007fun/joy007fun) | ENTP | 관광경영&컴퓨터공학, 클라우드 인프라 | EDA/모델링, 성능 고도화 |
| <img src="https://github.com/user-attachments/assets/10a2c088-72cb-45cd-8772-b683bc2fb550" width="200" height="200"> | [정서우](https://github.com/Seowoo-C) | INFJ | 화학 | EDA/모델링, 성능 고도화 |
| <img src="https://github.com/user-attachments/assets/5c04a858-46ed-4043-9762-b7eaf7b1149a" width="200" height="200"> | [최현화](https://github.com/AIBootcamp14/computervisioncompetition-cv-1) | ISTP | 컴퓨터공학 | EDA/모델링, 성능 고도화, 모듈화 설계/구현, Git 브랜치·병합·충돌 관리 |
# Computer Vision Competition

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![F1 Score](https://img.shields.io/badge/F1_Score-0.9689+-brightgreen.svg)](https://github.com/your-repo/issues)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

---

## 프로젝트 개요

17개 클래스 문서 이미지 분류를 위한 Computer Vision 경진대회 솔루션입니다. 단일 폴드부터 다중 모델 앙상블까지 다양한 전략을 지원하며, 전체 파이프라인이 자동화되어 있습니다.

### 주요 성과

- **F1 Score 0.9750+ 달성** (다중 모델 앙상블)
- **경진대회 2등** 수상
- Optuna 베이지안 최적화를 통한 체계적인 하이퍼파라미터 튜닝
- 200+ 실험 추적 및 관리

---

## 프로젝트 구조

```
computer-vision-competition-1SEN/
├── data/raw/                   # 원본 데이터 (학습 1570개, 테스트)
├── configs/                    # 설정 파일 (학습/추론/최적화)
├── src/
│   ├── training/               # 학습 시스템
│   ├── inference/              # 추론 시스템
│   ├── models/                 # 모델 아키텍처
│   ├── data/                   # 데이터 처리 및 증강
│   ├── optimization/           # Optuna 최적화
│   ├── calibration/            # Temperature Scaling
│   └── utils/                  # 유틸리티
├── experiments/                # 실험 결과
├── submissions/                # 제출 파일
├── docs/                       # 상세 문서
└── notebooks/                  # 실험 노트북
```

---

## 빠른 시작

### 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd computer-vision-competition-1SEN

# Python 환경 설정
pyenv install 3.11.9
pyenv virtualenv 3.11.9 cv_py3_11_9
pyenv activate cv_py3_11_9
pip install -r requirements.txt
```

### 데이터 준비

```
data/raw/
├── train/          # 학습 이미지 (17개 클래스)
├── test/           # 테스트 이미지
├── train.csv       # 학습 라벨
└── sample_submission.csv
```

### 실행 방법

**기본 학습 및 추론**
```bash
# 단일 폴드 학습
python src/training/train_main.py --config configs/train.yaml --mode basic

# 추론
python src/inference/infer_main.py --config configs/infer.yaml --mode basic
```

**고성능 파이프라인**
```bash
# K-fold 교차검증
python src/training/train_main.py --config configs/train_highperf.yaml --mode highperf

# K-fold 앙상블 추론
python src/inference/infer_main.py \
    --config configs/infer_highperf.yaml \
    --mode highperf \
    --fold-results experiments/train/latest-train/fold_results.yaml
```

**최적화 파이프라인**
```bash
# Optuna 하이퍼파라미터 최적화
python src/training/train_main.py \
    --config configs/train_multi_model_ensemble.yaml \
    --mode full-pipeline \
    --optimize --n-trials 50 \
    --use-calibration \
    --auto-continue
```

---

## 핵심 기능

### 모델 아키텍처

지원하는 10개 모델 중 주요 모델:

| 모델 | 최고 F1 | 학습 시간 | GPU 메모리 | 특징 |
|------|---------|----------|-----------|------|
| ConvNeXt Base 384 | 0.9836 | 52분 | 12GB | ImageNet-22k 사전학습 |
| Swin Transformer Base 384 | 0.9489 | 63분 | 14GB | Vision Transformer 기반 |
| EfficientNet V2 B3 | 0.9305 | 45분 | 10GB | 효율성과 성능의 균형 |

RECOMMENDED_MODELS 레지스트리로 간편한 모델 관리:
```yaml
models:
  fold_0: "efficientnet_b3"
  fold_1: "swin_base_384"
  fold_2: "convnext_base_384"
```

### 데이터 증강 시스템

**학습 증강**
- Hard Augmentation: 에포크별 동적 확률 스케줄링
- Mixup & CutMix: 데이터 믹싱 기법

**TTA (Test-Time Augmentation)**
- Essential (5가지): 회전 + 밝기 조정 (17분 추론)
- Comprehensive (15가지): 전체 변환 세트 (50분+ 추론)

### 최적화 기법

**Optuna 하이퍼파라미터 최적화**
- 캐싱 시스템: 150-300x 속도 향상
- 성공 사례: ConvNeXt F1 0.8234 → 0.9478 (+15.09%)
- TPE Sampler 베이지안 최적화

**Temperature Scaling**
- 확률 보정으로 모델 신뢰도 향상
- Calibrated Inference 지원

**GPU 메모리 최적화**
- 하드웨어별 자동 배치 크기 조정
- Mixed Precision Training (FP16)
- 메모리 프로파일링

### 앙상블 전략

**K-Fold 앙상블**
- 5개 폴드 독립 학습
- 성능 기반 가중 평균
- F1 0.95-0.98 달성

**다중 모델 앙상블**
- 아키텍처 다양성: ConvNeXt + Swin + EfficientNet
- F1 0.96-0.99 달성 가능

---

## 성능 벤치마크

### 최고 성능 기록

| 순위 | F1 Score | 전략 | 모델 | 시간 | 최적화 기법 |
|-----|----------|------|------|------|-----------|
| 1위 | 0.98362 | 단일 폴드 최적화 | ConvNeXt Base 384 | 23분 | Optuna + Hard Aug + EMA |
| 2위 | 0.97918 | 장기 학습 | ConvNeXt Base 384 | 300 epoch | 기본 설정 |
| 3위 | 0.96909 | 기준 모델 | ConvNeXt Base 384 | 100 epoch | 표준 설정 |

### 전략별 성능 비교

| 학습 전략 | 시간 | 예상 F1 | GPU 메모리 | 최적 활용 상황 |
|-----------|------|---------|-----------|---------------|
| 단일 폴드 | 30분 | 0.92-0.95 | 8GB | 초기 실험, 빠른 검증 |
| K-fold CV | 2시간 | 0.95-0.98 | 16GB | 최종 제출, 대회용 |
| 다중 모델 | 3시간 | 0.96-0.99 | 24GB+ | 고사양 GPU, 우승용 |
| Optuna 최적화 | 5시간 | 0.97-0.99+ | 16GB | 시간 여유, 최고 성능 |

### TTA 전략 비교

| TTA 전략 | 변환 수 | 추론 시간 | F1 향상 | 메모리 사용 |
|---------|--------|----------|---------|-------------|
| No TTA | 1 | 5분 | 기준점 | 4GB |
| Essential TTA | 5 | 17분 | +0.015 | 8GB |
| Comprehensive TTA | 15 | 50분+ | +0.035 | 24GB+ |

---

## 설정 파일 예시

### 학습 설정 (configs/train_highperf.yaml)

```yaml
data:
  folds: 5
  valid_fold: "all"
  stratify: true

model:
  name: "convnext_base_384"
  drop_rate: 0.1
  drop_path_rate: 0.2

train:
  epochs: 50
  batch_size: 64
  lr: 1e-4
  use_mixup: true
  use_ema: true
  use_advanced_augmentation: true
  temperature_scaling: true
```

### 추론 설정 (configs/infer_highperf.yaml)

```yaml
inference:
  model_paths:
    - "experiments/train/latest-train/fold_0/best_model.pth"
    - "experiments/train/latest-train/fold_1/best_model.pth"
    # ... fold_2, fold_3, fold_4
  
  tta:
    enabled: true
    strategy: "essential"
    transformations: 5
  
  ensemble:
    method: "weighted_average"
    weights: [0.2, 0.2, 0.2, 0.2, 0.2]
```

---

## 시스템 관리

### 모니터링

```bash
# GPU 모니터링
watch -n 1 nvidia-smi

# 메모리 프로파일링
python src/utils/gpu_optimization/memory_profiler.py

# 학습 로그 확인
tail -f logs/$(date +%Y%m%d)/train/*.log
```

### 트러블슈팅

**GPU 메모리 부족**
```bash
# 자동 배치 크기 조정
python src/utils/gpu_optimization/auto_batch_size.py \
    --config configs/train_highperf.yaml \
    --target-memory 0.9
```

**체크포인트 검증**
```bash
python src/utils/core/validate_checkpoint.py \
    --checkpoint experiments/train/latest-train/fold_0/best_model.pth
```

**제출 파일 검증**
```bash
python src/utils/core/validate_submission.py \
    --submission submissions/latest/final_submission.csv \
    --sample data/raw/sample_submission.csv
```

---

## 결과 분석

```bash
# 최신 실험 결과 요약
python src/utils/core/experiment_summary.py \
    --date $(date +%Y%m%d) \
    --top 10

# 모델별 성능 비교
python src/utils/visualizations/performance_comparison.py \
    --experiments experiments/train/20250910/
```

---

## 주요 성과 및 기여

- F1 Score 0.9750+ 달성 (경진대회 2등)
- 완전 자동화된 학습-추론 파이프라인 구축
- Optuna 최적화로 15% 성능 향상 달성
- 10개 모델 아키텍처 지원 및 체계적 벤치마킹
- 200+ 실험 추적 및 재현 가능한 연구 환경 구축

## 라이선스

MIT License
