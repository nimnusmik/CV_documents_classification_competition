# 🏆 Computer Vision Competition - Document Classification Framework

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![F1 Score](https://img.shields.io/badge/F1_Score-0.98362-brightgreen.svg)](https://github.com/your-repo/issues)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/Optimization-Optuna-purple.svg)](https://optuna.org/)

## 📋 프로젝트 개요

**완전 자동화된 고성능 문서 분류 경진대회 프레임워크**입니다.

- 🎯 **최고 성능**: **F1 Score 0.98362** 달성 (2025-09-10)
- ⚡ **초고속 최적화**: Optuna 캐싱으로 trial당 2초 완료
- 🤖 **완전 자동화**: 학습 → 최적화 → 추론 → 제출 전 과정 원클릭
- 🔄 **유연한 구조**: 단일 폴드 ↔ K-Fold 설정 하나로 전환
- 📊 **체계적 추적**: WandB 통합 + 100+ 실험 기록

---

## 🏆 최고 성능 기록

### 🥇 F1 Score **0.98362** (2025-09-10 12:13)
```yaml
모델: ConvNeXt Base 384 (ImageNet-22k 사전학습)
학습시간: 23분 12초 (150 epoch)
핵심기법: Optuna 최적화 + Mixup + Hard Augmentation + EMA
설정파일: configs/20250910/train_optimized_*_1213.yaml
재현가능: ✅ 완전 재현 검증됨
```

### 📊 성능 순위 (최신 실험들)
| 순위 | F1 Score | 모델 | 날짜/시간 | 특징 |
|-----|----------|------|-----------|------|
| 🥇 | **0.98362** | ConvNeXt Base 384 | 2025-09-10 12:13 | Optuna 최적화 |
| 🥈 | 0.97918 | ConvNeXt Base 384 | 2025-09-10 09:29 | 장기 학습 (300 epoch) |
| 🥉 | 0.96909 | ConvNeXt Base 384 | 2025-09-10 09:08 | 기준 모델 (100 epoch) |
| 4위 | 0.95568 | ConvNeXt Base 384 | 2025-09-11 14:38 | 최신 실험 |

---

## 🚀 Quick Start

### 📦 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd computer-vision-competition-1SEN

# Python 환경 (pyenv 권장)
pyenv install 3.11.9
pyenv virtualenv 3.11.9 cv_py3_11_9
pyenv activate cv_py3_11_9
pip install -r requirements.txt
```

### 📁 2. 데이터 준비

```bash
# 데이터 구조 확인
data/raw/
├── train/          # 학습 이미지 (1570개)
├── test/           # 테스트 이미지
├── train.csv       # 학습 라벨 (17개 클래스)
└── sample_submission.csv
```

### ⚡ 3. 최고 성능 재현 (원클릭)

```bash
# 🏆 F1 0.98362 달성 설정으로 전체 파이프라인 실행
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline \
    --use-calibration \
    --optimize \
    --optuna-config configs/optuna_single_fold_config.yaml \
    --auto-continue
```

**실행 과정**:
1. 📊 Optuna 최적화 (20 trials × 2초 = 40초)
2. 🏋️ 최적 설정으로 전체 학습 (150 epoch, ~23분)
3. 🎯 Temperature Calibration 적용
4. 🔮 고성능 TTA 추론 실행
5. 📤 제출 파일 자동 생성

---

## 🏗️ 프로젝트 아키텍처

### 📂 디렉토리 구조

```
computer-vision-competition-1SEN/
├── configs/                                 # ⚙️ 설정 관리
│   ├── train_highperf.yaml                  # 메인 고성능 학습 설정
│   ├── infer_highperf.yaml                  # 메인 고성능 추론 설정
│   ├── optuna_single_fold_config.yaml       # 단일 폴드 최적화 설정
│   ├── 20250910/                           # 최적화된 설정 백업 (20개)
│   │   ├── train_optimized_*_1213.yaml     # 🏆 F1 0.98362 달성 설정
│   │   └── ...
│   └── 20250911/                           # 최신 실험 설정들
│
├── src/                                     # 🧠 핵심 소스 코드 (54개 파일)
│   ├── training/                            # 학습 시스템
│   │   ├── train_main.py                    # 🚀 통합 실행 인터페이스
│   │   ├── train_highperf.py                # 고성능 학습 (Mixup, Hard Aug)
│   │   └── train.py                         # 기본 학습
│   ├── inference/                           # 추론 시스템
│   │   ├── infer_main.py                    # 추론 실행 인터페이스
│   │   ├── infer_highperf.py                # 고성능 TTA 앙상블 추론
│   │   └── infer_calibrated.py              # 캘리브레이션 추론
│   ├── optimization/                        # 하이퍼파라미터 최적화
│   │   ├── optuna_tuner.py                  # Optuna 캐싱된 자동 튜닝
│   │   ├── hyperopt_utils.py                # 최적화 유틸리티
│   │   └── test_*.py                        # 최적화 테스트 모듈들
│   ├── models/                              # 모델 아키텍처
│   │   └── build.py                         # 다중 모델 빌드 시스템 (10개 모델)
│   ├── data/                                # 데이터 처리
│   │   ├── dataset.py                       # HighPerfDocClsDataset, Mixup
│   │   └── transforms.py                    # 고급 증강 (Essential/Comprehensive TTA)
│   ├── pipeline/                            # 통합 파이프라인
│   │   └── full_pipeline.py                 # 전체 파이프라인 오케스트레이션
│   ├── logging/                             # 로깅 시스템
│   │   ├── wandb_logger.py                  # WandB 통합 로거
│   │   └── logger.py                        # 기본 로거
│   ├── metrics/                             # 평가 메트릭
│   │   └── f1.py                           # F1 스코어 계산
│   ├── calibration/                         # 모델 캘리브레이션
│   └── utils/                               # 유틸리티 (23개 파일)
│       ├── config/                          # 설정 관리
│       ├── gpu_optimization/                # GPU 최적화
│       ├── visualizations/                  # 시각화 시스템
│       └── core/                           # 핵심 유틸리티
│
├── experiments/                             # 📊 실험 결과 저장
│   ├── train/20250910/20250910_1213_*/     # 🏆 F1 0.98362 실험 결과
│   ├── optimization/                        # Optuna 최적화 결과
│   └── infer/                              # 추론 결과
│
├── logs/                                    # 📝 로그 파일들 (날짜별 정리)
│   └── 20250910/train/                     # 최고 성능 달성 로그
│
├── docs/                                    # 📚 포괄적 문서화
│   ├── FAQ/                                # 질문 대응 FAQ (F1 0.98362 관련)
│   ├── 학습결과/                           # ConvNeXt 최고성능 분석 보고서
│   ├── 시스템/                             # 파이프라인 비교분석, 시각화 가이드
│   ├── 최적화/                             # Optuna 최적화 전략 분석
│   ├── 파이프라인/                         # 학습/추론 파이프라인 가이드
│   ├── 대회전략분석/                       # 경진대회 전략 문서
│   └── 모델/                               # 모델 아키텍처 가이드
│
├── submissions/                             # 🎯 제출 파일들 (날짜별)
├── wandb/                                   # 📈 WandB 실험 추적 (100+ 실험)
├── notebooks/                               # 📔 Jupyter 노트북
│   ├── modular/                            # 모듈형 분석 노트북
│   └── team/                               # 팀별 실험 노트북
└── scripts/                                # 🔧 유틸리티 스크립트
```

### 🔧 핵심 기술 스택

| 구분 | 기본 파이프라인 | **고성능 파이프라인** |
|-----|--------------|---------------------|
| **모델** | EfficientNet B3 | **ConvNeXt Base 384** (ImageNet-22k) |
| **검증 전략** | 5-Fold CV | **단일 폴드** (80:20) + 앙상블 |
| **데이터 증강** | 기본 증강 | **Hard Augmentation + Mixup** |
| **최적화** | 기본 Optuna | **캐싱된 단일 폴드 최적화** (2초/trial) |
| **추론** | 단일 예측 | **TTA 앙상블** (Essential/Comprehensive) |
| **모니터링** | 기본 로깅 | **WandB 통합** + 실시간 시각화 |
| **성능** | F1 ~0.93 | **F1 0.98362** ⭐ |
| **실행 시간** | 2-3시간 | **40분** (최적화 포함) |

---

## 🎯 주요 기능 상세

### 1. 🧠 지원 모델 아키텍처 (10개)

```python
# Vision Transformers
"swin_base_384"      # Swin Transformer Base 384
"vit_large"          # Vision Transformer Large 384
"deit_base"          # DeiT Base 384

# CNN 아키텍처  
"convnext_base_384"  # ConvNeXt Base 384 (최고 성능) ⭐
"convnext_large"     # ConvNeXt Large
"efficientnet_b3"    # EfficientNet B3
"efficientnet_v2_b3" # EfficientNet V2 B3
"resnet50"           # ResNet-50
```

### 2. ⚡ 단일 폴드 vs K-Fold 지원

```yaml
# 단일 폴드 모드 (경진대회 최적화) - 6배 빠름
data:
  folds: 1                    # 단일 폴드 활성화
  stratify: true              # 계층적 분할 (80:20)

# K-Fold 교차검증 모드 (안정성 우선)
data:
  folds: 5                    # K-Fold 활성화 (2 이상)
  valid_fold: 0               # 현재 검증 폴드
```

### 3. 🔬 Optuna 하이퍼파라미터 최적화

```python
# 자동 최적화 파라미터
search_space = {
    "learning_rate": (1e-6, 1e-2),      # 로그 균등 분포
    "weight_decay": (1e-4, 1e-1),       # 정규화 강도
    "dropout": (0.0, 0.3),              # 드롭아웃 비율
    "batch_size": [8, 16, 32, 64],      # 배치 크기
    "mixup_alpha": (0.1, 1.0),          # Mixup 강도
}

# 최적화 전략
- TPE Sampler: 베이지안 최적화
- Median Pruner: 조기 종료
- 캐싱 시스템: 150배 속도 향상
```

### 4. 🎨 고급 데이터 증강

```python
# Hard Augmentation (에폭별 강도 조절)
- HorizontalFlip: 50% 확률
- RandomRotation: ±15도
- ColorJitter: 밝기/대비/채도 조절
- GaussianBlur: 가우시안 블러
- ShiftScaleRotate: 복합 변환

# Mixup 데이터 증강
- 두 이미지 선형 결합
- 라벨도 동일 비율로 혼합
- 과적합 방지 효과
```

### 5. 🔮 고성능 TTA 추론

```python
# Essential TTA (5가지) - 빠른 추론
tta_transforms = [
    "original", "horizontal_flip", 
    "vertical_flip", "rotate_90", "rotate_180"
]

# Comprehensive TTA (15가지) - 최고 성능
tta_transforms = [
    "original", "horizontal_flip", "vertical_flip",
    "rotate_90", "rotate_180", "rotate_270",
    "scale_0.9", "scale_1.1", "brightness_0.9", 
    "brightness_1.1", "contrast_0.9", "contrast_1.1",
    "gaussian_blur", "sharpen", "random_crop"
]
```

---

## 📊 성능 벤치마크 및 최적화 결과

### 🏆 F1 Score 0.98362 달성 설정

```yaml
# Optuna 최적화된 하이퍼파라미터
train:
  lr: 0.00012802227271884058          # 최적 학습률
  weight_decay: 0.013163367232645818  # 균형잡힌 정규화
  dropout: 0.10286340155629473        # 최적 드롭아웃
  batch_size: 16                      # 메모리 효율적
  epochs: 150                         # 적절한 학습 길이
  mixup_alpha: 0.8                    # Mixup 강도
  use_mixup: true                     # Mixup 활성화
  use_advanced_augmentation: true     # 고급 증강
  use_ema: true                       # EMA 안정화
  temperature_scaling: true           # 캘리브레이션
```

### ⚡ 실행 모드별 성능 비교

| 실행 명령어 | 시간 | 예상 F1 | GPU 메모리 | 추천 상황 |
|------------|------|---------|------------|-----------|
| `--mode basic` | 30분 | 0.920-0.930 | 8GB | 빠른 프로토타입 |
| `--mode highperf` | 2시간 | 0.950-0.965 | 16GB | 고품질 실험 |
| **🚀 단일 폴드 최적화** | **40분** | **0.98362** | **12GB** | **⚡ 경진대회용** |

### 📈 모델별 성능 비교

| 모델 | F1 Score | 학습시간 | 추론속도 | 안정성 | 비고 |
|------|----------|----------|----------|--------|------|
| ConvNeXt Base 384 | **0.98362** | 52분 | 28ms/img | ⭐⭐⭐⭐⭐ | 최고 성능 |
| ConvNeXt Large | 0.9712 | 125분 | 52ms/img | ⭐⭐⭐ | 고성능, 느림 |
| Swin Base 384 | 0.9487 | 63분 | 35ms/img | ⭐⭐⭐⭐ | Transformer |
| EfficientNet V2 B3 | 0.9524 | 45분 | 23ms/img | ⭐⭐⭐ | 경량 모델 |

---

## 🚀 고급 사용법

### 1. 실험 추적 및 재현

```bash
# 특정 고성능 실험 재현
python src/training/train_main.py \
    --config configs/20250910/train_optimized_*_1213.yaml \
    --mode full-pipeline \
    --seed 42

# WandB 프로젝트 지정
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline \
    --wandb-project document-classification-highperf
```

### 2. 커스텀 Optuna 최적화

```bash
# 더 많은 Trial과 긴 타임아웃
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --optimize \
    --optuna-config configs/optuna_single_fold_config.yaml \
    --n-trials 100 \
    --timeout 7200
```

### 3. 다양한 TTA 추론

```bash
# Essential TTA (빠른 추론)
python src/training/train_main.py \
    --config configs/infer_highperf.yaml \
    --mode infer \
    --tta essential

# Comprehensive TTA (최고 품질)
python src/training/train_main.py \
    --config configs/infer_highperf.yaml \
    --mode infer \
    --tta comprehensive
```

### 4. 앙상블 추론

```bash
# 여러 모델 앙상블
python src/training/train_main.py \
    --config configs/infer_highperf.yaml \
    --mode infer \
    --ensemble-models experiments/train/20250910/*/ckpt/best_*.pth
```

---

## 🛠️ 트러블슈팅

### 일반적인 문제들

#### 1. GPU 메모리 부족
```bash
# 배치 크기 자동 조정
python src/utils/auto_batch_size.py --config configs/train_highperf.yaml

# 수동 배치 크기 조정
# configs/train_highperf.yaml 수정
train:
  batch_size: 32  # 기본값에서 줄이기: 90 → 64 → 32 → 16
```

#### 2. 학습 중단 후 재개
```bash
# 체크포인트에서 자동 재개
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --resume
```

#### 3. Optuna 최적화 오류
```bash
# 단일 폴드 테스트 실행
python src/optimization/test_single_fold_quick.py

# Optuna 최적화 테스트
python src/optimization/test_optuna_single_fold.py
```

### 로그 확인

```bash
# 최신 학습 로그 확인
tail -f logs/$(date +%Y%m%d)/train/*.log

# 최신 최적화 로그 확인
tail -f logs/optimization/optuna_*.log

# WandB 실험 확인
wandb sync wandb/latest-run
```

---

## 📚 상세 문서

### 📖 핵심 가이드
- [🎓 학습 파이프라인 가이드](docs/파이프라인/학습_파이프라인_가이드.md) - 고성능 학습 과정 상세
- [🔮 추론 파이프라인 가이드](docs/파이프라인/추론_파이프라인_가이드.md) - TTA 앙상블 추론 설명
- [🌟 전체 파이프라인 가이드](docs/파이프라인/전체_파이프라인_가이드.md) - End-to-End 워크플로우

### 🔧 기술 문서
- [⚙️ 기본 vs 고성능 파이프라인 비교](docs/시스템/기본_vs_고성능_파이프라인_비교분석.md) - 상세 성능 비교
- [🔧 모델 설정 가이드](docs/모델/모델_설정_가이드.md) - 10개 모델 구성 및 설정
- [⚡ GPU 최적화 가이드](docs/최적화/GPU_최적화_가이드.md) - 메모리 최적화 전략

### 📊 분석 문서
- [🏆 ConvNeXt 최고성능 분석](docs/학습결과/ConvNeXt_최고성능_학습결과_분석_20250910.md) - F1 0.98362 달성 분석
- [📈 경진대회 최적 전략](docs/대회전략분석/경진대회_최적학습전략_비교분석_20250910.md) - 단일 폴드 vs K-Fold 비교
- [🔬 Optuna 최적화 전략](docs/최적화/Optuna_최적화_효과_및_전략분석.md) - 하이퍼파라미터 최적화 가이드

### 🤔 FAQ 및 질문 대응
- [💬 전문가 질문 대응 FAQ](docs/FAQ/질문_대응_FAQ.md) - F1 0.98362 관련 실증적 답변

---

## 🤝 Contributing / 기여하기

1. 저장소를 포크하세요
2. 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 열어주세요

---

## 🙏 Acknowledgments

- **ConvNeXt Base 384**: F1 0.98362 달성의 핵심 모델
- **Optuna**: 하이퍼파라미터 최적화 프레임워크
- **단일 폴드 최적화**: 경진대회를 위한 고속 최적화 전략  
- **데이터셋 캐싱**: 매 trial 2초 달성의 핵심 기술
- **WandB**: 100+ 실험 추적 및 시각화

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](../../issues)
- **Wiki**: [프로젝트 Wiki](../../wiki)
- **Docs**: `docs/` 폴더 내 포괄적 문서들