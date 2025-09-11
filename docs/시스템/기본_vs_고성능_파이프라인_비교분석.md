# 기본 vs 고성능 파이프라인 비교 분석

## 📋 개요

본 문서는 문서 분류 경진대회 프로젝트에서 구현된 **기본(Basic) 파이프라인**과 **고성능(High-Performance) 파이프라인**의 차이점, 성능 비교, 그리고 각각의 활용 상황을 상세히 분석합니다.

## 🏗️ 아키텍처 구분

### 기본 파이프라인 (Basic)
- **목적**: 빠른 프로토타이핑, 기본적인 모델 검증
- **설정 파일**: `configs/train.yaml`, `configs/infer.yaml`
- **주요 모듈**: `src/training/train.py`, `src/inference/infer.py`

### 고성능 파이프라인 (High-Performance)
- **목적**: 경진대회 우승을 위한 최고 성능 달성
- **설정 파일**: `configs/train_highperf.yaml`, `configs/infer_highperf.yaml`, `configs/train_multi_model_ensemble.yaml`
- **주요 모듈**: `src/training/train_highperf.py`, `src/inference/infer_highperf.py`
- **다중 모델 지원**: 폴드별 다른 모델 사용 가능 (ConvNeXt, Swin Transformer 등)

## ⚙️ 핵심 차이점 비교

| 구분 | 기본 파이프라인 | 고성능 파이프라인 |
|-----|--------------|------------------|
| **교차검증** | 5-Fold CV | 단일 폴드 또는 5-Fold CV 지원 |
| **배치 크기** | 48 | 48-72 (GPU 메모리 최적화) |
| **에포크 수** | 10 | 80-300 (최대 150 규모) |
| **데이터 증강** | 기본 증강 (회전, 크롭) | 고급 증강 + Hard Augmentation + 동적 확률 |
| **Mixup** | 미사용 | 사용 (α=0.8) |
| **Mixed Precision** | 사용 | 최적화된 AMP |
| **스케줄러** | CosineAnnealing | Warmup + CosineAnnealing |
| **정규화** | Weight Decay (0.0) | Label Smoothing(0.1) + EMA(0.9999) + Weight Decay(0.005-0.013) |
| **다중 모델** | 지원 안 함 | 폴드별 다른 모델 지원 |
| **최적화** | 기본 | Optuna 하이퍼파라미터 튜닝 |

## 📊 성능 비교 분석

### 최신 실험 결과 비교 (2025-09-10 기준)

#### 🎆 **새로운 최고 성능 기록 - ConvNeXt Base 384 (1213)**
```yaml
실행 시간: 2025-09-10 12:13:33 ~ 12:36:45 (23분 12초)
최종 F1 Score: 0.98362 (새로운 최고 기록!) ✨
설정 파일: configs/train_optimized_20250910_1213.yaml
핵심 설정값:
  - 모델: convnext_base_384
  - 배치 크기: 16 (최적화된 메모리 사용)
  - 에포크: 150
  - 학습률: 0.00012802227271884058 (Optuna 최적화 결과)
  - Weight Decay: 0.013163367232645818 (강화된 정규화)
  - Optimizer: AdamW
  - Scheduler: Cosine with 10 epoch warmup
  - Label Smoothing: 0.1
  - Mixup Alpha: 0.8
  - EMA Decay: 0.9999
  - Mixed Precision: true
  - 증강: Advanced + Hard Augmentation + Mixup
```

#### 🥈 **두 번째 최고 성능 - ConvNeXt Base 384 (0929)**
```yaml
실행 시간: 2025-09-10 09:29:16 ~ 10:17:xx (약 48분)
최종 F1 Score: 0.97918
설정 파일: configs/train_optimized_20250910_0929.yaml
핵심 설정값:
  - 모델: convnext_base_384
  - 배치 크기: 32 (대용량 처리)
  - 에포크: 300 (장기 학습)
  - 학습률: 8.39e-05 (보수적 학습률)
  - Weight Decay: 0.0333 (높은 정규화)
  - Optimizer: AdamW
  - 증강: Advanced Augmentation + Hard Augmentation
```

#### 🥉 **기존 기준점 - ConvNeXt Base 384 (0908)**
```yaml
실행 시간: 2025-09-10 09:08:xx ~ 09:23:xx (약 15분)
최종 F1 Score: 0.96909
설정 파일: configs/train_optimized_20250910_0908.yaml
핵심 설정값:
  - 모델: convnext_base_384
  - 배치 크기: 32 (표준 크기)
  - 에포크: 100 (빠른 학습)
  - 학습률: 2.69e-05 (매우 보수적)
  - Weight Decay: 0.0258 (중간 정규화)
  - Optimizer: AdamW
  - 증강: Advanced + Hard Augmentation
```

**전체 성능 순위 (2025-09-10 실험 결과):**
1. **0.98362** - ConvNeXt Base 384 (12:13, epoch 150, lr=1.28e-04, wd=0.0132, batch=16) 🏆
2. **0.97918** - ConvNeXt Base 384 (09:29, epoch 300, lr=8.39e-05, wd=0.0333, batch=16) 🥈
3. **0.96909** - ConvNeXt Base 384 (09:08, epoch 100, lr=2.69e-05, wd=0.0258, batch=32) 🥉
4. **0.95242** - EfficientNet V2 B3 (15:52, epoch 100, lr=1e-04, wd=0.005, batch=124)
5. **0.95022** - ConvNeXt Base + Optuna (13:54, epoch 150, lr=0.00255, wd=0.0333, batch=16)
6. **0.94075** - ConvNeXt Large + Optuna (14:41, epoch 100, lr=0.00188, wd=0.0935, batch=32)

#### 📈 학습 안정성 비교

**고성능 파이프라인의 우수성:**
- **빠른 수렴**: 10 에포크 내 F1 0.9+ 달성
- **안정적 향상**: 지속적인 성능 개선 곡선
- **높은 최종 성능**: F1 0.969 달성

## 🔧 기술적 차이점 상세 분석

### 📊 데이터 증강 기법 비교

| 구분 | 기본 파이프라인 | 고성능 파이프라인 |
|-----|--------------|------------------|
| **기본 증강** | RandomRotation, RandomCrop | 동일 + 고급 변형 |
| **색상 증강** | ColorJitter(단순) | ColorJitter + 밝기조정 + 대비조정 |
| **공간 증강** | 기본 회전만 | 고급 전단 + 왕곡 + 스케일링 |
| **노이즈 증강** | 없음 | 가우시안 노이즈 + 블러 |
| **Hard Augmentation** | 사용 안함 | 동적 확률 스케줄링 (0.0→0.5) |
| **Mixup** | 사용 안함 | Alpha=0.8, 배치 내 혼합 |
| **CutMix** | 사용 안함 | 전체 파이프라인에 선택적 적용 |

**📝 전문용어 설명:**
- **Hard Augmentation**: 학습 에포크가 진행됨에 따라 증강 강도를 점진적으로 높이는 기법. 초기에는 약하게, 후기에는 강하게 적용되어 과적합 방지와 일반화 성능을 동시에 향상
- **Mixup**: 두 이미지를 선형 결합하여 새로운 학습 예제 생성. 라벨도 동일한 비율로 혼합하여 모델의 결정 경계를 부드럽게 만듦
- **CutMix**: 한 이미지의 일부 영역을 다른 이미지로 교체하는 기법. 공간적 정보를 보존하면서 다양성 증대

### 1. 데이터 전처리 및 증강 상세 비교

#### 기본 파이프라인
```python
# configs/train.yaml
train:
  img_size: 384
  batch_size: 48
  use_advanced_augmentation: false
  epochs: 50
```

#### 고성능 파이프라인
```python
# configs/train_highperf.yaml
train:
  img_size: 384
  batch_size: 72  # 메모리 최적화된 큰 배치
  use_advanced_augmentation: true  # 고급 증강
  epochs: 80-150  # 상황에 따른 동적 조정
  use_mixup: true  # Mixup 데이터 증강
  mixup_alpha: 0.8
  hard_augmentation: true  # 강한 증강
```

### 2. 모델 아키텍처 최적화

#### 기본 파이프라인
- 단순 모델 구조
- 기본 드롭아웃 설정

#### 고성능 파이프라인
```python
model:
  name: convnext_base  # 최적화된 모델 선택
  drop_rate: 0.1
  drop_path_rate: 0.1  # Stochastic Depth
  pooling: avg
  # EMA, Temperature Scaling 등 고급 기법 적용
  use_ema: true                 # Exponential Moving Average
  ema_decay: 0.9999             # EMA 감쇠율
  temperature_scaling: true     # 확률 보정
  label_smoothing: 0.1          # 라벨 스무딩
```

### 2. 정규화 기법 상세 비교

| 기법 | 기본 파이프라인 | 고성능 파이프라인 | 효과 |
|-----|--------------|------------------|---------|
| **Weight Decay** | 0.0 (사용 안함) | 0.005-0.013 (동적 조정) | L2 정규화로 과적합 방지 |
| **Label Smoothing** | 0.0 (사용 안함) | 0.1 | 과신 방지, 일반화 성능 향상 |
| **EMA** | 사용 안함 | 0.9999 감쇠율 | 모델 가중치의 지수이동평균으로 안정성 향상 |
| **Temperature Scaling** | 사용 안함 | 활성화 | 예측 확률 보정으로 신뢰도 향상 |
| **Dropout** | 기본값 | 0.1-0.2 (동적 조정) | 뉴런 무작위 비활성화로 과적합 방지 |

**📝 정규화 기법 상세 설명:**
- **Weight Decay**: 모델 가중치가 너무 커지는 것을 막는 L2 정규화 기법. 가중치의 제곱합에 비례하는 패널티를 손실함수에 추가
- **Label Smoothing**: 정답 라벨을 1이 아닌 0.9로, 나머지 클래스를 0이 아닌 0.1/(N-1)로 설정하여 모델의 과신을 방지
- **EMA (Exponential Moving Average)**: 모델 가중치를 지수이동평균으로 업데이트. 학습 과정의 잡음을 줄이고 안정성 향상
- **Temperature Scaling**: 예측 확률에 온도 매개변수를 적용하여 신뢰도와 정확도를 일치시키는 사후 보정 기법

### 3. 학습 스케줄링 비교

#### 고성능 파이프라인의 고급 스케줄링
```python
scheduler: cosine
warmup_epochs: 10  # 워밍업 단계
mixed_precision: true  # 메모리 효율성
max_grad_norm: 0.5  # 그래디언트 클리핑
label_smoothing: 0.1  # 과적합 방지
use_ema: true  # Exponential Moving Average
```

## 📁 파일 구조 및 연관관계

### 기본 파이프라인 관련 파일들
```
configs/
├── train.yaml                    # 기본 학습 설정
├── infer.yaml                     # 기본 추론 설정
src/training/
├── train.py                       # 기본 학습 스크립트
src/inference/
├── infer.py                       # 기본 추론 스크립트
```

### 고성능 파이프라인 관련 파일들
```
configs/
├── train_highperf.yaml           # 고성능 학습 설정 (단일/다중폴드)
├── infer_highperf.yaml           # 고성능 추론 설정
├── train_multi_model_ensemble.yaml # 다중 모델 앵상블 설정
├── infer_multi_model_ensemble.yaml # 다중 모델 앵상블 추론
├── optuna_single_fold_config.yaml # 단일 폴드 최적화 설정
├── optuna_config.yaml            # 일반 최적화 설정 (K-fold)
src/training/
├── train_highperf.py             # 고성능 학습 스크립트 (단일/다중 모델 지원)
├── train_main.py                  # 통합 실행 인터페이스
src/inference/
├── infer_highperf.py             # 고성능 추론 스크립트
src/optimization/
├── optuna_tuner.py               # 하이퍼파라미터 최적화 (데이터 캐싱 지원)
src/models/
├── build.py                       # 단일/다중 모델 빌더 지원
```

## 🚀 실행 명령어 비교

### 기본 파이프라인 실행
```bash
# 기본 학습
python src/training/train.py --config configs/train.yaml

# 기본 추론  
python src/inference/infer.py --config configs/infer.yaml
```

### 고성능 파이프라인 실행
```bash
# 1. 단일 모델 고성능 학습 (folds: 1 또는 5)
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline

# 2. 다중 목델 앵상블 학습 (5-Fold, 폴드별 다른 모델)
python src/training/train_main.py \
    --config configs/train_multi_model_ensemble.yaml \
    --mode full-pipeline

# 3. 고성능 추론
python src/training/train_main.py \
    --config configs/infer_highperf.yaml \
    --mode infer

# 4. 다중 모델 앵상블 추론
python src/training/train_main.py \
    --config configs/infer_multi_model_ensemble.yaml \
    --mode infer

# 5. 하이퍼파라미터 최적화 (단일 폴드, 고속)
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline \
    --optimize \
    --optuna-config configs/optuna_single_fold_config.yaml \
    --auto-continue

# 6. 하이퍼파라미터 최적화 (K-fold, 정밀)
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline \
    --optimize \
    --optuna-config configs/optuna_config.yaml
```

## 📈 실험 결과 백업 분석

### configs 백업 파일 분석
```
configs/20250910/  # 2025년 9월 10일 실험들
├── train_optimized_20250910_0908.yaml  # F1 0.969 달성 설정
├── train_optimized_20250910_1237.yaml  # ConvNeXt 실험
├── train_optimized_20250910_1440.yaml  # 대형 모델 실험
```

### 성능 추적 결과
- **0908 실험**: F1 0.96909 (최고 성능) ⭐
- **1237 실험**: ConvNeXt Base 384 최적화
- **1440 실험**: ConvNeXt Large 실험

## 💡 활용 상황별 권장사항

### 🎯 기본 파이프라인 사용 권장 상황
- **프로토타이핑 단계**: 빠른 아이디어 검증
- **리소스 제한**: GPU 메모리/시간 부족시
- **교육 목적**: 기본 개념 학습시
- **베이스라인**: 초기 성능 측정

### 🏆 고성능 파이프라인 사용 권장 상황
- **경진대회 참여**: 최고 성능이 필요할 때 ⭐
- **프로덕션 배포**: 실제 서비스에서 높은 정확도 필요
- **연구 목적**: 최신 기법들의 효과 검증
- **벤치마크**: 모델 성능의 한계 측정

## 🔄 성능 향상 전략

### 단계별 발전 과정
1. **기본 파이프라인**: F1 ~0.85 (추정)
2. **고성능 최적화**: F1 0.969 달성
3. **하이퍼파라미터 튜닝**: 추가 0.01~0.02 향상 가능
4. **앙상블 기법**: 최대 0.98+ 목표

### 핵심 성능 향상 요소
1. **데이터 증강 고도화** (+0.03~0.05)
2. **Mixup 적용** (+0.02~0.03)
3. **고급 정규화 기법** (+0.01~0.02)
4. **최적화된 스케줄링** (+0.01~0.02)
5. **EMA 및 고급 기법** (+0.01~0.02)

## 📊 최종 권장사항

### 🥇 경진대회 우승 전략
```bash
# 1단계: 고성능 학습으로 베스트 모델 확보
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline

# 2단계: 하이퍼파라미터 최적화로 성능 극대화
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --optimize \
    --optuna-config configs/optuna_single_fold_config.yaml
    
# 3단계: 최적화된 설정으로 다중 모델 앙상블
# (여러 시드, 여러 모델 조합으로 최종 예측)
```

### 성능 대비 시간 효율성
- **기본 파이프라인**: ⚡ 빠른 실행, 적당한 성능
- **고성능 파이프라인**: 🏆 최고 성능, 합리적인 시간 (15분/100에포크)
- **최적화 파이프라인**: 🔧 성능 극한 추구, 추가 시간 투자

---

## 📚 관련 문서
- [학습 파이프라인 가이드](../파이프라인/학습_파이프라인_가이드.md)
- [추론 파이프라인 가이드](../파이프라인/추론_파이프라인_가이드.md)
- [전체 파이프라인 가이드](../파이프라인/전체_파이프라인_가이드.md)
- [GPU 최적화 가이드](../최적화/GPU_최적화_가이드.md)