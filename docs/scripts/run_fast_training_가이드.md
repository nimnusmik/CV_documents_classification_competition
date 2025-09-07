# 🚀 run_fast_training.sh 가이드

## 개요
경진대회용 빠른 최적화 학습을 위한 스크립트입니다. 20-30분 내에 경쟁력 있는 결과를 도출하는 것이 목표입니다.

## 주요 특징
- ⚡ **빠른 실행**: 20-30분 내 완료
- 🎯 **경진대회 최적화**: 반복 실험에 적합한 설정
- 📊 **효율적인 탐색**: 8회 시도로 최적 하이퍼파라미터 발견
- 💾 **자동 결과 저장**: submissions 폴더에 자동 저장

## 설정 파라미터

### 학습 설정 (train_fast_optimized.yaml)
```yaml
# 빠른 학습을 위한 최적화된 설정
train:
  img_size: 224                    # 작은 이미지 크기로 빠른 처리
  batch_size: 64                   # 큰 배치 사이즈로 효율성 증대
  epochs: 6                        # 짧은 에포크로 빠른 수렴
  use_advanced_augmentation: false # 기본 증강으로 처리 속도 향상
  
data:
  num_folds: 3                     # 3-fold로 검증 시간 단축

model:
  name: swin_base_224              # 224px 최적화 모델
```

### 최적화 설정 (optuna_fast_config.yaml)
```yaml
# 빠른 하이퍼파라미터 탐색
n_trials: 8                        # 8회 시도 (빠른 탐색)
timeout: 1800                      # 30분 타임아웃
pruning:
  enabled: true
  patience: 1                      # 빠른 조기 종료
```

## 사용법

### 기본 실행
```bash
# 프로젝트 루트에서 실행
./scripts/run_fast_training.sh
```

### 실행 과정
```
🚀 빠른 최적화 학습 시작
목표 시간: 20-30분
설정: train_fast_optimized.yaml + optuna_fast_config.yaml

시작 시간: 2025-09-08 01:00:00

[실행 중...]
- Optuna 최적화 시작
- Trial 1/8: lr=0.0002, batch_size=64, epochs=6
- Trial 2/8: lr=0.0001, batch_size=48, epochs=8
- ...

✅ 빠른 최적화 완료!
총 실행 시간: 0시간 28분 35초

📊 결과 파일 확인:
submissions/20250908/submission_20250908_0128_basic_augmentation.csv

📝 로그 파일:
logs/20250908/train/train_fast_opt_20250908_0100.log
```

## 성능 최적화 전략

### 1. 이미지 크기 최적화
- **224px**: Swin Transformer의 기본 크기로 빠른 처리
- **vs 384px**: 약 3배 빠른 학습 속도
- **성능 손실**: 약 1-2% F1 스코어 감소

### 2. 에포크 수 최적화
- **6 에포크**: 빠른 수렴을 위한 최소 에포크
- **조기 종료**: patience=1로 빠른 중단
- **학습률 스케줄링**: 짧은 에포크에 맞춘 cosine decay

### 3. 증강 전략
```python
# 기본 증강 (빠름)
- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter (약한 강도)
- Normalize

# vs 고급 증강 (느림)
- Mixup, CutMix
- AutoAugment
- Heavy ColorJitter
```

### 4. 배치 크기 최적화
```yaml
# GPU 메모리에 따른 권장 배치 크기
RTX 4090 (24GB): batch_size=64
RTX 3080 (12GB): batch_size=32
GTX 1660 (6GB): batch_size=16
```

## 예상 성능

### 실행 시간
- **최소**: 18분 (모든 trial이 빨리 종료되는 경우)
- **평균**: 25분 (정상적인 탐색)
- **최대**: 30분 (timeout 도달)

### F1 스코어
- **목표**: 0.90+ (경진대회 상위권)
- **평균**: 0.92+ (안정적인 성능)
- **최고**: 0.93+ (운이 좋은 경우)

## 결과 분석

### 최적 하이퍼파라미터 확인
```bash
# Optuna 최적화 결과 확인
grep -A 5 "Best trial" logs/20250908/train/train_fast_opt_*.log

# 예시 출력:
# Best trial: 
#   Value: 0.924
#   Params: 
#     lr: 0.00025
#     batch_size: 48
#     epochs: 7
```

### 제출 파일 검증
```bash
# 제출 파일 형식 확인
head -5 submissions/20250908/submission_*_basic_augmentation.csv

# 예시:
# ID,target
# TEST_000001,5
# TEST_000002,12
# TEST_000003,3
```

## 시간 단축 팁

### 1. 프리컴파일된 환경 사용
```bash
# PyTorch 컴파일 캐시 활성화
export TORCH_COMPILE_CACHE_DIR=./cache
```

### 2. 데이터 로딩 최적화
```yaml
# num_workers 설정
data:
  num_workers: 4  # CPU 코어 수의 절반
```

### 3. 메모리 최적화
```bash
# GPU 메모리 정리 후 실행
nvidia-smi --gpu-reset
./scripts/run_fast_training.sh
```

## 문제 해결

### 메모리 부족 에러
```yaml
# train_fast_optimized.yaml 수정
train:
  batch_size: 32  # 64 → 32로 감소
  mixed_precision: true  # 메모리 절약
```

### 시간 초과 문제
```yaml
# optuna_fast_config.yaml 수정
n_trials: 5      # 8 → 5로 감소
timeout: 1200    # 20분으로 단축
```

### 성능 부족 문제
```yaml
# 에포크 수 증가
train:
  epochs: 8      # 6 → 8로 증가
  
# 이미지 크기 증가 (시간 trade-off)
train:
  img_size: 288  # 224 → 288로 증가
```

## 관련 파일
- `configs/train_fast_optimized.yaml`: 빠른 학습 설정
- `configs/optuna_fast_config.yaml`: 빠른 최적화 설정
- `src/training/train_main.py`: 메인 학습 스크립트
- `scripts/monitor_training.sh`: 진행 상황 모니터링

## 다음 단계
```bash
# 빠른 학습 완료 후 고성능 학습 실행
./scripts/run_highperf_training.sh

# 또는 결과 분석
python src/utils/analyze_results.py --date $(date +%Y%m%d)
```

## 성능 벤치마크

| GPU 모델 | 배치 크기 | 예상 시간 | F1 스코어 |
|---------|----------|----------|-----------|
| RTX 4090 | 64 | 18-22분 | 0.925+ |
| RTX 3080 | 32 | 22-28분 | 0.920+ |
| GTX 1660 | 16 | 28-35분 | 0.915+ |

> **팁**: 첫 실행 시 모델 다운로드로 추가 5-10분 소요될 수 있습니다.
