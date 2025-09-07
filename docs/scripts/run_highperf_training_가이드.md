# 🏆 run_highperf_training.sh 가이드

## 개요
최종 제출용 고성능 학습을 위한 스크립트입니다. 최고 품질의 모델을 얻기 위해 1-2시간의 충분한 학습 시간을 투자합니다.

## 주요 특징
- 🎯 **최고 성능**: F1 스코어 0.934+ 목표
- 🔬 **완전한 탐색**: 20회 시도로 철저한 하이퍼파라미터 최적화
- 📈 **고급 기능**: Hard Augmentation, Mixup, 고해상도 이미지
- 🏅 **최종 제출용**: 경진대회 최종 제출에 적합한 품질

## 설정 파라미터

### 학습 설정 (train_highperf.yaml)
```yaml
# 고성능을 위한 최적화된 설정
train:
  img_size: 384                    # 고해상도 이미지로 세밀한 특징 추출
  batch_size: 32                   # 안정적인 학습을 위한 적절한 배치 크기
  epochs: 8                        # 충분한 학습을 위한 에포크 수
  use_advanced_augmentation: false # 기본 증강 (현재 최적화된 설정)
  hard_augmentation: true          # 강한 데이터 증강
  
data:
  num_folds: 5                     # 5-fold CV로 안정적인 검증

model:
  name: swin_base_384              # 384px 최적화 Swin Transformer
  drop_path_rate: 0.1              # Stochastic Depth for regularization
```

### 최적화 설정
```yaml
# 철저한 하이퍼파라미터 탐색
n_trials: 20                       # 20회 시도로 최적해 탐색
timeout: 7200                      # 2시간 타임아웃
pruning:
  enabled: true
  patience: 3                      # 안정적인 학습을 위한 여유
```

## 사용법

### 기본 실행
```bash
# 프로젝트 루트에서 실행
./scripts/run_highperf_training.sh
```

### GPU 상태 확인
```bash
# 실행 전 GPU 메모리 확인
nvidia-smi

# 권장 최소 사양
# - VRAM: 8GB 이상
# - System RAM: 16GB 이상
# - Storage: 50GB 여유 공간
```

### 실행 과정
```
🚀 Starting High-Performance Training Pipeline
===============================================

📊 GPU 상태 확인:
NVIDIA GeForce RTX 4090, 24564, 1024, 23540

🎯 실행 중인 설정:
- 모델: Swin Transformer Base (384px)
- Hard Augmentation + Mixup
- WandB 로깅 활성화
- 5-Fold Cross Validation

[고성능 학습 시작...]
Trial 1/20: F1=0.918, lr=0.0001, batch_size=32
Trial 2/20: F1=0.925, lr=0.0002, batch_size=48
...
Trial 15/20: F1=0.934, lr=0.00015, batch_size=40
Best so far: F1=0.934

✅ 고성능 학습 완료!
최종 F1 스코어: 0.934
총 실행 시간: 1시간 45분 22초
```

## 고성능 최적화 전략

### 1. 고해상도 이미지 처리
```python
# 384px vs 224px 비교
384px 장점:
- 세밀한 텍스트 인식 가능
- 문서 레이아웃 정보 보존
- 작은 객체 검출 개선

처리 시간:
- 224px: 1x (기준)
- 384px: 3x (약 3배 느림)
```

### 2. 고급 데이터 증강
```python
# Hard Augmentation 구성
- Mixup (alpha=0.8): 두 이미지 혼합
- CutMix: 이미지 패치 교체
- AutoAugment: 자동 최적화된 증강
- Heavy ColorJitter: 강한 색상 변화
- Perspective Transform: 원근 변환
```

### 3. 모델 아키텍처 최적화
```yaml
model:
  name: swin_base_384           # 384px 전용 아키텍처
  drop_path_rate: 0.1           # Stochastic Depth
  drop_rate: 0.1                # 일반 Dropout
  pooling: avg                  # 안정적인 평균 풀링
  pretrained: true              # ImageNet 사전학습 가중치
```

### 4. 학습 스케줄링
```python
# 최적화된 학습 스케줄
Optimizer: AdamW
Learning Rate: 0.0001-0.0003 (Optuna 최적화)
Scheduler: Cosine Annealing
Weight Decay: 0.01
Label Smoothing: 0.1
```

## 예상 성능 및 시간

### 실행 시간 분석
```
Phase 1: 환경 설정 및 데이터 로딩    (5분)
Phase 2: 모델 초기화 및 검증        (10분)
Phase 3: Optuna 최적화 (20 trials) (80-100분)
Phase 4: 최종 모델 학습             (15-20분)
Phase 5: 추론 및 제출 파일 생성     (10분)

총 소요 시간: 120-150분 (2-2.5시간)
```

### F1 스코어 예상
- **최소**: 0.925 (안정적인 성능)
- **평균**: 0.930 (일반적인 결과)
- **목표**: 0.934+ (최고 성능)
- **최고**: 0.940+ (완벽한 조건)

## 리소스 요구사항

### GPU 메모리 사용량
```yaml
# 384px, batch_size=32 기준
RTX 4090 (24GB): 사용률 85% (안전)
RTX 3080 (12GB): 사용률 95% (위험)
RTX 3070 (8GB):  메모리 부족 (batch_size=16 권장)
```

### 디스크 사용량
```
모델 체크포인트: ~2GB per trial × 20 = 40GB
로그 파일: ~500MB
WandB 아티팩트: ~1GB
임시 파일: ~2GB
총 필요 공간: ~45GB
```

## 최적화 과정 모니터링

### 실시간 모니터링
```bash
# 별도 터미널에서 실행
./scripts/monitor_training.sh

# WandB 대시보드 확인
# https://wandb.ai/your-project/runs
```

### 로그 분석
```bash
# 최적화 진행 상황 확인
tail -f logs/train/train_highperf_*.log | grep "Trial\|Best"

# GPU 사용률 모니터링
nvidia-smi -l 5

# 시스템 리소스 확인
htop
```

## 하이퍼파라미터 최적화 결과

### 최적 파라미터 예시
```yaml
# Optuna가 찾은 최적 설정 (예시)
best_params:
  lr: 0.00015
  batch_size: 40
  epochs: 12
  drop_rate: 0.08
  weight_decay: 0.015
  mixup_alpha: 0.7
  
# 최종 성능
best_f1_score: 0.9342
validation_accuracy: 0.9456
test_accuracy: 0.9378
```

### Cross-Validation 결과
```
Fold 1: F1=0.9325, Acc=0.9434
Fold 2: F1=0.9356, Acc=0.9467
Fold 3: F1=0.9334, Acc=0.9445
Fold 4: F1=0.9347, Acc=0.9458
Fold 5: F1=0.9338, Acc=0.9451

Average: F1=0.9340, Acc=0.9451
Std Dev: F1=0.0012, Acc=0.0013
```

## 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 감소
# configs/train_highperf.yaml 수정
train:
  batch_size: 16    # 32 → 16으로 감소
  
# 또는 그래디언트 누적 사용
train:
  gradient_accumulation_steps: 2
```

### 학습 시간 단축 필요시
```yaml
# 시도 횟수 감소
n_trials: 10        # 20 → 10으로 감소

# 조기 종료 강화
early_stopping:
  patience: 3       # 5 → 3으로 감소
```

### 성능이 기대에 못 미치는 경우
```yaml
# 에포크 수 증가
train:
  epochs: 15        # 8 → 15로 증가
  
# 고급 증강 활성화
train:
  use_advanced_augmentation: true
```

## 결과 활용

### 최고 성능 모델 선택
```bash
# 최고 성능 체크포인트 확인
find experiments/train/$(date +%Y%m%d) -name "*.pth" | \
xargs ls -la | sort -k5 -nr | head -5
```

### 앙상블 준비
```bash
# 상위 3개 모델로 앙상블
python src/inference/ensemble_inference.py \
  --models model1.pth,model2.pth,model3.pth \
  --weights 0.4,0.35,0.25
```

### 제출 파일 검증
```bash
# 제출 파일 무결성 확인
python src/utils/validate_submission.py \
  submissions/$(date +%Y%m%d)/submission_*_advanced_augmentation.csv
```

## 성능 벤치마크

| 설정 | F1 스코어 | 실행 시간 | GPU 메모리 |
|------|----------|----------|------------|
| Standard | 0.928 | 90분 | 18GB |
| Optimized | 0.934 | 120분 | 20GB |
| Maximum | 0.940 | 180분 | 22GB |

## 다음 단계
```bash
# 결과 분석 및 리포트 생성
python src/utils/generate_report.py --experiment-date $(date +%Y%m%d)

# 최종 제출 파일 준비
cp submissions/$(date +%Y%m%d)/submission_*_advanced_augmentation.csv \
   final_submission.csv
```

> **중요**: 고성능 학습은 상당한 시간과 리소스를 소모하므로, 실행 전 시스템 상태를 충분히 확인하고 다른 중요한 작업이 없는 시간에 실행하는 것을 권장합니다.
