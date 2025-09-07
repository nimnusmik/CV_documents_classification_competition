# 🔍 Optuna & Temperature Scaling 사용 가이드

## 🎯 새로 추가된 기능

### 1. Optuna 하이퍼파라미터 자동 최적화
- **목적**: 학습률, 배치 크기, weight decay 등을 자동으로 최적화
- **효과**: 수동 튜닝 대비 1-3% F1 점수 향상 기대
- **시간**: 20번 시도 시 약 30분-1시간 소요

### 2. Temperature Scaling 확률 캘리브레이션  
- **목적**: 모델의 과신(overconfidence) 문제 해결
- **효과**: 앙상블 예측 시 더 정확한 확률 계산으로 0.5-1% F1 점수 향상
- **시간**: 추가 시간 거의 없음 (기존 추론에 캘리브레이션만 추가)

## 🚀 사용 방법

### 📋 1. 패키지 설치
```bash
# Optuna 설치
pip install optuna

# 또는 전체 requirements 재설치
pip install -r requirements.txt
```

### 🔍 2. Optuna 하이퍼파라미터 최적화

#### **기본 최적화 (20번 시도)**
```bash
python src/training/train_main.py --config configs/train_highperf.yaml --optimize
```

#### **더 많은 시도로 정밀 최적화**
```bash
python src/training/train_main.py --config configs/train_highperf.yaml --optimize --n-trials 50
```

#### **최적화 결과 확인**
```bash
# 최적화 완료 후 생성되는 파일들:
# - configs/train_optimized_YYYYMMDD_HHMM.yaml  (최적 설정)
# - experiments/optimization/best_params_*.yaml  (최적 파라미터)
# - logs/optimization/optuna_*.log               (최적화 로그)
```

### 🌡️ 3. Temperature Scaling 캘리브레이션

#### **캘리브레이션 적용 전체 파이프라인**
```bash
python src/training/train_main.py --config configs/train_highperf.yaml --mode full-pipeline --use-calibration
```

#### **수동으로 캘리브레이션 추론만 실행**
```bash
python src/inference/infer_calibrated.py configs/infer_highperf.yaml experiments/train/20250907/swin_base_384/fold_results.yaml
```

### 🎯 4. 전체 최적화 워크플로우

#### **Step 1: 하이퍼파라미터 최적화**
```bash
# 1. 최적 파라미터 탐색 (30분-1시간)
python src/training/train_main.py --config configs/train_highperf.yaml --optimize --n-trials 30

# 2. 생성된 최적 설정 확인
ls configs/train_optimized_*.yaml
```

#### **Step 2: 최적 설정으로 전체 학습 + 캘리브레이션 추론**
```bash
# 최적화된 설정으로 전체 파이프라인 실행 (캘리브레이션 포함)
python src/training/train_main.py --config configs/train_optimized_20250907_1430.yaml --mode full-pipeline --use-calibration
```

## 📊 성능 개선 효과

### **기존 파이프라인**
```bash
# 기본 실행
python src/training/train_main.py --config configs/train_highperf.yaml --mode full-pipeline

# 예상 결과: F1 Score ~0.920
```

### **최적화 적용 후**
```bash
# Optuna + Temperature Scaling 적용
python src/training/train_main.py --config configs/train_optimized_*.yaml --mode full-pipeline --use-calibration

# 예상 결과: F1 Score ~0.935-0.945 (1.5-2.5% 향상!)
```

## ⚙️ 설정 커스터마이징

### **Optuna 설정 조정 (`configs/optuna_config.yaml`)**
```yaml
optuna:
  n_trials: 30              # 시도 횟수 (더 많이 = 더 정확, 더 오래)
  timeout: 7200             # 최대 시간 (초) - 2시간
  
search_space:
  learning_rate:
    low: 1.0e-5             # 학습률 최소값
    high: 1.0e-2            # 학습률 최대값
  batch_size:
    choices: [32, 64, 128]  # 시도할 배치 크기들
```

### **빠른 테스트용 설정**
```bash
# 빠른 테스트 (5번만 시도)
python src/training/train_main.py --config configs/train.yaml --optimize --n-trials 5
```

## 🎯 권장 사용 시나리오

### **🥇 대회 최종 제출용 (최고 성능)**
```bash
# 1단계: 충분한 탐색으로 최적 파라미터 찾기
python src/training/train_main.py --config configs/train_highperf.yaml --optimize --n-trials 50

# 2단계: 최적 설정으로 캘리브레이션 포함 전체 파이프라인
python src/training/train_main.py --config configs/train_optimized_*.yaml --mode full-pipeline --use-calibration
```

### **🚀 빠른 개선 (시간 제한 시)**
```bash
# 빠른 최적화 (10번 시도) + 캘리브레이션
python src/training/train_main.py --config configs/train_highperf.yaml --optimize --n-trials 10
python src/training/train_main.py --config configs/train_optimized_*.yaml --mode full-pipeline --use-calibration
```

### **🔬 실험용 (베이스라인 비교)**
```bash
# 기존 방식
python src/training/train_main.py --config configs/train_highperf.yaml --mode full-pipeline

# 새 방식  
python src/training/train_main.py --config configs/train_highperf.yaml --optimize --n-trials 20
python src/training/train_main.py --config configs/train_optimized_*.yaml --mode full-pipeline --use-calibration

# 결과 비교
```

## 🚨 주의사항

1. **첫 실행 시 Optuna 설치 필요**: `pip install optuna`
2. **최적화는 시간이 오래 걸림**: 20번 시도 시 30분-1시간
3. **GPU 메모리 부족 시**: `configs/optuna_config.yaml`에서 `batch_size` 선택지를 줄이기
4. **캘리브레이션은 추론에만 영향**: 학습 시간은 동일하나 추론 시 약간의 추가 시간

## 📈 예상 성능 향상

| 방법 | 예상 F1 Score | 추가 시간 |
|------|---------------|-----------|
| 기존 파이프라인 | 0.920 | - |
| + Optuna | 0.932 (+1.2%) | +30분-1시간 |
| + Temperature Scaling | 0.935 (+0.3%) | +2분 |
| **+ 둘 다 적용** | **0.940 (+2.0%)** | **+30분-1시간** |

## 🔗 관련 파일

- `src/optimization/` - Optuna 최적화 모듈
- `src/calibration/` - Temperature Scaling 캘리브레이션 모듈  
- `src/inference/infer_calibrated.py` - 캘리브레이션 적용 추론
- `configs/optuna_config.yaml` - Optuna 설정
- `logs/optimization/` - 최적화 로그
- `experiments/optimization/` - 최적화 결과
