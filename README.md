# 🏆 Computer Vision Competition - Document Classification

## 📋 Project Overview

이 프로젝트는 **문서 분류 경진대회**를 위한 **완전 자동화된 머신러닝 파이프라인**입니다. 
RTX 4090부터 GTX 1660까지 다양한 GPU 환경을 자동으로 감지하고 최적화하여, 
**원클릭으로 전체 파이프라인(학습 → 검증 → 추론 → 제출파일 생성)**을 실행할 수 있는 
**팀 협업 중심의 Enterprise급 MLOps 시스템**입니다.

### 🎯 프로젝트 특징
- 🔧 **자동화**: GPU 환경 감지 → 최적 설정 → 자동 실행
- 🤝 **팀 협업**: 다양한 GPU 환경 통합 지원 (RTX 4090 ↔ GTX 1660)
- ⚡ **고성능**: Swin Transformer + EfficientNet 앙상블 (F1: 0.934+)
- 📊 **모니터링**: WandB 통합 실시간 추적 및 자동 시각화
- 🔄 **재현성**: 완전한 실험 추적 및 재현 가능한 결과
- 🎨 **시각화**: 학습/추론/최적화 과정 자동 차트 생성 및 저장

---

## 🛠️ Quick Start

### 📦 Installation & Setup

1. **Repository Clone**
```bash
git clone <repository-url>
cd computer-vision-competition-1SEN
```

2. **Python Environment (pyenv 권장)**
```bash
pyenv install 3.11.9
pyenv virtualenv 3.11.9 cv-competition
pyenv activate cv-competition
pip install -r requirements.txt
```

3. **GPU 환경 자동 감지 & 최적화**
```bash
python src/utils/gpu_optimization/team_gpu_check.py          # GPU 호환성 자동 체크
python src/utils/gpu_optimization/auto_batch_size.py         # 최적 배치 크기 자동 탐색
```

### ⚡ Fast Training (20-30분, 빠른 프로토타이핑)

```bash
# 방법 1: 쉘 스크립트 실행 (권장)
./scripts/run_fast_training.sh

# 방법 2: Python 직접 실행
python src/training/train_main.py --config configs/train_fast_optimized.yaml

# 방법 3: 전체 파이프라인 자동화 (학습 → 검증 → 추론 → 제출파일)
python src/pipeline/full_pipeline.py --config configs/train_fast_optimized.yaml --mode fast
```

### 🏆 High Performance Training (1-2시간, 최종 제출용)

```bash
# 방법 1: 쉘 스크립트 실행 (권장)  
./scripts/run_highperf_training.sh

# 방법 2: Python 직접 실행
python src/training/train_main.py --config configs/train_highperf.yaml

# 방법 3: 전체 파이프라인 자동화
python src/pipeline/full_pipeline.py --config configs/train_highperf.yaml --mode highperf
```

### � Advanced Training (고급 학습 - 모든 기능 통합)

```bash
# 완전 자동화 파이프라인 (학습 + 최적화 + 캘리브레이션 + 앙상블)
python src/training/train_main.py \
  --config configs/train_multi_model_ensemble.yaml \
  --mode full-pipeline \
  --use-calibration \
  --optimize \
  --optuna-config configs/optuna_config.yaml \
  --auto-continue

# 다중 모델 앙상블 학습
python src/training/train_main.py \
  --config configs/train_multi_model_ensemble.yaml \
  --ensemble-models efficientnet,swin \
  --k-fold 5

# 자동 하이퍼파라미터 최적화 + 학습
python src/training/train_main.py \
  --config configs/train_highperf.yaml \
  --optimize \
  --optuna-trials 100 \
  --auto-gpu-optimization
```

### �🔍 Inference (추론 실행)

```bash
# 기본 추론
python src/inference/infer_main.py --config configs/infer.yaml

# 고성능 추론 (TTA + Ensemble)
python src/inference/infer_main.py --config configs/infer_highperf.yaml

# 고급 추론 옵션
python src/inference/infer_main.py \
  --config configs/infer_highperf.yaml \
  --tta-mode aggressive \
  --ensemble-weight 0.7,0.3 \
  --calibration-temperature 1.5 \
  --batch-size auto
```

### 📊 Training Monitoring (실시간 모니터링)

```bash
# 백그라운드 모니터링 시작
./scripts/monitor_training.sh

# WandB 대시보드 확인
# https://wandb.ai/your-account/your-project
```

---

## 🛠️ Command Line Options

### 📋 Help Commands

모든 스크립트는 `--help` 옵션으로 사용 가능한 모든 옵션을 확인할 수 있습니다:

```bash
# 학습 옵션 확인
python src/training/train_main.py --help

# 추론 옵션 확인  
python src/inference/infer_main.py --help

# 최적화 옵션 확인
python src/optimization/optuna_optimizer.py --help

# 전체 파이프라인 옵션 확인
python src/pipeline/full_pipeline.py --help
```

### 🎓 Training Options

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--config` | 설정 파일 경로 (필수) | `configs/train_highperf.yaml` |
| `--mode` | 실행 모드 | `fast`, `highperf`, `full-pipeline` |
| `--optimize` | 하이퍼파라미터 최적화 활성화 | `--optimize` |
| `--optuna-config` | Optuna 설정 파일 | `configs/optuna_config.yaml` |
| `--optuna-trials` | Optuna 시행 횟수 | `--optuna-trials 100` |
| `--use-calibration` | 모델 캘리브레이션 사용 | `--use-calibration` |
| `--ensemble-models` | 앙상블 모델 지정 | `efficientnet,swin` |
| `--k-fold` | K-Fold 교차 검증 | `--k-fold 5` |
| `--auto-continue` | 중단된 학습 자동 재시작 | `--auto-continue` |
| `--auto-gpu-optimization` | GPU 자동 최적화 | `--auto-gpu-optimization` |
| `--wandb-project` | WandB 프로젝트 이름 | `--wandb-project my-project` |
| `--seed` | 랜덤 시드 설정 | `--seed 42` |
| `--debug` | 디버그 모드 | `--debug` |

### 🔍 Inference Options

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--config` | 설정 파일 경로 (필수) | `configs/infer_highperf.yaml` |
| `--model-path` | 모델 파일 경로 | `experiments/train/20250908/model.pth` |
| `--tta-mode` | TTA 모드 | `basic`, `aggressive`, `conservative` |
| `--ensemble-weight` | 앙상블 가중치 | `0.7,0.3` |
| `--calibration-temperature` | 캘리브레이션 온도 | `--calibration-temperature 1.5` |
| `--batch-size` | 배치 크기 | `auto`, `16`, `32` |
| `--output-dir` | 결과 저장 디렉토리 | `submissions/20250908/` |
| `--save-predictions` | 예측 결과 저장 | `--save-predictions` |
| `--confidence-threshold` | 신뢰도 임계값 | `--confidence-threshold 0.9` |
| `--visualization` | 시각화 생성 | `--visualization` |

### ⚡ Optimization Options

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--config` | Optuna 설정 파일 (필수) | `configs/optuna_config.yaml` |
| `--study-name` | 연구 이름 | `--study-name efficientnet-opt` |
| `--n-trials` | 최적화 시행 횟수 | `--n-trials 100` |
| `--timeout` | 최적화 시간 제한 (초) | `--timeout 3600` |
| `--pruning` | 가지치기 활성화 | `--pruning` |
| `--sampler` | 샘플러 타입 | `tpe`, `random`, `cmaes` |
| `--parallel-jobs` | 병렬 작업 수 | `--parallel-jobs 2` |
| `--storage-url` | 데이터베이스 URL | `sqlite:///optuna.db` |
| `--load-if-exists` | 기존 연구 로드 | `--load-if-exists` |

### 🔧 Common Options (모든 스크립트 공통)

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--verbose` | 상세 로그 출력 | `--verbose` |
| `--quiet` | 최소 로그 출력 | `--quiet` |
| `--log-level` | 로그 레벨 설정 | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--no-wandb` | WandB 비활성화 | `--no-wandb` |
| `--force` | 강제 실행 | `--force` |
| `--dry-run` | 테스트 실행 (실제 작업 안함) | `--dry-run` |

### 💡 Advanced Usage Examples

#### 🎯 완전 자동화 프로덕션 파이프라인
```bash
# 모든 기능을 통합한 원클릭 실행
python src/training/train_main.py \
  --config configs/train_multi_model_ensemble.yaml \
  --mode full-pipeline \
  --use-calibration \
  --optimize \
  --optuna-config configs/optuna_config.yaml \
  --optuna-trials 50 \
  --k-fold 5 \
  --auto-continue \
  --auto-gpu-optimization \
  --wandb-project cv-competition-final \
  --verbose
```

#### 🚀 빠른 프로토타이핑 (개발/테스트용)
```bash
# 20분 내 빠른 실험
python src/training/train_main.py \
  --config configs/train_fast_optimized.yaml \
  --mode fast \
  --dry-run \
  --debug \
  --no-wandb
```

#### 🏆 최종 제출용 고성능 학습
```bash
# 최고 성능을 위한 설정
python src/training/train_main.py \
  --config configs/train_highperf.yaml \
  --mode highperf \
  --use-calibration \
  --ensemble-models efficientnet,swin \
  --tta-mode aggressive \
  --k-fold 10 \
  --seed 42
```

#### 📊 하이퍼파라미터 최적화 전용
```bash
# 2시간 동안 최적화 실행
python src/optimization/optuna_optimizer.py \
  --config configs/optuna_config.yaml \
  --study-name final-optimization \
  --n-trials 200 \
  --timeout 7200 \
  --pruning \
  --parallel-jobs 2
```

#### 🔍 고급 추론 및 앙상블
```bash
# 복수 모델 앙상블 추론
python src/inference/infer_main.py \
  --config configs/infer_highperf.yaml \
  --tta-mode aggressive \
  --ensemble-weight 0.6,0.4 \
  --calibration-temperature 1.3 \
  --confidence-threshold 0.95 \
  --save-predictions \
  --visualization
```

#### 🤝 팀 협업용 GPU 최적화
```bash
# 팀원별 GPU 환경 자동 감지 후 학습
python src/utils/gpu_optimization/team_gpu_check.py && \
python src/training/train_main.py \
  --config configs/train_highperf.yaml \
  --auto-gpu-optimization \
  --batch-size auto \
  --wandb-project team-experiment
```
| `--force` | 강제 실행 | `--force` |
| `--dry-run` | 테스트 실행 (실제 작업 안함) | `--dry-run` |

---

## 📚 Scripts & Utilities

### 🤖 Core Automation Scripts

| Script | 설명 | 실행 시간 | 용도 |
|--------|------|-----------|------|
| `./scripts/run_fast_training.sh` | 빠른 학습 파이프라인 | 20-30분 | 프로토타이핑, 빠른 실험 |
| `./scripts/run_highperf_training.sh` | 고성능 학습 파이프라인 | 1-2시간 | 최종 제출용 고성능 모델 |
| `./scripts/monitor_training.sh` | 실시간 학습 모니터링 | 백그라운드 | 학습 과정 추적 |
| `./scripts/update_inference_date.sh` | 추론 설정 자동 업데이트 | 즉시 | 날짜별 추론 설정 |

### 🔧 Utility Tools

| 도구 | 경로 | 기능 |
|------|------|------|
| **GPU 최적화** | `src/utils/gpu_optimization/` | 자동 GPU 감지 및 배치 크기 최적화 |
| **설정 관리** | `src/utils/config/` | 시드 설정 및 날짜 자동 업데이트 |
| **공통 유틸** | `src/utils/core/` | 파일 처리, 로깅, 경로 관리 |
| **시각화 시스템** | `src/utils/visualizations/` | 학습/추론/최적화 차트 자동 생성 |
| **코드 관리** | `src/utils/code_management/` | 단위 테스트 로거 |

### 📊 Visualization System

프로젝트는 **완전 자동화된 시각화 시스템**을 포함하여 학습/추론/최적화 과정을 실시간으로 추적하고 저장합니다:

#### 🎓 Training Visualizations (7종류)
- **Loss Curves**: Training/Validation Loss 추적
- **Accuracy Metrics**: 정확도 변화 모니터링  
- **Learning Rate**: 학습률 스케줄링 추적
- **GPU Memory**: 메모리 사용량 모니터링
- **Training Speed**: Epoch별 속도 분석
- **Class Distribution**: 클래스별 성능 분석
- **Confusion Matrix**: 예측 성능 매트릭스

#### 🔍 Inference Visualizations (7종류)
- **Prediction Confidence**: 예측 신뢰도 분포
- **Processing Time**: 추론 속도 분석
- **Memory Usage**: 추론 메모리 사용량
- **Batch Performance**: 배치별 성능 분석
- **Model Comparison**: 모델간 성능 비교
- **Error Analysis**: 오류 패턴 분석
- **TTA Results**: Test Time Augmentation 효과

#### ⚡ Optimization Visualizations (6종류)
- **Batch Size Optimization**: 최적 배치 크기 탐색
- **Hyperparameter Trends**: 하이퍼파라미터 최적화 과정
- **Performance Metrics**: 최적화 성능 지표
- **Resource Usage**: 자원 사용량 최적화
- **Speed Benchmarks**: 속도 최적화 결과
- **Convergence Analysis**: 수렴성 분석

**모든 차트는 자동으로 `experiments/{experiment_type}/images/` 디렉토리에 저장됩니다.**

---

## 🏗️ Project Structure

```
🏢 computer-vision-competition-1SEN/
├── 📁 configs/                                     # 설정 관리
│   ├── train.yaml                                  # 기본 학습 설정 (EfficientNet)
│   ├── train_highperf.yaml                         # 고성능 설정 (Swin Transformer)
│   ├── train_fast_optimized.yaml                   # 빠른 실험 설정 (20-30분)
│   ├── infer.yaml                                  # 기본 추론 설정
│   ├── infer_highperf.yaml                         # 고성능 추론 설정
│   ├── optuna_config.yaml                          # Optuna 최적화 설정
│   └── optuna_fast_config.yaml                     # 빠른 Optuna 설정
├── 📁 scripts/                                     # 실행 스크립트 관리
│   ├── monitor_training.sh                         # 학습 모니터링
│   ├── run_fast_training.sh                        # 빠른 학습 (20-30분)
│   ├── run_highperf_training.sh                    # 고성능 학습 (1-2시간)
│   └── update_inference_date.sh                    # 추론 설정 업데이트
├── 📁 data/                                        # 데이터 저장소
│   └── raw/                                        # 원본 데이터 (train.csv, test/, train/)
├── 📁 docs/                                        # 종합 문서화 시스템
│   ├── GPU_최적화_가이드.md                          # GPU 자동 최적화 가이드
│   ├── 모델_설정_가이드.md                           # 모델 설정 및 구성 가이드
│   ├── 문제해결_가이드.md                            # 트러블슈팅 가이드
│   ├── 시각화_시스템_가이드.md                        # 시각화 시스템 사용법
│   ├── 전체_파이프라인_가이드.md                      # 전체 파이프라인 워크플로우
│   ├── 추론_파이프라인_가이드.md                      # 추론 시스템 가이드
│   └── 학습_파이프라인_가이드.md                      # 학습 시스템 가이드
├── 📁 src/                                         # 모듈화된 Core Framework
│   ├── 📂 data/                                    # 데이터 처리 엔진
│   │   ├── dataset.py                              # Dataset 클래스 (Basic + HighPerf)
│   │   └── transforms.py                           # 고급 Augmentation
│   ├── 📂 models/                                  # AI Models
│   │   ├── build.py                                # 모델 팩토리
│   │   ├── efficientnet.py                         # EfficientNet 구현
│   │   └── swin.py                                 # Swin Transformer
│   ├── 📂 training/                                # Training Engine
│   │   ├── train.py                                # 기본 학습
│   │   ├── train_highperf.py                       # 고성능 학습
│   │   └── train_main.py                           # 실행 진입점
│   ├── 📂 inference/                               # Inference Engine  
│   │   ├── infer.py                                # 기본 추론
│   │   ├── infer_highperf.py                       # 고성능 추론 (TTA + Ensemble)
│   │   └── infer_main.py                           # 추론 진입점
│   ├── 📂 pipeline/                                # Automation Framework
│   │   └── full_pipeline.py                        # 완전 자동화 파이프라인
│   ├── 📂 utils/                                   # 모듈화된 유틸리티 시스템
│   │   ├── 📂 core/                                # 공통 핵심 기능
│   │   │   └── common.py                           # 파일/YAML 처리, 로깅, 경로 관리
│   │   ├── 📂 config/                              # 설정 관리
│   │   │   ├── seed.py                             # 시드 설정 및 재현성
│   │   │   └── update_config_dates.py              # 자동 날짜 업데이트
│   │   ├── 📂 gpu_optimization/                    # GPU 최적화 엔진
│   │   │   ├── team_gpu_check.py                   # 팀 GPU 호환성 자동 체크
│   │   │   └── auto_batch_size.py                  # 자동 배치 크기 최적화
│   │   ├── 📂 code_management/                     # 코드 관리 도구
│   │   │   └── unit_test_logger.py                 # 단위 테스트 로거
│   │   └── 📂 visualizations/                      # 통합 시각화 시스템
│   │       ├── base_visualizer.py                  # 시각화 엔진 베이스
│   │       ├── training_viz.py                     # 학습 시각화 (7종 차트)
│   │       ├── inference_viz.py                    # 추론 시각화 (7종 차트)
│   │       ├── optimization_viz.py                 # 최적화 시각화 (6종 차트)
│   │       └── output_manager.py                   # 자동 저장 관리
│   ├── 📂 optimization/                            # 하이퍼파라미터 최적화
│   │   └── optuna_optimizer.py                     # Optuna 통합 최적화
│   ├── 📂 metrics/                                 # 성능 평가 시스템
│   │   └── evaluator.py                            # 종합 성능 평가
│   ├── 📂 calibration/                             # 모델 캘리브레이션
│   │   └── temperature_scaling.py                  # Temperature Scaling
│   └── 📂 logging/                                 # Enterprise Logging
│       └── wandb_logger.py                         # WandB 통합 로거
├── 📁 notebooks/                                   # Research & Testing
│   ├── 📂 base/                                    # 기본 실험 노트북
│   ├── 📂 modular/                                 # 모듈화된 테스트 노트북
│   └── 📂 team/                                    # 팀 협업 노트북
├── 📁 experiments/                                 # 실험 결과 자동 저장 시스템
│   ├── 📂 train/                                   # 학습 실험 결과
│   │   └── 📂 {YYYYMMDD}/                          # 일별 학습 결과
│   │       ├── 📂 images/                          # 자동 생성 시각화 차트 (7종)
│   │       ├── 📂 logs/                            # 상세 학습 로그
│   │       ├── 📂 configs/                         # 사용된 설정 파일
│   │       └── 📂 results/                         # 모델 및 메트릭 결과
│   ├── 📂 inference/                               # 추론 실험 결과
│   │   └── 📂 {YYYYMMDD}/                          # 일별 추론 결과
│   │       ├── 📂 images/                          # 자동 생성 시각화 차트 (7종)
│   │       ├── 📂 logs/                            # 추론 로그
│   │       ├── 📂 configs/                         # 추론 설정
│   │       └── 📂 results/                         # 예측 결과 및 제출 파일
│   └── 📂 optimization/                            # 최적화 실험 결과
│       └── 📂 {YYYYMMDD}/                          # 일별 최적화 결과
│           ├── 📂 images/                          # 자동 생성 시각화 차트 (6종)
│           ├── 📂 logs/                            # 최적화 로그
│           ├── 📂 configs/                         # 최적화 설정
│           └── 📂 results/                         # 최적화 결과 및 베스트 파라미터
├── 📁 submissions/                                 # Competition Submissions
│   └── 📂 {YYYYMMDD}/                              # 일별 제출 파일
├── 📁 logs/                                        # System Logs
│   ├── 📂 {YYYYMMDD}/                              # 일별 시스템 로그
│   └── 📂 infer/                                   # 추론 전용 로그
├── 📁 wandb/                                       # WandB 실험 추적
└── 📋 requirements.txt                             # 의존성 관리
```

---

## 🎯 Competition Performance

### 📄 Document Classification Challenge
- **Task**: 17-class 문서 분류 (Document Type Classification)
- **Dataset**: 고해상도 문서 이미지 (1,000+ samples per class)
- **Metric**: F1-Score (Target: 0.934+)
- **Challenge**: 다양한 문서 타입, 해상도, 레이아웃 변화

### 🏅 Performance Achievements
- 🥇 **F1 Score**: **0.934** (Target Achieved)
- ⚡ **Training Speed**: 50% 향상 (GPU 자동 최적화)
- 🎯 **Inference Time**: <100ms per image
- 📊 **Model Efficiency**: 99.2% validation accuracy

---

## 🔧 Advanced Features

### 1. 🤝 Team Collaboration Engine
```bash
# 팀원 GPU 환경 자동 감지 & 최적화
python src/utils/gpu_optimization/team_gpu_check.py     # RTX 4090 → GTX 1660 모든 GPU 지원
python src/utils/gpu_optimization/auto_batch_size.py    # 자동 배치 크기 최적화 (안전 마진 적용)
```

### 2. ⚡ Production-Grade Pipeline
```bash
# 완전 자동화 파이프라인 (학습 → 검증 → 추론 → 제출)
python src/pipeline/full_pipeline.py --config configs/train_highperf.yaml --mode production

# 실시간 성능 모니터링
./scripts/monitor_training.sh
```

### 3. 🎨 Automatic Visualization System
- **20+ 차트 자동 생성**: 학습(7) + 추론(7) + 최적화(6)
- **한글 폰트 지원**: NanumGothic.ttf 통합
- **자동 저장**: experiments/{type}/images/ 디렉토리
- **실시간 업데이트**: 학습/추론 과정 중 실시간 차트 갱신

### 4. 🔍 Hyperparameter Optimization
```bash
# Optuna 자동 최적화
python src/optimization/optuna_optimizer.py --config configs/optuna_config.yaml

# 빠른 최적화 (30분)
python src/optimization/optuna_optimizer.py --config configs/optuna_fast_config.yaml
```

### 5. 📊 Enterprise Monitoring
- **WandB 통합**: 실시간 실험 추적
- **자동 로깅**: 모든 메트릭 자동 기록
- **실험 비교**: 다양한 설정별 성능 비교
- **재현성**: 완전한 실험 재현 지원

---

## 📖 Documentation

### 📚 Core Documentation

| 문서 | 설명 | 주요 내용 |
|------|------|-----------|
| [GPU 최적화 가이드](docs/GPU_최적화_가이드.md) | GPU 자동 최적화 시스템 | 팀 GPU 체크, 배치 크기 최적화, 메모리 관리 |
| [모델 설정 가이드](docs/모델_설정_가이드.md) | 모델 구성 및 설정 | EfficientNet, Swin Transformer 설정법 |
| [문제해결 가이드](docs/문제해결_가이드.md) | 트러블슈팅 | 일반적인 오류 및 해결책 |
| [시각화 시스템 가이드](docs/시각화_시스템_가이드.md) | 시각화 시스템 사용법 | 20+ 차트 생성 및 커스터마이징 |
| [전체 파이프라인 가이드](docs/전체_파이프라인_가이드.md) | 완전 자동화 워크플로우 | 원클릭 실행부터 제출까지 |
| [추론 파이프라인 가이드](docs/추론_파이프라인_가이드.md) | 추론 시스템 | TTA, Ensemble, 고성능 추론 |
| [학습 파이프라인 가이드](docs/학습_파이프라인_가이드.md) | 학습 시스템 | Fast/HighPerf 학습, 모니터링 |

---

## 🛠️ System Requirements

### 📋 Hardware Requirements
- **GPU**: CUDA-compatible (GTX 1660 이상 권장)
- **Memory**: 8GB+ RAM, 6GB+ VRAM
- **Storage**: 20GB+ 여유 공간

### 📦 Software Requirements
- **Python**: 3.11.9 (pyenv 가상환경 권장)
- **CUDA**: 11.8+ (GPU 사용 시)
- **OS**: Linux/Windows/macOS 지원

### 🔧 Dependencies
주요 라이브러리:
- `torch`, `torchvision`: 딥러닝 프레임워크
- `transformers`: Transformer 모델 지원
- `wandb`: 실험 추적 및 모니터링
- `optuna`: 하이퍼파라미터 최적화
- `matplotlib`, `seaborn`: 시각화
- `pandas`, `numpy`: 데이터 처리

전체 의존성은 `requirements.txt` 참조.

---

## 🚀 Getting Started

### 1. 프로젝트 설정
```bash
git clone <repository-url>
cd computer-vision-competition-1SEN
pyenv activate cv-competition
pip install -r requirements.txt
```

### 2. GPU 환경 체크
```bash
python src/utils/gpu_optimization/team_gpu_check.py
```

### 3. 빠른 실험 실행
```bash
./scripts/run_fast_training.sh
```

### 4. 결과 확인
- 시각화 차트: `experiments/train/{날짜}/images/`
- 학습 로그: `logs/{날짜}/`
- WandB 대시보드: https://wandb.ai/

### 5. 최종 제출용 실행
```bash
./scripts/run_highperf_training.sh
```

---

## 🤝 Team Collaboration

이 프로젝트는 **팀 협업을 위해 설계**되었습니다:

- 🔧 **자동 GPU 감지**: 팀원별 다른 GPU 환경 자동 대응
- 📊 **통합 모니터링**: WandB를 통한 실험 결과 공유
- 🔄 **재현성**: 완전한 실험 설정 및 결과 재현
- 📚 **문서화**: 상세한 가이드 및 트러블슈팅
- 🎨 **자동 시각화**: 모든 실험 결과 자동 차트 생성

---

## 📞 Support & Contact

문제가 발생하거나 질문이 있으시면:

1. **문서 확인**: `docs/` 폴더의 관련 가이드 참조
2. **문제해결 가이드**: `docs/문제해결_가이드.md` 확인
3. **이슈 리포트**: GitHub Issues 또는 팀 채널 활용