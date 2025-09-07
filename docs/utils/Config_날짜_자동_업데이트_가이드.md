# 📅 Config 날짜 자동 업데이트 가이드

추론 설정 파일들의 실험 날짜를 자동으로 업데이트하는 유틸리티입니다.

## 🚀 빠른 사용법

### 1. 🆕 **Latest-train 폴더 기준 업데이트** (권장!)
```bash
# 스크립트 사용 - 날짜와 관계없이 항상 최신 학습 결과 참조
bash scripts/update_inference_date.sh --latest-train

# 도움말 확인
bash scripts/update_inference_date.sh --help
```

### 2. 🔥 **최신 실험 날짜로 업데이트**
```bash
# 스크립트 사용 (scripts 폴더로 이동됨)
bash scripts/update_inference_date.sh --latest

# 또는 Python 스크립트 직접 사용
python src/utils/update_config_dates.py --latest
```

### 3. 📅 **특정 날짜로 업데이트**
```bash
# 스크립트 사용
bash scripts/update_inference_date.sh 20250908

# Python 스크립트
python src/utils/update_config_dates.py --date 20250908
```

### 4. 🌅 **오늘 날짜로 업데이트**
```bash
# 스크립트 사용 (기본값)
bash scripts/update_inference_date.sh

# Python 스크립트
python src/utils/update_config_dates.py
```

## 🆕 Latest-train 시스템

### 📁 **폴더 구조**
```
experiments/train/
├── 20250907/                    # 원본: 날짜별 저장
│   └── swin-highperf_20250907_1825/
├── 20250908/
│   └── efficientnet-basic_20250908_1030/
└── latest-train/                # 🆕 최신 결과 자동 복사
    └── efficientnet-basic_20250908_1030/
```

### 🎯 **Latest-train 기능의 이점**
- ✅ **날짜 독립**: 학습이 자정을 넘어도 항상 최신 결과 접근
- ✅ **워크플로우 간소화**: `--latest-train` 하나로 해결
- ✅ **실수 방지**: 잘못된 날짜 지정으로 인한 오류 방지
- ✅ **자동화**: 학습 완료 시 자동으로 latest-train 폴더에 복사

## 📋 업데이트되는 파일들

### `configs/infer.yaml`
```yaml
# 변경 전
ckpt:
  path: "../../experiments/train/20250906/efficientnet_b3/ckpt/best_model_fold_1.pth"

# 변경 후 (20250907로 업데이트)
ckpt:
  path: "../../experiments/train/20250907/efficientnet_b3/ckpt/best_model_fold_1.pth"
```

### `configs/infer_highperf.yaml`
```yaml
# 변경 전
ensemble:
  fold_results_path: "../../experiments/train/20250906/v094-swin-highperf/fold_results.yaml"

# 변경 후 (20250907로 업데이트, 폴더명도 자동 감지)
ensemble:
  fold_results_path: "../../experiments/train/20250907/swin-sighperf/fold_results.yaml"
```

## 🛡️ 안전 기능

### 자동 백업
- 모든 설정 파일을 수정하기 전에 `.backup` 파일로 백업 생성
- Python: `configs/infer.yaml.backup`
- Shell: `configs/infer.yaml.backup.20250907_1430` (타임스탬프 포함)

### 폴더명 자동 감지
- `efficientnet*` 패턴의 폴더 자동 탐지
- `swin*` 패턴의 폴더 자동 탐지
- 실제 존재하는 폴더만 업데이트

## 🔧 고급 사용법

### 특정 설정 파일만 업데이트
```bash
# infer.yaml만 업데이트
python src/utils/update_config_dates.py --latest --configs configs/infer.yaml

# 사용자 정의 설정 파일 업데이트
python src/utils/update_config_dates.py --latest --configs configs/my_config.yaml
```

### 사용 가능한 날짜 확인
```bash
ls experiments/train/
# 출력: 20250905  20250906  20250907
```

## ⚡ 워크플로우 예시

### 🆕 시나리오 1: Latest-train 기반 완전 자동화 워크플로우 (권장!)
```bash
# 1. 새로운 실험 완료 후 (통합 CLI) - 자동으로 latest-train에 복사됨
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --optimize \
    --use-calibration \
    --auto-continue

# 2. 설정 파일 자동 업데이트 (latest-train 기준)
bash scripts/update_inference_date.sh --latest-train

# 3. 바로 추론 실행 - 날짜 걱정 없음!
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline \
    --skip-training \
    --use-calibration
```

### 시나리오 2: 기존 방식 - 최신 날짜 기준
```bash
# 1. 새로운 실험 완료 후 (통합 CLI)
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --optimize \
    --use-calibration \
    --auto-continue

# 2. 설정 파일 자동 업데이트 (최신 날짜 기준)
bash scripts/update_inference_date.sh --latest

# 3. 바로 추론 실행
python src/training/train_main.py \
    --config configs/train_highperf.yaml \
    --mode full-pipeline \
    --skip-training \
    --use-calibration
```

### 시나리오 3: 특정 날짜의 모델로 재추론
```bash
# 2025년 9월 5일 모델로 추론하고 싶을 때
bash scripts/update_inference_date.sh 20250905
python src/inference/infer_main.py --config configs/infer.yaml --mode basic
```

### 시나리오 4: 여러 날짜의 모델 비교 추론
```bash
# 날짜별로 설정 업데이트하며 추론 비교
for date in 20250905 20250906 20250907; do
    echo "=== $date 모델 추론 ==="
    bash scripts/update_inference_date.sh $date
    python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf
done

# 🆕 Latest-train 결과와 비교
echo "=== Latest-train 모델 추론 ==="
bash scripts/update_inference_date.sh --latest-train
python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf
```
python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf
```

## 🎯 일반적인 사용 시나리오

### 시나리오 1: 매일 새로운 실험 후 추론
```bash
# 오늘 실험 완료 → 설정 업데이트 → 추론
./update_inference_date.sh --latest
python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf
```

### 시나리오 2: 특정 날짜의 모델로 재추론
```bash
# 2025년 9월 5일 모델로 추론하고 싶을 때
./update_inference_date.sh 20250905
python src/inference/infer_main.py --config configs/infer.yaml --mode basic
```

### 시나리오 3: 여러 날짜의 모델 비교 추론
```bash
# 날짜별로 설정 업데이트하며 추론 비교
for date in 20250905 20250906 20250907; do
    echo "=== $date 모델 추론 ==="
    ./update_inference_date.sh $date
    python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf
done
```

## 🚨 주의사항

1. **🆕 Latest-train 우선 사용**: 대부분의 경우 `--latest-train` 옵션 사용 권장
2. **실험 디렉터리 존재 확인**: 지정한 날짜의 `experiments/train/YYYYMMDD/` 디렉터리가 존재해야 함
3. **모델 파일 존재 확인**: 업데이트된 경로에 실제 모델 파일들이 있는지 확인
4. **백업 파일 관리**: 필요 없는 백업 파일들은 주기적으로 정리
5. **🆕 Latest-train 폴더**: 학습 완료 후 자동 생성되므로, 학습을 한 번도 실행하지 않았다면 존재하지 않음

## 🔍 문제 해결

### Q: "날짜 디렉터리를 찾을 수 없습니다" 오류
```bash
# 사용 가능한 날짜 확인
ls experiments/train/
# 해당 날짜로 실험이 실행되었는지 확인
```

### Q: "모델 실험을 찾을 수 없습니다" 오류  
```bash
# 해당 날짜의 모델 폴더 확인
ls experiments/train/20250907/
# efficientnet* 또는 swin* 폴더가 있는지 확인
```

### Q: 백업에서 복원하고 싶을 때
```bash
# Python 백업에서 복원
cp configs/infer.yaml.backup configs/infer.yaml

# Shell 백업에서 복원 (최신 백업 사용)
cp $(ls -t configs/infer.yaml.backup.* | head -1) configs/infer.yaml
```

---

💡 **팁**: 쉘 스크립트(`./update_inference_date.sh`)가 더 빠르고 간단하므로 일상적인 사용에는 이를 권장합니다!
