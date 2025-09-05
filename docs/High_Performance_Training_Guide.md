# 🚀 High-Performance Training Pipeline 사용 가이드

기존 0.87점에서 **0.934점**을 달성한 노트북을 기반으로 모듈화된 고성능 학습 파이프라인입니다.

## ✨ 주요 특징

### 🎯 성능 향상 요소
- **Swin Transformer Base** (384x384 해상도)
- **Hard Augmentation** (에포크별 강도 증가)
- **Mixup 데이터 증강**
- **Test Time Augmentation (TTA)**
- **5-Fold 앙상블**
- **WandB 실시간 로깅**

### 📊 지원 모델
- `swin_base_384`: Swin Transformer Base (384px) - **추천**
- `convnext_base_384`: ConvNext Base (384px) - **추천**  
- `efficientnet_b3`: EfficientNet-B3
- `efficientnet_v2_b3`: EfficientNetV2-B3

## 🏃‍♂️ 빠른 시작

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# WandB 로그인 (선택사항)
wandb login
```

### 2. 고성능 학습 실행
```bash
# 실행 권한 부여
chmod +x run_highperf_training.sh

# 학습 시작
./run_highperf_training.sh
```

### 사전 준비 (권장)
```bash
# 1. pyenv 가상환경 활성화
pyenv activate cv_py3_11_9

# 2. GPU 호환성 빠른 체크
python src/utils/team_gpu_check.py

# 3. 자동 배치 크기 최적화
python src/utils/auto_batch_size.py --config configs/train_highperf.yaml
```

### 고성능 학습 실행
```bash
# 직접 실행 (권장)
python -m src.training.train_highperf configs/train_highperf.yaml

# 또는 메인 스크립트 사용
python src/training/train_main.py --mode highperf
```

### 완전한 실행 시퀀스
```bash
# 1-3. 사전 준비
pyenv activate cv_py3_11_9
python src/utils/team_gpu_check.py
python src/utils/auto_batch_size.py --config configs/train_highperf.yaml

# 4. 고성능 학습 시작
python -m src.training.train_highperf configs/train_highperf.yaml
```

### 3. 추론 실행
```bash
# 추론용 배치 크기 최적화 (옵션)
python src/utils/auto_batch_size.py --config configs/infer.yaml --test-only

# 추론 실행
python -m src.inference.infer_highperf \
  configs/train_highperf.yaml \
  experiments/train/YYYYMMDD/v094-swin-highperf/fold_results.yaml \
  submissions/highperf_result.csv
```

## 📁 결과 파일 구조

```
experiments/train/YYYYMMDD/v094-swin-highperf/
├── ckpt/
│   ├── best_model_fold_1.pth
│   ├── best_model_fold_2.pth
│   ├── best_model_fold_3.pth
│   ├── best_model_fold_4.pth
│   └── best_model_fold_5.pth
├── config.yaml                # 설정 스냅샷
├── fold_results.yaml          # 폴드별 결과
└── metrics.jsonl              # 메트릭 로그

logs/train/
└── train_highperf_YYYYMMDD-HHMM_[run_id].log

submissions/YYYYMMDD/
└── highperf_ensemble.csv      # 최종 제출 파일
```

## ⚙️ 설정 커스터마이징

### 모델 변경
`configs/train_highperf.yaml`에서 모델 변경:

```yaml
# Swin Transformer (기본값, 최고 성능)
model:
  name: "swin_base_384"

# ConvNext 사용시
model:
  name: "convnext_base_384"

# EfficientNet 사용시  
model:
  name: "efficientnet_b3"
train:
  img_size: 300  # EfficientNet에 맞는 해상도
```

### 하이퍼파라미터 조정
```yaml
train:
  img_size: 384        # 이미지 해상도
  batch_size: 32       # 배치 크기
  epochs: 15           # 에포크 수
  lr: 0.0001          # 학습률
  mixup_alpha: 1.0     # Mixup 강도
  use_mixup: true      # Mixup 사용 여부
```

## 🔄 기존 모듈화 코드와의 차이점

| 항목 | 기존 모듈 | 고성능 모듈 |
|------|----------|------------|
| 모델 | EfficientNet-B3 | Swin Transformer |
| 해상도 | 224x224 | 384x384 |
| 데이터 증강 | 기본 | Hard + Mixup |
| 배치 크기 | 64 | 32 |
| 학습률 | 0.001 | 0.0001 |
| 로깅 | 기본 | WandB |
| 앙상블 | 없음 | 5-Fold + TTA |

## 📈 예상 성능

- **기존 모듈**: F1 ~0.372
- **고성능 모듈**: F1 ~0.934 (목표)

## 🛠️ 트러블슈팅

### 메모리 부족
```yaml
train:
  batch_size: 16      # 배치 크기 감소
  img_size: 320       # 해상도 감소
```

### 학습 속도 개선
```yaml
project:
  num_workers: 8      # 워커 수 증가
train:
  mixed_precision: true  # AMP 활성화
```

### WandB 없이 실행
`src/training/train_highperf.py`에서 WandB 관련 코드 주석 처리

## 🎯 성능 최적화 팁

1. **GPU 메모리 최적화**: 배치 크기와 해상도 조절
2. **하이퍼파라미터 튜닝**: Optuna 등으로 자동 튜닝
3. **앙상블 강화**: 더 많은 폴드나 다른 모델 조합
4. **데이터 증강**: 더 강한 증강 기법 추가

## 📞 문의

- 학습 관련: `src/training/train_highperf.py` 참조
- 추론 관련: `src/inference/infer_highperf.py` 참조  
- 설정 관련: `configs/train_highperf.yaml` 참조
