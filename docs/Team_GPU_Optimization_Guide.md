# 팀 협업용 GPU 최적화 자동 배치 크기 도구 사용 가이드

## 🚀 개요
팀원들이 다양한 GPU 환경에서 일관된 성능을 얻을 수 있도록 각 GPU에 최적화된 배치 크기를 자동으로 찾아주는 도구입니다.

## 🎯 팀 협업을 위한 핵심 기능

### GPU 등급별 자동 분류
- **high_end**: RTX 4090, RTX 4080, RTX 3090, A100, V100
- **mid_range**: RTX 3080, RTX 3070, RTX 4070  
- **budget**: RTX 3060, RTX 2070, RTX 2080
- **low_end**: GTX 1660, GTX 1080 등 구형 GPU

### 이미지 크기별 최적화
- **224px**: 높은 배치 크기 가능
- **384px**: 중간 배치 크기
- **512px**: 낮은 배치 크기 (메모리 절약)

## 📋 사용법

### 기본 사용법
```bash
# 테스트만 수행 (설정 파일 수정 안함)
python src/utils/auto_batch_size.py --config configs/train.yaml --test-only

# 설정 파일 자동 업데이트
python src/utils/auto_batch_size.py --config configs/train.yaml

# 고성능 설정 최적화
python src/utils/auto_batch_size.py --config configs/train_highperf.yaml
```

### 고급 옵션
```bash
# 특정 모델 지정
python src/utils/auto_batch_size.py --config configs/train.yaml --model swin_base_384

# 특정 이미지 크기 지정
python src/utils/auto_batch_size.py --config configs/train.yaml --img-size 384
```

## 🤝 팀 협업 시나리오

### 시나리오 1: RTX 4090 사용자 (현재 환경)
```bash
python src/utils/auto_batch_size.py --config configs/train_highperf.yaml
# 결과: batch_size = 48 (384px 이미지)
```

### 시나리오 2: RTX 3060 사용자
```bash
python src/utils/auto_batch_size.py --config configs/train.yaml
# 예상 결과: batch_size = 12-16 (384px 이미지)
# 추가 권장: gradient_accumulation_steps = 3-4
```

### 시나리오 3: GTX 1080 사용자
```bash
python src/utils/auto_batch_size.py --config configs/train.yaml
# 예상 결과: batch_size = 6-8 (384px 이미지)
# 추가 권장: mixed precision 비활성화, gradient_accumulation_steps = 6-8
```

## 📊 GPU별 권장 설정

### RTX 4090/3090 (24GB)
- **384px**: batch_size 32-64
- **512px**: batch_size 16-32
- **추가 권장**: Multi-GPU training 가능, ensemble 모델 고려

### RTX 3080/3070 (10-12GB)
- **384px**: batch_size 16-32
- **512px**: batch_size 8-16
- **추가 권장**: gradient_accumulation_steps = 2

### RTX 3060 (8GB)
- **384px**: batch_size 8-16
- **512px**: batch_size 4-8
- **추가 권장**: gradient_accumulation_steps = 3-4

### GTX 1080/1660 (6-8GB)
- **384px**: batch_size 4-8
- **512px**: batch_size 2-4
- **추가 권장**: mixed precision 비활성화, gradient_accumulation_steps = 6-8

## 🛡️ 안전 마진 및 최적화

### 안전 마진 적용
- **고사양 GPU**: 20% 안전 마진 (더 공격적 최적화)
- **중급 GPU**: 20% 안전 마진
- **예산형 GPU**: 15% 안전 마진
- **구형 GPU**: 10% 안전 마진 (보수적 접근)

### 메모리 효율성
- 모든 배치 크기는 4의 배수로 조정 (GPU 효율성)
- 이진 탐색으로 최적 배치 크기 자동 탐지
- 실제 모델 구조 시뮬레이션으로 정확한 메모리 사용량 측정

## ⚙️ 팀 협업 워크플로우

### 1. 각 팀원별 최적화
```bash
# 각자의 GPU 환경에서 실행
python src/utils/auto_batch_size.py --config configs/train.yaml --test-only
```

### 2. 설정 공유
```bash
# 최적화된 설정으로 업데이트
python src/utils/auto_batch_size.py --config configs/train.yaml

# Git으로 설정 공유
git add configs/train.yaml
git commit -m "Optimize batch size for [GPU_NAME]"
```

### 3. 실험 실행
```bash
# 최적화된 설정으로 훈련
python src/training/train_main.py --mode highperf

# GPU 모니터링
nvidia-smi -l 1
```

## 📈 예상 성능 향상

### RTX 4090 기준
- **기존 batch_size 32** → **최적화 batch_size 48**: ~50% 속도 향상
- **메모리 사용률**: ~38% (여유 공간으로 추가 실험 가능)

### RTX 3060 기준  
- **기존 batch_size 16** → **최적화 batch_size 12 + gradient_accumulation_steps 4**: 동일한 effective batch size, 안정성 향상

### GTX 1080 기준
- **기존 batch_size 8** → **최적화 batch_size 6 + gradient_accumulation_steps 8**: 메모리 안정성 확보

## 🔍 모니터링 및 디버깅

### GPU 사용량 모니터링
```bash
# 실시간 GPU 모니터링
nvidia-smi -l 1

# 메모리 사용량 확인
python -c "import torch; print(f'GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"
```

### 문제 해결
- **Out of Memory 오류**: batch_size를 50% 줄이고 gradient_accumulation_steps 증가
- **느린 훈련 속도**: batch_size 증가 시도, GPU 사용률 확인
- **불안정한 훈련**: mixed precision 비활성화, learning rate 조정

## 🚨 주의사항

### 팀 협업 시
1. **설정 파일 충돌 방지**: 각자 브랜치에서 최적화 후 merge
2. **실험 재현성**: 동일한 effective batch size 유지 (batch_size × gradient_accumulation_steps)
3. **하드웨어 차이 고려**: 결과 비교 시 GPU 등급별 성능 차이 인정

### 실제 훈련 전
1. **작은 epoch 테스트**: 2-3 epoch으로 안정성 확인
2. **메모리 모니터링**: 전체 GPU 메모리의 90% 이하 사용
3. **백업 설정**: 원본 설정 파일 백업 유지

## 📝 팀 설정 공유 템플릿

```yaml
# 팀원별 권장 설정 (configs/team_settings.yaml)
team_gpu_settings:
  rtx_4090:
    batch_size: 48
    gradient_accumulation_steps: 1
    mixed_precision: true
  
  rtx_3060:
    batch_size: 12
    gradient_accumulation_steps: 4
    mixed_precision: true
  
  gtx_1080:
    batch_size: 6
    gradient_accumulation_steps: 8
    mixed_precision: false
```

## ✅ 성공 사례

### 현재 최적화 결과 (RTX 4090)
- **모델**: swin_base_384
- **이미지 크기**: 384px
- **최적 배치 크기**: 48
- **메모리 사용률**: 38%
- **예상 성능 향상**: 50%

---

**Created by**: AI Team  
**Date**: 2025-01-05  
**Tool**: `/src/utils/auto_batch_size.py`  
**Status**: ✅ Production Ready for Team Collaboration
