#!/bin/bash
# run_highperf_training.sh
# 고성능 학습 파이프라인 실행 스크립트

echo "🚀 Starting High-Performance Training Pipeline"
echo "==============================================="

# 프로젝트 루트로 이동
cd "$(dirname "$0")"

# Python 환경 활성화 (필요시)
# source venv/bin/activate

# GPU 메모리 확인
echo "📊 GPU 상태 확인:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "🎯 실행 중인 설정:"
echo "- 모델: Swin Transformer Base (384px)"
echo "- Hard Augmentation + Mixup"
echo "- WandB 로깅 활성화"
echo "- 5-Fold Cross Validation"

echo ""
echo "🏃‍♂️ 학습 시작..."

# 고성능 학습 실행
python src/training/train_main.py --config configs/train_highperf.yaml --mode highperf

echo ""
echo "✅ 학습 완료! 결과는 experiments/train/ 폴더에서 확인하세요."
echo "📈 WandB 대시보드: https://wandb.ai"
