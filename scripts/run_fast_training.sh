#!/bin/bash

# 빠른 최적화 실행 스크립트
# 목표: 20-30분 내 결과 도출

echo "🚀 빠른 최적화 학습 시작"
echo "목표 시간: 20-30분"
echo "설정: train_fast_optimized.yaml + optuna_fast_config.yaml"
echo ""

# 시작 시간 기록
start_time=$(date +%s)
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 빠른 최적화 실행
python src/training/train_main.py \
    --config configs/train_fast_optimized.yaml \
    --optimize \
    --optuna-config configs/optuna_fast_config.yaml \
    --n-trials 8 \
    --mode full-pipeline \
    --auto-continue

# 종료 시간 계산
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "✅ 빠른 최적화 완료!"
echo "총 실행 시간: ${hours}시간 ${minutes}분 ${seconds}초"

# 결과 확인
echo ""
echo "📊 결과 파일 확인:"
find submissions/ -name "*$(date +%Y%m%d)*" -type f | tail -3
echo ""
echo "📝 로그 파일:"
find logs/ -name "*$(date +%Y%m%d)*" -type f | tail -2
