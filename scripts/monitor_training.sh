#!/bin/bash

# 현재 실행 중인 학습 프로세스 모니터링 스크립트
# 사용법: ./monitor_training.sh

echo "=== 학습 프로세스 모니터링 ==="
echo "현재 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 실행 중인 프로세스 확인
echo "📊 실행 중인 학습 프로세스:"
ps aux | grep -E "train_main|python.*train" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    time=$(echo $line | awk '{print $10}')
    echo "  PID: $pid | CPU: $cpu% | MEM: $mem% | TIME: $time"
done

echo ""

# 최신 로그 파일 확인
echo "📝 최신 로그 파일:"
latest_log=$(find logs/ -name "*.log" -mtime -1 | sort -r | head -1)
if [ -n "$latest_log" ]; then
    echo "  파일: $latest_log"
    echo "  마지막 업데이트: $(stat -c %y "$latest_log")"
    echo ""
    echo "📋 최근 로그 (마지막 5줄):"
    tail -n 5 "$latest_log"
else
    echo "  최근 로그 파일을 찾을 수 없습니다."
fi

echo ""
echo "=== 모니터링 완료 ==="
echo "다시 확인하려면: ./scripts/monitor_training.sh"
echo "프로세스 종료하려면: pkill -f train_main.py"
