# 📁 Scripts 종합 가이드

## 개요
`scripts/` 폴더는 경진대회 프로젝트의 핵심 실행 스크립트들을 관리하는 중앙 집중식 폴더입니다. 각 스크립트는 특정 목적에 최적화되어 있으며, 함께 사용하여 완전한 ML 파이프라인을 구성할 수 있습니다.

## 📂 폴더 구조
```
scripts/
├── monitor_training.sh          # 🔍 학습 진행 상황 실시간 모니터링
├── run_fast_training.sh         # ⚡ 빠른 최적화 실행 (20-30분)
├── run_highperf_training.sh     # 🏆 고성능 학습 실행 (1-2시간)
└── update_inference_date.sh     # 🔄 추론 설정 날짜 업데이트
```

## 🎯 스크립트별 용도

### 🔍 monitor_training.sh
**목적**: 실시간 학습 모니터링 및 상태 확인
```bash
./scripts/monitor_training.sh
```
- 실행 중인 프로세스 상태 확인
- CPU/메모리 사용률 모니터링
- 최신 로그 파일 내용 확인
- **사용 시기**: 학습 중 진행 상황 체크

### ⚡ run_fast_training.sh
**목적**: 경진대회용 빠른 프로토타이핑 및 실험
```bash
./scripts/run_fast_training.sh
```
- **실행 시간**: 20-30분
- **목표 F1**: 0.92+
- **사용 시기**: 빠른 실험, 하이퍼파라미터 테스트

### 🏆 run_highperf_training.sh
**목적**: 최종 제출용 고품질 모델 학습
```bash
./scripts/run_highperf_training.sh
```
- **실행 시간**: 1-2시간
- **목표 F1**: 0.934+
- **사용 시기**: 최종 제출, 벤치마크 설정

### 🔄 update_inference_date.sh
**목적**: 추론 설정을 최신 모델로 자동 업데이트
```bash
./scripts/update_inference_date.sh --latest
```
- 추론 config 파일 날짜 업데이트
- 백업 파일 자동 생성
- **사용 시기**: 학습 완료 후, 추론 실행 전

## 🔄 권장 워크플로우

### 1. 빠른 실험 사이클 (개발 단계)
```bash
# 1단계: 빠른 학습으로 아이디어 검증
./scripts/run_fast_training.sh

# 2단계: 진행 상황 모니터링
./scripts/monitor_training.sh

# 3단계: 결과가 좋으면 추론 설정 업데이트
./scripts/update_inference_date.sh --latest

# 4단계: 추론 실행
python src/inference/infer.py --config configs/infer.yaml
```

### 2. 고품질 최종 제출 사이클
```bash
# 1단계: 고성능 학습 실행
./scripts/run_highperf_training.sh

# 2단계: 주기적 모니터링 (별도 터미널)
watch -n 300 './scripts/monitor_training.sh'

# 3단계: 완료 후 추론 설정 업데이트
./scripts/update_inference_date.sh --latest

# 4단계: 고성능 추론 실행
python src/inference/infer_highperf.py --config configs/infer_highperf.yaml
```

### 3. 연속 실험 자동화
```bash
# 자동화 스크립트 예시
for config in configs/train_*.yaml; do
    echo "실행 중: $config"
    
    # 빠른 학습
    ./scripts/run_fast_training.sh
    
    # 결과 저장
    cp submissions/$(date +%Y%m%d)/submission_*.csv \
       results/experiment_$(basename $config .yaml).csv
    
    # 설정 업데이트
    ./scripts/update_inference_date.sh --latest
done
```

## ⚙️ 설정 및 커스터마이징

### 환경 변수 설정
```bash
# .bashrc 또는 .zshrc에 추가
export PROJECT_ROOT="/path/to/computer-vision-competition-1SEN"
export FAST_TRAINING_TIME=30    # 빠른 학습 시간 (분)
export HIGHPERF_TRAINING_TIME=120  # 고성능 학습 시간 (분)

# 스크립트에서 사용
cd $PROJECT_ROOT
timeout ${FAST_TRAINING_TIME}m ./scripts/run_fast_training.sh
```

### GPU별 최적화 설정
```bash
# RTX 4090 사용자
export FAST_BATCH_SIZE=64
export HIGHPERF_BATCH_SIZE=32

# RTX 3080 사용자
export FAST_BATCH_SIZE=32
export HIGHPERF_BATCH_SIZE=16

# GTX 1660 사용자
export FAST_BATCH_SIZE=16
export HIGHPERF_BATCH_SIZE=8
```

## 📊 성능 비교표

| 스크립트 | 실행시간 | F1 스코어 | GPU 메모리 | 사용 목적 |
|---------|----------|----------|------------|-----------|
| `run_fast_training.sh` | 20-30분 | 0.92+ | 8-12GB | 빠른 실험 |
| `run_highperf_training.sh` | 1-2시간 | 0.934+ | 16-20GB | 최종 제출 |

## 🛠️ 문제 해결 체크리스트

### 실행 권한 문제
```bash
# 모든 스크립트에 실행 권한 부여
chmod +x scripts/*.sh

# 개별 파일 권한 확인
ls -la scripts/
```

### 경로 문제
```bash
# 반드시 프로젝트 루트에서 실행
cd /path/to/computer-vision-competition-1SEN
./scripts/스크립트명.sh

# 현재 위치 확인
pwd
```

### 메모리 부족 문제
```bash
# GPU 메모리 확인
nvidia-smi

# 배치 크기 감소 (config 파일 수정)
# train_fast_optimized.yaml
train:
  batch_size: 16  # 32 → 16으로 감소
```

### 프로세스 충돌 문제
```bash
# 기존 프로세스 확인 및 종료
./scripts/monitor_training.sh
pkill -f train_main.py

# GPU 메모리 정리
nvidia-smi --gpu-reset
```

## 🔧 고급 활용법

### 1. 로그 집계 및 분석
```bash
# 모든 실행 결과 집계
./scripts/aggregate_results.sh() {
    echo "=== 실행 결과 요약 ==="
    
    # 빠른 학습 결과
    find logs/train -name "*fast*" -exec grep "Best F1" {} \;
    
    # 고성능 학습 결과
    find logs/train -name "*highperf*" -exec grep "Best F1" {} \;
    
    # 최신 제출 파일
    ls -la submissions/$(date +%Y%m%d)/ | tail -5
}
```

### 2. 스케줄링 자동화
```bash
# crontab을 이용한 자동 실험
# crontab -e

# 매일 새벽 2시에 고성능 학습
0 2 * * * cd /path/to/project && ./scripts/run_highperf_training.sh

# 매주 일요일에 전체 파이프라인 실행
0 1 * * 0 cd /path/to/project && ./scripts/weekly_experiment.sh
```

### 3. 알림 시스템 연동
```bash
# Slack 알림 (webhook 설정 필요)
notify_slack() {
    local message="$1"
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"$message\"}" \
         $SLACK_WEBHOOK_URL
}

# 학습 완료 알림
./scripts/run_fast_training.sh && \
notify_slack "빠른 학습 완료! F1 스코어: $(grep 'Best F1' logs/train/latest.log)"
```

## 📈 모니터링 대시보드

### 실시간 상태 대시보드
```bash
# tmux를 이용한 분할 모니터링
tmux new-session -d -s training
tmux split-window -h
tmux send-keys -t 0 './scripts/monitor_training.sh; sleep 60; exit' Enter
tmux send-keys -t 1 'nvidia-smi -l 5' Enter
tmux attach-session -t training
```

### 웹 대시보드 (선택사항)
```python
# simple_dashboard.py
import streamlit as st
import subprocess
import time

st.title("학습 모니터링 대시보드")

if st.button("현재 상태 확인"):
    result = subprocess.run(['./scripts/monitor_training.sh'], 
                          capture_output=True, text=True)
    st.text(result.stdout)

# 실행: streamlit run simple_dashboard.py
```

## 🔄 업데이트 및 유지보수

### 스크립트 버전 관리
```bash
# 스크립트 버전 확인
grep -H "# Version" scripts/*.sh

# 백업 생성
cp -r scripts scripts_backup_$(date +%Y%m%d)

# Git을 이용한 버전 관리
git add scripts/
git commit -m "Update scripts: $(date +%Y%m%d)"
```

### 성능 프로파일링
```bash
# 실행 시간 측정
time ./scripts/run_fast_training.sh

# 상세 프로파일링
strace -c -f ./scripts/run_fast_training.sh 2>&1 | tail -20
```

## 📚 관련 문서
- [monitor_training 가이드](./monitor_training_가이드.md)
- [run_fast_training 가이드](./run_fast_training_가이드.md)
- [run_highperf_training 가이드](./run_highperf_training_가이드.md)
- [update_inference_date 가이드](./update_inference_date_가이드.md)

## 🆘 지원 및 문의
- **문제 리포팅**: GitHub Issues
- **성능 최적화**: docs/utils/GPU_자동_설정_가이드.md 참조
- **팀 협업**: docs/utils/팀_GPU_최적화_가이드.md 참조

---

> **💡 팁**: 각 스크립트는 독립적으로 실행 가능하지만, 함께 사용할 때 최대 효과를 발휘합니다. 실험 목적에 따라 적절한 스크립트를 선택하여 사용하세요.
