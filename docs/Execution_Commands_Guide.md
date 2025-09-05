# 🚀 실행 명령어 완전 가이드

## 📋 목차
1. [환경 설정](#환경-설정)
2. [기본 실행 명령어](#기본-실행-명령어)
3. [학습 파이프라인](#학습-파이프라인)
4. [추론 파이프라인](#추론-파이프라인)
5. [전체 파이프라인](#전체-파이프라인)
6. [단위 테스트](#단위-테스트)
7. [로그 및 결과 확인](#로그-및-결과-확인)
8. [고급 사용법](#고급-사용법)
9. [문제 해결](#문제-해결)

---

## 🔧 환경 설정

### **1. 기본 환경 확인**
```bash
# 현재 디렉토리 확인
pwd
# 출력: /home/ieyeppo/AI_Lab/computer-vision-competition-1SEN

# Python 환경 확인
python3 --version
# 출력: Python 3.11.9

# 가상환경 활성화 상태 확인
echo $CONDA_DEFAULT_ENV
# 출력: cv_py3_11_9
```

### **2. 의존성 설치**
```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 설치 확인
pip list | grep -E "torch|pandas|numpy|matplotlib"

# 핵심 모듈 import 테스트
python3 -c "
import torch, pandas, numpy, matplotlib
from src.utils.common import load_yaml
from src.data.dataset import HighPerfDocClsDataset
print('✅ 모든 의존성 정상 설치됨')
"
```

### **3. GPU 환경 확인**
```bash
# CUDA 사용 가능 여부 확인
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# GPU 정보 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

---

## 🏃 기본 실행 명령어

### **프로젝트 구조 확인**
```bash
# 전체 구조 확인
tree -L 3 -I '__pycache__'

# 주요 디렉토리 확인
ls -la src/
ls -la configs/
ls -la data/raw/
ls -la notebooks/
```

### **설정 파일 검증**
```bash
# YAML 파일 유효성 확인
python3 -c "
import yaml
configs = ['configs/train.yaml', 'configs/train_highperf.yaml', 'configs/infer.yaml']
for config in configs:
    try:
        with open(config) as f:
            yaml.safe_load(f)
        print(f'✅ {config}')
    except Exception as e:
        print(f'❌ {config}: {e}')
"
```

---

## 🎓 학습 파이프라인

### **1. 기본 학습**

#### **단일 Fold 학습**
```bash
# 기본 설정으로 학습
python src/training/train_main.py

# 특정 설정 파일 사용
python src/training/train_main.py --config configs/train.yaml

# 특정 fold 학습
python src/training/train_main.py --fold 0

# GPU 지정하여 학습
CUDA_VISIBLE_DEVICES=0 python src/training/train_main.py --fold 0
```

#### **실행 예시**
```bash
# Fold 0 학습 (기본 설정)
python src/training/train_main.py --fold 0

# 출력 예시:
# 📋 학습 설정 로드 완료
# 🎯 모델: efficientnet_b3
# 📏 이미지 크기: 224
# 🔢 배치 크기: 16
# 📊 학습 데이터: 1,256개 샘플
# 📊 검증 데이터: 315개 샘플
# [1/30] Train Loss: 2.134, Val Loss: 1.876, Val F1: 0.456
# ...
```

### **2. 고성능 학습 (권장)**

#### **전체 K-Fold 학습**
```bash
# 고성능 모드 (5-fold 전체)
python src/training/train_main.py --mode highperf

# 고성능 설정 파일 지정
python src/training/train_main.py --mode highperf --config configs/train_highperf.yaml

# 백그라운드 실행 (장시간 학습)
nohup python src/training/train_main.py --mode highperf > training.log 2>&1 &

# 실행 상태 확인
tail -f training.log
```

#### **특정 Fold들만 학습**
```bash
# Fold 0, 1, 2만 학습
python src/training/train_main.py --mode highperf --folds 0,1,2

# 단일 Fold 고성능 모드
python src/training/train_main.py --mode highperf --folds 0
```

#### **WandB 통합 학습**
```bash
# WandB 로그인 (최초 1회)
wandb login

# WandB 통합 학습
python src/training/train_main.py --mode highperf --use-wandb

# 특정 프로젝트명 지정
WANDB_PROJECT="cv-competition-1sen" python src/training/train_main.py --mode highperf
```

### **3. 학습 모니터링**

#### **실시간 로그 확인**
```bash
# 최신 학습 로그 확인
ls -t logs/train/ | head -1 | xargs -I {} tail -f logs/train/{}

# 특정 로그 파일 확인
tail -f logs/train/train_20250905-1400_v087-abc123.log

# 학습 진행률 확인 (에포크별)
grep "Epoch" logs/train/train_*.log | tail -10
```

#### **학습 중단 및 재시작**
```bash
# 학습 중단
pkill -f "python src/training/train_main.py"

# 체크포인트에서 재시작 (구현 필요시)
python src/training/train_main.py --mode highperf --resume experiments/train/20250905/fold_0/checkpoint.pth
```

---

## 🔮 추론 파이프라인

### **1. 기본 추론**

#### **단일 모델 추론**
```bash
# 기본 추론
python src/inference/infer_main.py

# 설정 파일 지정
python src/inference/infer_main.py --config configs/infer.yaml

# 특정 모델 파일 사용
python src/inference/infer_main.py --model-path experiments/train/20250905/fold_0/best_model.pth

# GPU 지정
CUDA_VISIBLE_DEVICES=0 python src/inference/infer_main.py
```

### **2. 고성능 추론 (Fold 앙상블)**

#### **Fold 결과 앙상블**
```bash
# 고성능 추론 (모든 fold 앙상블)
python src/inference/infer_main.py --mode highperf --fold-results experiments/train/20250905/fold_results

# 특정 fold들만 앙상블
python src/inference/infer_main.py --mode highperf --fold-results experiments/train/20250905/fold_results --folds 0,1,2,3,4

# 앙상블 방법 지정
python src/inference/infer_main.py --mode highperf --fold-results experiments/train/20250905/fold_results --ensemble-method mean
```

#### **실행 예시**
```bash
# 날짜별 실험 결과 확인
ls -la experiments/train/

# 최신 실험의 fold 결과 사용
LATEST_EXP=$(ls -t experiments/train/ | head -1)
python src/inference/infer_main.py --mode highperf --fold-results experiments/train/$LATEST_EXP/fold_results

# 출력 예시:
# 📋 추론 설정 로드 완료
# 🎯 모델: swin_base_patch4_window7_224
# 📁 Fold 결과 디렉토리: experiments/train/20250905/fold_results
# 🔍 발견된 fold 모델: [0, 1, 2, 3, 4]
# 📊 테스트 데이터: 3,141개 샘플
# [1/5] Fold 0 추론 완료
# ...
# 🎯 앙상블 추론 완료
# 💾 제출 파일 저장: submissions/20250905/submission.csv
```

### **3. 추론 결과 확인**

#### **제출 파일 검증**
```bash
# 최신 제출 파일 확인
ls -la submissions/$(ls -t submissions/ | head -1)/

# 제출 파일 형식 확인
head -10 submissions/$(ls -t submissions/ | head -1)/*.csv

# 제출 파일 통계
python3 -c "
import pandas as pd
import glob
latest_sub = sorted(glob.glob('submissions/*/submission.csv'))[-1]
df = pd.read_csv(latest_sub)
print(f'제출 파일: {latest_sub}')
print(f'샘플 수: {len(df)}')
print(f'클래스 분포: {df.iloc[:, 1].value_counts().head()}')
"
```

---

## 🔄 전체 파이프라인

### **1. 완전 자동화 파이프라인**
```bash
# 학습 + 추론 전체 파이프라인
python src/pipeline/full_pipeline.py

# 설정 파일 지정
python src/pipeline/full_pipeline.py --train-config configs/train_highperf.yaml --infer-config configs/infer.yaml

# 백그라운드 실행
nohup python src/pipeline/full_pipeline.py > full_pipeline.log 2>&1 &
```

### **2. 단계별 실행**
```bash
# 1단계: 고성능 학습
echo "🎓 1단계: 학습 시작..."
python src/training/train_main.py --mode highperf

# 2단계: 결과 확인
echo "📊 2단계: 학습 결과 확인..."
ls -la experiments/train/$(ls -t experiments/train/ | head -1)/fold_results/

# 3단계: 앙상블 추론
echo "🔮 3단계: 앙상블 추론 시작..."
LATEST_EXP=$(ls -t experiments/train/ | head -1)
python src/inference/infer_main.py --mode highperf --fold-results experiments/train/$LATEST_EXP/fold_results

# 4단계: 결과 검증
echo "✅ 4단계: 최종 결과 확인..."
ls -la submissions/$(ls -t submissions/ | head -1)/
```

---

## 🧪 단위 테스트

### **1. Jupyter 노트북 실행**

#### **노트북 서버 시작**
```bash
# Jupyter Notebook 시작
jupyter notebook --port=8888 --no-browser

# JupyterLab 시작 (권장)
jupyter lab --port=8888 --no-browser

# 백그라운드 실행
nohup jupyter lab --port=8888 --no-browser > jupyter.log 2>&1 &
```

#### **개별 단위 테스트 노트북**
```bash
# 1. 고성능 데이터셋 테스트
# notebooks/unit_tests/test_highperf_dataset.ipynb

# 2. Mixup 증강 테스트  
# notebooks/unit_tests/test_mixup_augmentation.ipynb

# 3. Swin 모델 테스트
# notebooks/unit_tests/test_swin_model.ipynb

# 4. 로깅 통합 테스트 (예시)
# notebooks/test_highperf_dataset_with_logging.ipynb
```

### **2. 통합 테스트 노트북**

#### **전체 파이프라인 테스트**
```bash
# 전체 파이프라인 검증
# notebooks/test_full_pipeline.ipynb

# WandB 통합 테스트
# notebooks/test_wandb_integration.ipynb
```

### **3. 명령줄에서 노트북 실행**
```bash
# nbconvert로 노트북 실행 (선택적)
jupyter nbconvert --to notebook --execute notebooks/unit_tests/test_highperf_dataset.ipynb

# 결과 HTML로 변환
jupyter nbconvert --to html notebooks/unit_tests/test_highperf_dataset.ipynb
```

---

## 📊 로그 및 결과 확인

### **1. 학습 로그 분석**

#### **기본 로그 확인**
```bash
# 최신 학습 로그 확인
ls -t logs/train/ | head -5

# 특정 로그 내용 확인
cat logs/train/train_20250905-1400_v087-abc123.log

# 에러 로그만 확인
grep -i "error\|fail" logs/train/train_*.log

# 성능 지표 추출
grep "Val F1\|Best F1" logs/train/train_*.log | tail -10
```

#### **실시간 모니터링**
```bash
# 실시간 로그 스트림
tail -f logs/train/train_*.log

# 특정 패턴 모니터링
tail -f logs/train/train_*.log | grep --line-buffered "Epoch\|F1"

# 다중 로그 모니터링
multitail logs/train/train_*.log logs/infer/infer_*.log
```

### **2. 추론 로그 분석**

#### **추론 결과 확인**
```bash
# 최신 추론 로그
ls -t logs/infer/ | head -3

# 추론 성능 확인
grep "Inference completed\|Processing time" logs/infer/infer_*.log

# 앙상블 결과 확인
grep "Ensemble\|Average" logs/infer/infer_*.log
```

### **3. 단위 테스트 로그 (로깅 시스템 사용시)**

#### **로깅 시스템 결과 확인**
```bash
# 단위 테스트 로그 디렉토리
ls -la logs/unit_test/

# 특정 테스트 결과 확인
TEST_NAME="highperf_dataset"
LATEST_RUN=$(ls -t logs/unit_test/$TEST_NAME/ | head -1)
ls -la logs/unit_test/$TEST_NAME/$LATEST_RUN/

# 테스트 요약 JSON 확인
cat logs/unit_test/$TEST_NAME/$LATEST_RUN/test_summary.json | jq

# 저장된 이미지 확인
ls logs/unit_test/$TEST_NAME/$LATEST_RUN/images/

# 처리된 데이터 확인
ls logs/unit_test/$TEST_NAME/$LATEST_RUN/data/
```

### **4. 실험 결과 관리**

#### **실험 디렉토리 구조**
```bash
# 실험 결과 확인
tree experiments/ -L 3

# 특정 날짜 실험 확인
ls -la experiments/train/20250905/

# Fold별 결과 확인
ls -la experiments/train/20250905/fold_results/

# 모델 파일 확인
find experiments/ -name "*.pth" -type f | head -10
```

#### **제출 파일 관리**
```bash
# 제출 파일 히스토리
ls -la submissions/

# 최신 제출 파일 분석
python3 -c "
import pandas as pd
import glob
import os

# 최신 제출 파일 찾기
submission_files = glob.glob('submissions/*/submission.csv')
if submission_files:
    latest = max(submission_files, key=os.path.getctime)
    df = pd.read_csv(latest)
    print(f'최신 제출 파일: {latest}')
    print(f'생성 시간: {pd.Timestamp.fromtimestamp(os.path.getctime(latest))}')
    print(f'파일 크기: {len(df)} 행')
    print(f'클래스 분포:')
    print(df.iloc[:, 1].value_counts().head())
else:
    print('제출 파일이 없습니다.')
"
```

---

## 🎛️ 고급 사용법

### **1. 환경 변수 활용**

#### **GPU 관리**
```bash
# 특정 GPU 사용
export CUDA_VISIBLE_DEVICES=0
python src/training/train_main.py --mode highperf

# 멀티 GPU 사용
export CUDA_VISIBLE_DEVICES=0,1
python src/training/train_main.py --mode highperf

# GPU 메모리 제한
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python src/training/train_main.py --mode highperf
```

#### **WandB 설정**
```bash
# WandB 프로젝트 설정
export WANDB_PROJECT="cv-competition-1sen"
export WANDB_ENTITY="your-team-name"

# WandB 실행 이름 설정
export WANDB_RUN_NAME="highperf-swin-fold-all"
python src/training/train_main.py --mode highperf
```

#### **로깅 레벨 설정**
```bash
# 디버그 모드
export LOG_LEVEL=DEBUG
python src/training/train_main.py --mode highperf

# 조용한 모드
export LOG_LEVEL=WARNING
python src/training/train_main.py --mode highperf
```

### **2. 배치 스크립트**

#### **전체 실험 자동화**
```bash
#!/bin/bash
# run_full_experiment.sh

set -e  # 에러시 중단

echo "🚀 전체 실험 시작: $(date)"

# 1. 환경 확인
echo "📋 환경 확인..."
python3 -c "from src.utils.common import load_yaml; print('✅ 환경 준비 완료')"

# 2. 학습 실행
echo "🎓 학습 시작..."
python src/training/train_main.py --mode highperf

# 3. 최신 실험 결과 찾기
LATEST_EXP=$(ls -t experiments/train/ | head -1)
echo "📁 최신 실험: $LATEST_EXP"

# 4. 추론 실행
echo "🔮 추론 시작..."
python src/inference/infer_main.py --mode highperf --fold-results experiments/train/$LATEST_EXP/fold_results

# 5. 결과 확인
echo "✅ 실험 완료: $(date)"
ls -la submissions/$(ls -t submissions/ | head -1)/

echo "🎯 실험 요약:"
echo "   학습 결과: experiments/train/$LATEST_EXP/"
echo "   제출 파일: submissions/$(ls -t submissions/ | head -1)/"
```

#### **스크립트 실행**
```bash
# 실행 권한 부여
chmod +x run_full_experiment.sh

# 백그라운드 실행
nohup ./run_full_experiment.sh > experiment.log 2>&1 &

# 진행 상황 모니터링
tail -f experiment.log
```

### **3. 성능 최적화**

#### **메모리 최적화**
```bash
# 배치 크기 줄이기
sed -i 's/batch_size: 16/batch_size: 8/' configs/train_highperf.yaml

# 이미지 크기 줄이기
sed -i 's/img_size: 384/img_size: 224/' configs/train_highperf.yaml

# 메모리 사용량 모니터링
watch -n 1 nvidia-smi
```

#### **속도 최적화**
```bash
# 데이터 로더 워커 수 증가
export NUM_WORKERS=8
python src/training/train_main.py --mode highperf

# 혼합 정밀도 학습 (구현시)
export USE_AMP=true
python src/training/train_main.py --mode highperf
```

---

## 🔧 문제 해결

### **1. 일반적인 문제**

#### **Import 에러**
```bash
# 문제: ModuleNotFoundError
# 해결: Python 경로 확인
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 또는 sys.path 추가
python3 -c "import sys; sys.path.append('.'); from src.utils.common import load_yaml"
```

#### **CUDA 에러**
```bash
# 문제: CUDA out of memory
# 해결: 배치 크기 줄이기
export BATCH_SIZE=4
python src/training/train_main.py --fold 0

# GPU 메모리 정리
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### **파일 경로 에러**
```bash
# 문제: 데이터 파일을 찾을 수 없음
# 해결: 현재 디렉토리 확인
pwd
ls -la data/raw/

# 프로젝트 루트로 이동
cd /home/ieyeppo/AI_Lab/computer-vision-competition-1SEN
```

### **2. 로그 기반 디버깅**

#### **에러 로그 분석**
```bash
# 최근 에러 찾기
grep -r -i "error\|exception\|failed" logs/ | tail -10

# 특정 에러 패턴 찾기
grep -r "CUDA\|memory\|import" logs/ | tail -5

# 스택트레이스 확인
grep -A 10 -B 2 "Traceback" logs/train/train_*.log
```

#### **성능 이슈 디버깅**
```bash
# 학습 속도 확인
grep "Epoch.*Time" logs/train/train_*.log | tail -5

# 메모리 사용량 확인
grep -i "memory\|GPU" logs/train/train_*.log | tail -5

# 데이터 로딩 시간 확인
grep -i "loading\|batch" logs/train/train_*.log | tail -5
```

### **3. 검증 명령어**

#### **전체 시스템 검증**
```bash
# 종합 건강도 검사
python3 -c "
import sys
sys.path.append('.')

# 1. 모듈 import 테스트
try:
    from src.utils.common import load_yaml
    from src.data.dataset import HighPerfDocClsDataset
    from src.models.build import build_model
    print('✅ 모든 모듈 import 성공')
except Exception as e:
    print(f'❌ 모듈 import 실패: {e}')
    sys.exit(1)

# 2. 설정 파일 테스트
try:
    cfg = load_yaml('configs/train_highperf.yaml')
    print('✅ 설정 파일 로드 성공')
except Exception as e:
    print(f'❌ 설정 파일 로드 실패: {e}')
    sys.exit(1)

# 3. 데이터 파일 테스트
import os
required_files = [
    'data/raw/train.csv',
    'data/raw/meta.csv', 
    'data/raw/sample_submission.csv'
]
for file in required_files:
    if os.path.exists(file):
        print(f'✅ {file} 존재')
    else:
        print(f'❌ {file} 없음')

print('🎯 시스템 검증 완료!')
"
```

#### **성능 벤치마크**
```bash
# 빠른 학습 테스트 (1 에포크)
python3 -c "
import sys
sys.path.append('.')
import time
from src.training.train import train_model
from src.utils.common import load_yaml

cfg = load_yaml('configs/train.yaml')
cfg['training']['epochs'] = 1
cfg['training']['batch_size'] = 4

start_time = time.time()
# train_model(cfg, fold=0)  # 실제 실행시 주석 해제
end_time = time.time()

print(f'⏱️ 1 에포크 예상 시간: {end_time - start_time:.1f}초')
"
```

---

## 📞 추가 지원

### **로그 수집**
```bash
# 문제 발생시 로그 수집
mkdir -p debug_logs/$(date +%Y%m%d_%H%M%S)
cp -r logs/ debug_logs/$(date +%Y%m%d_%H%M%S)/
cp configs/*.yaml debug_logs/$(date +%Y%m%d_%H%M%S)/
```

### **시스템 정보 수집**
```bash
# 시스템 정보 저장
echo "=== 시스템 정보 ===" > system_info.txt
date >> system_info.txt
uname -a >> system_info.txt
python3 --version >> system_info.txt
pip list | grep -E "torch|pandas|numpy" >> system_info.txt
nvidia-smi >> system_info.txt
```

**🎯 이 가이드를 통해 모든 실행 시나리오를 커버할 수 있습니다!** 🚀
