# 🔄 update_inference_date.sh 가이드

## 개요
추론 설정 파일의 날짜를 빠르게 업데이트하여 최신 학습된 모델을 사용할 수 있도록 하는 유틸리티 스크립트입니다.

## 주요 기능
- 📅 **자동 날짜 탐지**: 최신 실험 날짜 자동 감지
- 🔧 **설정 파일 업데이트**: 추론 config 파일들의 날짜 자동 수정
- 💾 **백업 생성**: 원본 파일 안전 보관
- 🎯 **다중 파일 지원**: 여러 추론 설정 파일 동시 업데이트

## 사용법

### 1. 최신 날짜로 자동 업데이트
```bash
# 가장 최신 실험 날짜로 자동 업데이트
./scripts/update_inference_date.sh --latest
```

### 2. 특정 날짜로 업데이트
```bash
# 특정 날짜 지정 (YYYYMMDD 형식)
./scripts/update_inference_date.sh 20250908
```

### 3. 대화형 모드
```bash
# 대화형으로 날짜 선택
./scripts/update_inference_date.sh
```

## 실행 과정 예시

### 자동 날짜 감지 모드
```
🔄 추론 설정 날짜 업데이트 유틸리티
========================================

📅 가장 최신 날짜: 20250908

🔍 업데이트할 설정 파일 검색 중...
발견된 파일들:
✓ configs/infer.yaml
✓ configs/infer_highperf.yaml
✓ configs/infer_calibrated.yaml

📝 백업 파일 생성 중...
✓ configs/infer.yaml.backup.20250908_0115
✓ configs/infer_highperf.yaml.backup.20250908_0115
✓ configs/infer_calibrated.yaml.backup.20250908_0115

🔄 날짜 업데이트 중...
✓ configs/infer.yaml: 20250907 → 20250908
✓ configs/infer_highperf.yaml: 20250907 → 20250908
✓ configs/infer_calibrated.yaml: 20250907 → 20250908

✅ 업데이트 완료!
```

### 특정 날짜 지정 모드
```bash
./scripts/update_inference_date.sh 20250905

🔄 추론 설정 날짜 업데이트 유틸리티
========================================

📅 지정된 날짜: 20250905

⚠️  주의: 지정된 날짜의 실험 디렉터리가 존재하는지 확인하세요.
실험 디렉터리: experiments/train/20250905

계속하시겠습니까? (y/N): y

[업데이트 진행...]
```

## 대상 설정 파일

### 자동 감지되는 파일들
```
configs/
├── infer.yaml              # 기본 추론 설정
├── infer_highperf.yaml     # 고성능 추론 설정
├── infer_calibrated.yaml   # 캘리브레이션 추론 설정
└── infer_ensemble.yaml     # 앙상블 추론 설정 (있는 경우)
```

### 업데이트되는 필드들
```yaml
# 수정되는 YAML 필드들
model:
  checkpoint_date: "20250908"    # 모델 체크포인트 날짜
  
paths:
  model_dir: "experiments/train/20250908"  # 모델 디렉터리 경로
  
inference:
  output_date: "20250908"        # 출력 파일 날짜
```

## 고급 사용법

### 1. 특정 파일만 업데이트
```bash
# 환경 변수로 파일 지정
INFERENCE_CONFIGS="configs/infer.yaml,configs/infer_highperf.yaml" \
./scripts/update_inference_date.sh --latest
```

### 2. 백업 없이 업데이트
```bash
# --no-backup 옵션 (위험)
./scripts/update_inference_date.sh --latest --no-backup
```

### 3. 드라이런 모드
```bash
# 실제 수정 없이 미리보기
./scripts/update_inference_date.sh --latest --dry-run
```

## 안전 기능

### 백업 시스템
```bash
# 백업 파일 명명 규칙
원본파일.backup.YYYYMMDD_HHMM

# 예시
configs/infer.yaml.backup.20250908_0115
configs/infer_highperf.yaml.backup.20250908_0115
```

### 백업 복원
```bash
# 백업에서 복원
cp configs/infer.yaml.backup.20250908_0115 configs/infer.yaml

# 또는 스크립트 내장 복원 기능
./scripts/update_inference_date.sh --restore 20250908_0115
```

### 유효성 검사
```bash
# 날짜 형식 검증 (YYYYMMDD)
if [[ ! $date =~ ^[0-9]{8}$ ]]; then
    echo "❌ 잘못된 날짜 형식입니다. YYYYMMDD 형식을 사용하세요."
    exit 1
fi

# 실험 디렉터리 존재 확인
if [ ! -d "experiments/train/$date" ]; then
    echo "⚠️ 경고: experiments/train/$date 디렉터리가 존재하지 않습니다."
fi
```

## 문제 해결

### 실험 디렉터리가 없는 경우
```bash
# 사용 가능한 날짜 확인
ls experiments/train/ | grep -E "^[0-9]{8}$" | sort

# 출력 예시:
20250905
20250906
20250907
20250908
```

### 권한 문제
```bash
# 스크립트 실행 권한 확인
ls -la scripts/update_inference_date.sh

# 권한 부여
chmod +x scripts/update_inference_date.sh
```

### 설정 파일 손상
```bash
# YAML 문법 검사
python -c "import yaml; yaml.safe_load(open('configs/infer.yaml'))"

# 백업에서 복원
cp configs/infer.yaml.backup.최신백업 configs/infer.yaml
```

## 스크립트 커스터마이징

### 새로운 설정 파일 추가
```bash
# 스크립트 내 CONFIG_FILES 배열 수정
CONFIG_FILES=(
    "configs/infer.yaml"
    "configs/infer_highperf.yaml"
    "configs/infer_calibrated.yaml"
    "configs/infer_ensemble.yaml"     # 새로 추가
    "configs/custom_infer.yaml"       # 커스텀 설정
)
```

### 날짜 형식 변경
```bash
# 다른 날짜 형식 지원 (예: YYYY-MM-DD)
if [[ $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    TARGET_DATE=$(echo $1 | sed 's/-//g')
fi
```

## 자동화 예시

### Cron으로 자동 업데이트
```bash
# crontab 편집
crontab -e

# 매일 새벽 2시에 자동 업데이트
0 2 * * * cd /path/to/project && ./scripts/update_inference_date.sh --latest
```

### 학습 완료 후 자동 실행
```bash
# 학습 스크립트 마지막에 추가
if [ $? -eq 0 ]; then
    echo "학습 완료. 추론 설정 업데이트 중..."
    ./scripts/update_inference_date.sh --latest
fi
```

## 통합 워크플로우

### 전체 파이프라인 예시
```bash
# 1. 고성능 학습 실행
./scripts/run_highperf_training.sh

# 2. 자동으로 추론 설정 업데이트
./scripts/update_inference_date.sh --latest

# 3. 업데이트된 설정으로 추론 실행
python src/inference/infer_highperf.py --config configs/infer_highperf.yaml

# 4. 결과 확인
ls submissions/$(date +%Y%m%d)/
```

## 로그 및 히스토리

### 업데이트 히스토리 확인
```bash
# 백업 파일로 업데이트 히스토리 추적
ls configs/*.backup.* | sort

# 특정 날짜의 변경 사항 확인
diff configs/infer.yaml.backup.20250907_1530 configs/infer.yaml
```

### 업데이트 로그
```bash
# 스크립트 실행 로그 저장
./scripts/update_inference_date.sh --latest 2>&1 | \
tee logs/update_inference_$(date +%Y%m%d_%H%M).log
```

## 관련 파일
- `configs/infer*.yaml`: 추론 설정 파일들
- `experiments/train/YYYYMMDD/`: 학습 결과 디렉터리
- `scripts/monitor_training.sh`: 학습 모니터링
- `src/inference/`: 추론 스크립트들

## 참고 명령어
```bash
# 최신 실험 날짜 확인
ls experiments/train/ | grep -E "^[0-9]{8}$" | sort | tail -1

# 특정 날짜의 모델 파일 확인
find experiments/train/20250908 -name "*.pth" | head -5

# 추론 설정 파일 문법 검사
find configs/ -name "infer*.yaml" -exec python -m yaml {} \;
```

> **팁**: 학습이 완료될 때마다 이 스크립트를 실행하여 항상 최신 모델을 사용할 수 있도록 하세요.
