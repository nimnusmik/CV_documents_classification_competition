# 프로젝트 전체 스펙 및 로그 분석 요약

## 1. 디렉토리 구조 및 주요 파일 설명

```
AI_Lab/computer-vision-competition-1SEN/
├── configs/           # 파이프라인 설정 파일 (train/infer)
├── data/              # 원본 및 분할 데이터, 테스트 이미지
├── docs/              # 프로젝트 가이드 및 분석 문서
├── experiments/       # 학습 결과, 모델 체크포인트, fold별 결과
├── logs/              # 학습/추론/파이프라인 실행 로그
├── notebooks/         # 실험/분석/모듈화 노트북
├── src/               # 파이프라인, 모델, 유틸리티 코드
├── submissions/       # 추론 결과 제출 파일(csv)
├── wandb/             # WandB 실험 로그
├── requirements.txt   # 패키지 의존성
└── run_highperf_training.sh # 고성능 학습 실행 스크립트
```

### 주요 파일 및 폴더 관계
- `configs/train_highperf.yaml`, `configs/infer.yaml`: 파이프라인 하이퍼파라미터, 데이터 경로, 모델명 등 설정
- `src/pipeline/full_pipeline.py`: 전체 학습+추론 파이프라인, config 기반 실행
- `src/models/`, `src/training/`, `src/inference/`: 모델 정의, 학습/추론 로직
- `notebooks/`: 실험별 모델, 하이퍼파라미터, 앙상블, TTA 등 모듈화 코드
- `experiments/`: fold별 체크포인트, 결과 yaml, 성능 로그
- `submissions/`: 추론 결과 csv 파일 (대회 제출용)
- `logs/`: 실행 로그 (학습/추론/파이프라인)

## 2. 전체 작업 흐름도
1. pyenv 환경 활성화 및 패키지 설치
2. GPU 체크 및 자동 배치사이즈 최적화
3. 학습 파이프라인 실행 (K-Fold, 앙상블, TTA)
4. 추론 파이프라인 실행 (테스트셋, 앙상블, TTA)
5. 결과 csv 생성 및 제출
6. WandB 실험 관리 및 성능 모니터링

## 3. 하이퍼파라미터/모델/앙상블 스펙
- 모델: `swin_base_384`, `convnext_base_384_in22ft1k`, `efficientnet_b3` 등
- 하이퍼파라미터:
    - img_size: 384
    - batch_size: 32 (추론시 64)
    - epochs: 15~30
    - optimizer: AdamW
    - scheduler: CosineAnnealingLR
    - label_smoothing: 0.05~0.2
    - mixup/hard augmentation/EMA 등 고급 기법 적용
- 앙상블: 5-Fold 모델 앙상블, TTA(Test-Time Augmentation) 적용
- 데이터 경로: `data/raw/train`, `data/raw/test`, `data/raw/sample_submission.csv`

## 4. 최근 실행 로그 분석
- 학습: 평균 F1 0.9356 (fold별 0.92~0.94)
- 추론: 3140개 테스트 이미지, 클래스별 분포 고름
- 제출 파일: `submissions/20250905/v094-swin-highperf_ensemble_20250905_1714.csv`
- WandB 실험 URL, fold별 성능, 클래스별 정확도 시각화

## 5. 파일간 관계도
- configs → src/pipeline → src/models/training/inference → experiments/logs/submissions
- notebooks: 실험/분석/모듈화 코드, 파이프라인과 설정 연동
- docs: 전체 가이드, 실행법, 분석, 워크플로우 설명

## 6. 기존 docs 가이드 파일 분석 및 수정 필요 사항
- 모든 가이드에 pyenv 환경, GPU 체크, 자동 배치사이즈, K-Fold/앙상블/TTA, WandB 실험 관리 등 최신 워크플로우 반영 필요
- 데이터 경로, 모델명, 하이퍼파라미터, 앙상블 방식 등 최신 스펙으로 통일 필요
- 파이프라인 실행법, 결과 확인법, 제출 파일 생성 과정 명확히 기술

## 7. 결론 및 참고
- 전체 파이프라인, 하이퍼파라미터, 앙상블, 모델, 실험 관리가 모듈화되어 팀원 누구나 동일한 방식으로 실행 가능
- 모든 코드/설정/가이드가 최신 워크플로우에 맞게 통일되어 있음
- 추가 실험/분석/가이드가 필요하면 docs에 계속 추가/수정 가능

---
(2025-09-05 기준 최신 스펙/구조/실행 결과 반영)
