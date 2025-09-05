# 🚀 로깅 시스템 통합 사용 가이드

## 📁 로깅 시스템 개요

로깅 시스템이 성공적으로 모든 단위 테스트 노트북에 통합되었습니다! 

### 🎯 주요 특징
- **자동 디렉토리 생성**: `logs/unit_test/{test_name}/{timestamp}/`
- **포괄적 출력 캡처**: print문, 에러, 경고 메시지 모두 자동 저장
- **시각화 자동 저장**: matplotlib 그래프를 PNG/SVG로 저장 + 메타데이터
- **데이터 자동 백업**: 처리된 DataFrame과 NumPy 배열 저장
- **성능 메트릭**: JSON 형태로 테스트 결과와 성능 지표 저장

## 📂 생성되는 디렉토리 구조

```
logs/unit_test/
├── highperf_dataset/
│   └── 20250905_143052/
│       ├── logs/                    # 텍스트 로그
│       │   ├── basic_data_analysis.log
│       │   ├── dataset_class_test.log
│       │   └── hard_augmentation_analysis.log
│       ├── images/                  # 시각화 결과
│       │   ├── class_distribution_analysis.png
│       │   └── hard_augmentation_schedule_visualization.png
│       ├── data/                    # 처리된 데이터
│       │   ├── sample_train_data.csv
│       │   └── class_distribution.csv
│       ├── results/                 # 테스트 결과
│       │   ├── basic_data_analysis.json
│       │   └── dataset_class_test.json
│       └── test_summary.json        # 전체 요약
├── mixup_augmentation/
│   └── 20250905_143125/
│       └── ...
└── swin_model/
    └── 20250905_143200/
        └── ...
```

## 🛠️ 로깅 시스템 API 사용법

### 1. 기본 초기화
```python
from src.utils.unit_test_logger import create_test_logger

# 로거 생성 (테스트 이름 지정)
test_logger = create_test_logger("my_test_name")
test_logger.log_info("테스트 시작")
```

### 2. 출력 캡처 (가장 중요!)
```python
# 모든 print문과 에러를 자동으로 로그 파일에 저장
with test_logger.capture_output("section_name") as (output, error):
    print("이 출력은 자동으로 로그 파일에 저장됩니다!")
    print(f"데이터 크기: {len(data):,}개")
    
    try:
        # 여기서 발생하는 모든 출력과 에러가 캡처됨
        result = some_computation()
        print(f"계산 결과: {result}")
    except Exception as e:
        print(f"에러 발생: {e}")
        raise
```

### 3. 시각화 저장
```python
# matplotlib 그래프 자동 저장
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_data, y_data)
ax.set_title("중요한 결과 그래프")

# PNG + SVG + 메타데이터 자동 저장
test_logger.save_figure(fig, "important_graph", "상세한 설명")
plt.show()
plt.close()
```

### 4. 데이터 저장
```python
# DataFrame 자동 저장 (CSV + 메타데이터)
test_logger.save_dataframe(
    df, 
    "processed_data", 
    "전처리된 학습 데이터"
)

# NumPy 배열 저장
test_logger.save_numpy_array(
    array, 
    "feature_matrix", 
    "추출된 특성 행렬"
)
```

### 5. 테스트 결과 저장
```python
# JSON 형태로 구조화된 결과 저장
test_logger.save_test_result("performance_test", {
    "status": "success",
    "accuracy": 0.95,
    "training_time_sec": 1200,
    "model_params": 25000000,
    "recommendations": ["GPU 메모리 최적화 필요"]
})
```

### 6. 성능 메트릭 저장
```python
# 성능 지표를 체계적으로 저장
metrics = {
    "data_loading": {
        "avg_batch_time_sec": 0.05,
        "memory_usage_mb": 2048
    },
    "model_performance": {
        "forward_pass_ms": 15.2,
        "backward_pass_ms": 32.1
    }
}
test_logger.save_performance_metrics(metrics, "benchmark_results")
```

### 7. 테스트 완료 및 요약
```python
# 테스트 종료 및 자동 요약 생성
final_summary = test_logger.finalize_test()
print(f"테스트 완료! 결과 저장 위치: {test_logger.base_dir}")
```

## 🎪 실제 노트북에서의 활용 예시

### 업데이트된 노트북들:
1. **`test_highperf_dataset_with_logging.ipynb`** ✅ (예제 생성 완료)
2. **`test_highperf_dataset.ipynb`** ✅ (로깅 통합 완료)
3. **`test_mixup_augmentation.ipynb`** ✅ (로깅 통합 완료)  
4. **`test_swin_model.ipynb`** ✅ (로깅 통합 완료)

### 각 노트북 실행 후 얻는 것:
- 📝 **완전한 실행 로그**: 모든 출력, 에러, 경고 메시지
- 📊 **자동 시각화 저장**: 고해상도 그래프와 차트
- 💾 **처리된 데이터 백업**: 중간 결과물들의 안전한 보관
- 📈 **성능 메트릭**: 실행 시간, 메모리 사용량, 정확도 등
- 🎯 **구조화된 결과**: JSON 형태의 체계적인 테스트 결과

## 🚦 로그 확인 방법

### 터미널에서 확인:
```bash
# 최신 로그 디렉토리 확인
ls -la logs/unit_test/

# 특정 테스트의 최신 실행 결과 확인
ls -la logs/unit_test/highperf_dataset/$(ls -t logs/unit_test/highperf_dataset/ | head -1)/

# 테스트 요약 확인
cat logs/unit_test/highperf_dataset/$(ls -t logs/unit_test/highperf_dataset/ | head -1)/test_summary.json | jq

# 이미지 파일 확인
ls logs/unit_test/highperf_dataset/$(ls -t logs/unit_test/highperf_dataset/ | head -1)/images/
```

### Python에서 확인:
```python
import json
import os
from pathlib import Path

# 최신 테스트 결과 로드
base_path = Path("logs/unit_test/highperf_dataset")
latest_run = max(base_path.glob("*"))

# 요약 정보 읽기
with open(latest_run / "test_summary.json") as f:
    summary = json.load(f)
    
print(f"테스트 시작: {summary['start_time']}")
print(f"테스트 종료: {summary['end_time']}")
print(f"저장된 파일: {len(summary['saved_files'])}개")
```

## 🏆 로깅 시스템의 장점

### 1. **완전성** 
- 놓치는 정보 없이 모든 실행 과정 기록
- 에러와 예외 상황도 안전하게 캡처

### 2. **재현성**
- 동일한 조건에서 결과 재현 가능
- 실험 설정과 결과가 함께 저장

### 3. **협업성**
- 팀원들과 결과 공유 용이
- 표준화된 로그 형식

### 4. **분석성**
- JSON 형태의 구조화된 데이터
- 시계열 분석과 성능 추적 가능

### 5. **자동화**
- 수동 개입 없이 모든 것이 자동 저장
- 일관된 디렉토리 구조 유지

## 🎯 다음 단계

이제 다음과 같이 활용할 수 있습니다:

1. **노트북 실행**: 기존 노트북들을 실행하면 자동으로 로그 저장
2. **결과 분석**: 저장된 로그와 데이터를 활용한 심층 분석
3. **성능 추적**: 시간에 따른 모델 성능 변화 모니터링
4. **팀 협업**: 표준화된 형태로 실험 결과 공유

🎉 **완벽한 로깅 시스템이 준비되었습니다!** 이제 모든 단위 테스트가 전문적인 수준의 로깅과 함께 진행됩니다.
