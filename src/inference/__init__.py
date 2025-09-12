# ------------------------- inference 패키지 초기화 모듈 ------------------------- #
# 추론 관련 모듈들을 모아놓은 패키지입니다.
# 기본 추론(infer.py), 고성능 추론(infer_highperf.py), 메인 추론 함수(infer_main.py) 등이 포함됩니다.

# ------------------------- 기본 추론 함수 Import ------------------------- #
from .infer import run_inference                            # 기본 추론 실행 함수

# ------------------------- 고성능 추론 함수 Import ------------------------- #
from .infer_highperf import (                               # 고성능 추론 관련 함수들
    run_highperf_inference,                                 # 고성능 추론 실행 함수
    predict_with_tta,                                       # TTA 예측 함수
    load_fold_models,                                       # 폴드 모델 로드 함수
    ensemble_predict                                        # 앙상블 예측 함수
)

# ------------------------- 메인 실행 함수 Import ------------------------- #
from .infer_main import main as inference_main             # CLI 메인 함수 (이름 충돌 방지)

# ------------------------- 외부 노출 함수 정의 ------------------------- #
__all__ = [                                                 # 패키지에서 외부로 노출할 함수들
    # 기본 추론
    'run_inference',                                        # 기본 추론 실행
    
    # 고성능 추론
    'run_highperf_inference',                               # 고성능 추론 실행
    'predict_with_tta',                                     # TTA 예측
    'load_fold_models',                                     # 폴드 모델 로드
    'ensemble_predict',                                     # 앙상블 예측
    
    # CLI 메인 함수
    'inference_main',                                       # CLI 추론 메인 함수
]