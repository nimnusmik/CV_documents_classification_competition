# src/training/__init__.py
"""
모델 학습 패키지

이 패키지는 컴퓨터 비전 경진대회를 위한 다양한 학습 모드를 제공합니다.
- 기본 학습 모드 (train.py)
- 고성능 학습 모드 (train_highperf.py) 
- CLI 실행 인터페이스 (train_main.py)

각 모드는 독립적으로 실행 가능하며, 다양한 설정과 최적화 옵션을 지원합니다.
"""

# ==================== 패키지 정보 ==================== #
__version__ = "1.0.0"                                      # 패키지 버전
__author__ = "Computer Vision Competition Team"             # 작성자 정보

# ==================== 학습 함수 임포트 ==================== #
# 임포트할 함수들을 위한 타입 정의
from typing import Callable, Any, Optional                  # 타입 힌트 임포트

# 기본 학습 함수 임포트
_run_training: Optional[Callable[[str], Any]] = None        # 기본 학습 함수 변수 초기화
# 예외 처리 시작
try:
    from .train import run_training as _run_training_func   # 기본 학습 실행 함수
    _run_training = _run_training_func                      # 함수 변수에 할당
    _run_training_available = True                          # 기본 학습 함수 사용 가능 플래그
# 임포트 실패 시
except ImportError as e:
    import warnings                                         # 경고 모듈 임포트
    warnings.warn(f"기본 학습 함수 임포트 실패: {e}")           # 경고 메시지 출력
    _run_training_available = False                         # 기본 학습 함수 사용 불가 플래그

# 고성능 학습 함수 임포트
_run_highperf_training: Optional[Callable[[str], Any]] = None  # 고성능 학습 함수 변수 초기화
# 예외 처리 시작
try:
    from .train_highperf import run_highperf_training as _run_highperf_training_func  # 고성능 학습 실행 함수
    _run_highperf_training = _run_highperf_training_func    # 함수 변수에 할당
    _run_highperf_training_available = True                 # 고성능 학습 함수 사용 가능 플래그
# 임포트 실패 시
except ImportError as e:
    import warnings                                         # 경고 모듈 임포트
    warnings.warn(f"고성능 학습 함수 임포트 실패: {e}")         # 경고 메시지 출력
    _run_highperf_training_available = False                # 고성능 학습 함수 사용 불가 플래그

# CLI 메인 함수 임포트 (선택적)
_train_main: Optional[Callable[[], Any]] = None             # CLI 메인 함수 변수 초기화
# 예외 처리 시작
try:
    from .train_main import main as _train_main_func        # CLI 메인 함수 (이름 변경)
    _train_main = _train_main_func                          # 함수 변수에 할당
    _train_main_available = True                            # CLI 메인 함수 사용 가능 플래그
# 임포트 실패 시
except ImportError as e:
    import warnings                                         # 경고 모듈 임포트
    warnings.warn(f"CLI 메인 함수 임포트 실패: {e}")           # 경고 메시지 출력
    _train_main_available = False                           # CLI 메인 함수 사용 불가 플래그

#------------------- 패키지 레벨 함수 재할당 ------------------- #
# 기본 학습 함수가 사용 가능한 경우
if _run_training_available and _run_training:
    run_training = _run_training                            # 패키지 레벨에서 직접 접근 가능하게 함

# 고성능 학습 함수가 사용 가능한 경우
if _run_highperf_training_available and _run_highperf_training:
    run_highperf_training = _run_highperf_training          # 패키지 레벨에서 직접 접근 가능하게 함

# CLI 메인 함수가 사용 가능한 경우
if _train_main_available and _train_main:
    train_main = _train_main                                # 패키지 레벨에서 직접 접근 가능하게 함

# ==================== 공개 API 정의 ==================== #
# 패키지에서 공개할 함수들 정의
# 공개 API 리스트
__all__ = [
    # 학습 실행 함수들
    "run_training",                                         # 기본 학습 함수
    "run_highperf_training",                                # 고성능 학습 함수
    "train_main",                                           # CLI 메인 함수
    
    # 패키지 메타 정보
    "__version__",                                          # 버전 정보
    "__author__",                                           # 작성자 정보
]

# ==================== 편의 함수 ==================== #
# 학습 모드별 실행 편의 함수 정의
def run_basic_training(config_path: str):
    # 기본 학습 함수가 임포트된 경우에만 실행
    if _run_training_available and _run_training:
        return _run_training(config_path)                   # 기본 학습 실행
    # 함수가 없는 경우
    else:
        raise ImportError("기본 학습 함수를 사용할 수 없습니다.")# 에러 발생


# 고성능 학습 모드 실행 편의 함수 정의
def run_performance_training(config_path: str):
    # 고성능 학습 함수가 임포트된 경우에만 실행
    # run_highperf_training 함수가 사용 가능한 경우
    if _run_highperf_training_available and _run_highperf_training:
        return _run_highperf_training(config_path)          # 고성능 학습 실행
    # 함수가 없는 경우
    else:
        raise ImportError("고성능 학습 함수를 사용할 수 없습니다.")# 에러 발생


# 편의 함수들도 공개 API에 추가
__all__.extend([                                            # 공개 API 리스트 확장
    "run_basic_training",                                   # 기본 학습 편의 함수
    "run_performance_training",                             # 고성능 학습 편의 함수
])
