# ------------------------- src 패키지 초기화 모듈 ------------------------- #
# 이 파일은 src 디렉토리를 Python 패키지로 인식하게 하는 파일입니다.
# 프로젝트의 모든 소스코드 모듈들이 이 패키지 하위에 위치합니다.

# ------------------------- 프로젝트 버전 정보 ------------------------- #
__version__ = "1.0.0"                                      # 프로젝트 버전
__author__ = "AI Team"                                     # 프로젝트 개발팀
__description__ = "Computer Vision Competition Framework"   # 프로젝트 설명

# ------------------------- 주요 하위 패키지 ------------------------- #
# 각 하위 패키지는 독립적으로 import하여 사용하는 것을 권장합니다.
# 예: from src.data import DocClsDataset
# 예: from src.models import build_model
# 예: from src.utils import set_seed

# 하위 패키지 목록:
# - data: 데이터셋 및 변환 관련 모듈
# - inference: 추론 실행 관련 모듈  
# - logging: 로깅 시스템 관련 모듈
# - metrics: 평가 메트릭 관련 모듈
# - models: 모델 생성 관련 모듈
# - pipeline: 통합 파이프라인 관련 모듈
# - training: 학습 실행 관련 모듈
# - utils: 공통 유틸리티 관련 모듈

# ------------------------- 패키지 메타데이터 ------------------------- #
__all__ = [                                                 # 명시적으로 노출할 모듈들
    '__version__',                                          # 버전 정보
    '__author__',                                           # 개발팀 정보
    '__description__',                                      # 프로젝트 설명
]
