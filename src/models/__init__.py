# src/models/__init__.py
"""
딥러닝 모델 패키지

이 패키지는 컴퓨터 비전 경진대회를 위한 딥러닝 모델 생성 기능을 제공합니다.
timm 라이브러리를 기반으로 한 다양한 사전 훈련된 모델들을 지원하며,
모델 생성, 설정, 추천 기능을 포함합니다.

주요 기능:
- timm 기반 모델 생성 (build_model)
- 고성능 추천 모델 조회 (get_recommended_model) 
- 다양한 pooling 및 dropout 설정 지원
- Swin Transformer, ConvNeXt, EfficientNet 등 지원
"""

# ==================== 패키지 정보 ==================== #
__version__ = "1.0.0"                                      # 패키지 버전
__author__ = "Computer Vision Competition Team"             # 작성자 정보

# ==================== 핵심 모델 빌더 임포트 ==================== #
# 모델 생성 관련 핵심 함수들 임포트
try:                                                        # 예외 처리 시작
    from .build import build_model, get_recommended_model   # 모델 빌드 함수들 임포트
except ImportError as e:                                    # 임포트 실패 시
    # 개발 중일 때는 경고만 출력하고 계속 진행
    import warnings                                         # 경고 모듈 임포트
    warnings.warn(f"모델 빌드 함수 임포트 실패: {e}")         # 경고 메시지 출력

# ==================== 공개 API 정의 ==================== #
# 패키지에서 공개할 함수들 정의
__all__ = [                                                 # 공개 API 리스트
    "build_model",                                          # 모델 생성 함수
    "get_recommended_model",                                # 추천 모델 조회 함수
    "__version__",                                          # 버전 정보
    "__author__",                                           # 작성자 정보
]