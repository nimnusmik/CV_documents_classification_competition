# ------------------------- data 패키지 초기화 모듈 ------------------------- #
# 데이터 처리 관련 모듈들을 모아놓은 패키지입니다.
# 데이터셋 클래스(dataset.py)와 이미지 전처리 변환(transforms.py) 등이 포함됩니다.

# ------------------------- 데이터셋 클래스 Import ------------------------- #
from .dataset import (                                      # 데이터셋 관련 클래스들
    DocClsDataset,                                          # 기본 문서 분류 데이터셋
    HighPerfDocClsDataset,                                  # 고성능 문서 분류 데이터셋
    mixup_data                                              # Mixup 데이터 증강 함수
)

# ------------------------- 변환 함수 Import ------------------------- #
from .transforms import (                                   # 이미지 변환 함수들
    build_train_tfms,                                       # 학습용 변환 파이프라인 생성 (기본)
    build_valid_tfms,                                       # 검증용 변환 파이프라인 생성
    build_advanced_train_tfms,                              # 고급 학습용 변환 파이프라인 (베이스라인 기반)
    # Team member's transforms (고성능 변환들)
    build_team_normal_tfms,                                 # 일반 증강
    build_team_hard_tfms,                                   # 하드 증강
    # TTA transforms (Test-Time Augmentation)
    get_essential_tta_transforms,                           # 핵심 TTA 변환 (5가지)
    get_comprehensive_tta_transforms,                       # 포괄적 TTA 변환 (15가지)
    get_tta_transforms_by_type                              # 설정 가능한 TTA 변환
)

# ------------------------- 외부 노출 클래스/함수 정의 ------------------------- #
__all__ = [                                                 # 패키지에서 외부로 노출할 클래스/함수들
    # 데이터셋 클래스
    'DocClsDataset',                                        # 기본 문서 분류 데이터셋
    'HighPerfDocClsDataset',                                # 고성능 문서 분류 데이터셋
    
    # 데이터 증강 함수
    'mixup_data',                                           # Mixup 데이터 증강
    
    # 변환 함수
    'build_train_tfms',                                     # 기본 학습용 변환 파이프라인
    'build_valid_tfms',                                     # 검증용 변환 파이프라인
    'build_advanced_train_tfms',                            # 고급 학습용 변환 파이프라인 (베이스라인)
    
    # Team member's transforms
    'build_team_normal_tfms',                               # 일반 증강
    'build_team_hard_tfms',                                 # 하드 증강
    
    # TTA transforms  
    'get_essential_tta_transforms',                         # 핵심 TTA 변환 (5가지)
    'get_comprehensive_tta_transforms',                     # 포괄적 TTA 변환 (15가지)
    'get_tta_transforms_by_type',                           # 설정 가능한 TTA 변환
]