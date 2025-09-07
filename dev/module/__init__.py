"""
CV Competition 모듈화 패키지

문서 타입 분류 대회를 위한 통합 모듈
- 5-Fold Cross Validation
- EfficientNet-B3 + 앙상블 TTA
- 고급 데이터 증강 및 최적화 기법
"""

__version__ = "1.0.0"
__author__ = "CV Competition Team"

# 주요 모듈 임포트
from .config import Config
from .utils.seed import set_seed
from .experiment.kfold import KFoldExperiment
from .inference.tta import EnsembleTTAPredictor

__all__ = [
    "Config",
    "set_seed", 
    "KFoldExperiment",
    "EnsembleTTAPredictor"
]
