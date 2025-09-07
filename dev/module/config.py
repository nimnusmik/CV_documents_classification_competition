"""
Configuration 모듈
모든 하이퍼파라미터와 설정을 중앙 관리
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    """
    CV Competition 설정 클래스
    모든 하이퍼파라미터와 경로 설정을 통합 관리
    """
    
    # 시드 설정
    seed: int = 42
    
    # 디바이스 설정
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 경로 설정  
    data_path: str = "../data/"
    train_path: str = "../data/train/"
    test_path: str = "../data/test/"
    output_path: str = "../output/"
    
    # 모델 설정
    model_name: str = "efficientnet_b3"  # timm 모델명
    num_classes: int = 17  # 문서 클래스 개수
    pretrained: bool = True
    
    # 학습 설정
    img_size: int = 384  # 이미지 해상도 (384x384)
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 5e-4
    num_workers: int = 30
    
    # 정규화 설정
    label_smoothing: float = 0.2  # Label Smoothing 강도
    mixup_prob: float = 0.3  # Mixup 적용 확률
    mixup_alpha: float = 1.0  # Mixup alpha 파라미터
    grad_clip_norm: float = 1.0  # Gradient Clipping 임계값
    
    # K-Fold 설정
    n_folds: int = 5  # Cross Validation Fold 수
    fold_random_state: int = 42
    
    # TTA 설정
    tta_batch_size: int = 64  # TTA 전용 배치 크기 (메모리 절약)
    confidence_threshold: float = 0.9  # Adaptive TTA 신뢰도 임계값
    
    # Augmentation 설정
    rotation_prob: float = 0.6  # 문서 회전 확률
    blur_prob: float = 0.9  # 블러 효과 확률  
    brightness_prob: float = 0.8  # 밝기 조정 확률
    noise_prob: float = 0.7  # 가우시안 노이즈 확률
    flip_prob: float = 0.5  # 수평 플립 확률
    
    # ImageNet 정규화 파라미터
    imagenet_mean: List[float] = None
    imagenet_std: List[float] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.imagenet_mean is None:
            self.imagenet_mean = [0.485, 0.456, 0.406]
        if self.imagenet_std is None:
            self.imagenet_std = [0.229, 0.224, 0.225]
    
    def get_train_csv_path(self) -> str:
        """학습 CSV 경로 반환"""
        return f"{self.data_path}train.csv"
    
    def get_test_csv_path(self) -> str:
        """테스트 CSV 경로 반환"""
        return f"{self.data_path}sample_submission.csv"
    
    def get_output_csv_path(self) -> str:
        """출력 CSV 경로 반환"""
        return f"{self.output_path}choice.csv"


class OptunaConfig(Config):
    """
    Optuna 하이퍼파라미터 튜닝용 설정 확장 클래스
    """
    
    # Optuna 설정
    use_optuna: bool = False
    n_trials: int = 10
    optuna_timeout: Optional[int] = None  # 초 단위 (None이면 무제한)
    quick_cv_folds: int = 3  # 빠른 튜닝을 위한 적은 fold 수
    quick_epochs: int = 2  # 빠른 튜닝을 위한 적은 epoch 수
    
    # 튜닝 범위
    lr_range: tuple = (1e-5, 1e-2)
    batch_size_choices: List[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.batch_size_choices is None:
            self.batch_size_choices = [32, 64, 128]


# 기본 설정 인스턴스
config = Config()
optuna_config = OptunaConfig()
