"""
재현성을 위한 시드 고정 유틸리티
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    재현 가능한 실험을 위해 모든 랜덤 시드를 고정
    
    Args:
        seed (int): 설정할 시드 값 (기본값: 42)
    
    Note:
        - Python random, NumPy, PyTorch의 시드를 모두 고정
        - CUDA 재현성을 위한 추가 설정 포함
        - cudnn.benchmark=True로 성능 최적화 유지
    """
    
    # Python random 시드 고정
    random.seed(seed)
    
    # NumPy 시드 고정  
    np.random.seed(seed)
    
    # PyTorch 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경
    
    # 환경변수 설정 (Python 해시 시드 고정)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDA 재현성 설정
    # 주의: deterministic=True는 성능을 저하시킬 수 있음
    # torch.backends.cudnn.deterministic = True
    
    # 성능 최적화를 위해 benchmark 모드 활성화
    # 입력 크기가 고정적일 때 성능 향상
    torch.backends.cudnn.benchmark = True
    
    print(f"Random seed set to: {seed}")


def get_current_seed() -> dict:
    """
    현재 설정된 시드 값들을 조회
    
    Returns:
        dict: 각 라이브러리별 현재 시드 상태
    """
    
    return {
        "python_random_state": random.getstate()[1][0],  # 첫 번째 상태값만 표시
        "numpy_random_state": np.random.get_state()[1][0],  # 첫 번째 상태값만 표시  
        "torch_random_state": torch.initial_seed(),
        "cuda_random_state": torch.cuda.initial_seed() if torch.cuda.is_available() else None,
        "pythonhashseed": os.environ.get('PYTHONHASHSEED', 'not set'),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark
    }


if __name__ == "__main__":
    # 테스트 코드
    print("Before setting seed:")
    print(get_current_seed())
    
    set_seed(42)
    
    print("\nAfter setting seed:")
    print(get_current_seed())
    
    # 재현성 테스트
    print("\nReproducibility test:")
    for i in range(3):
        set_seed(42)
        print(f"PyTorch random: {torch.randn(1).item():.6f}")
        print(f"NumPy random: {np.random.rand():.6f}")
        print(f"Python random: {random.random():.6f}")
        print("---")
