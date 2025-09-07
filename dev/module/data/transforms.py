"""
데이터 변환(Augmentation) 정의
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List
from ..config import Config


def get_train_transforms(config: Config) -> A.Compose:
    """
    훈련용 데이터 변환 파이프라인
    
    Features:
        - 비율 보존 리사이징 (문서 특화)
        - 정확한 90도 단위 회전 (문서 특성 고려)
        - 테스트 특화 노이즈/블러 강화
        - ImageNet 정규화
    
    Args:
        config (Config): 설정 객체
        
    Returns:
        A.Compose: 훈련용 변환 파이프라인
    """
    
    return A.Compose([
        # 1. 비율 보존 리사이징 (핵심 개선)
        A.LongestMaxSize(max_size=config.img_size),
        A.PadIfNeeded(
            min_height=config.img_size, 
            min_width=config.img_size,
            border_mode=0,  # 상수값으로 패딩 (검은색)
            value=0
        ),
        
        # 2. 문서 특화 회전 (정확한 90도 배수만)
        A.OneOf([
            A.Rotate(limit=[90, 90], p=1.0),    # 90도 회전
            A.Rotate(limit=[180, 180], p=1.0),  # 180도 회전  
            A.Rotate(limit=[270, 270], p=1.0),  # 270도 회전
        ], p=config.rotation_prob),
        
        # 3. 테스트 특화 강화 증강 (블러)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),   # 움직임 블러
            A.GaussianBlur(blur_limit=7, p=1.0), # 가우시안 블러
        ], p=config.blur_prob),
        
        # 4. 밝기/대비 조정 (문서 조명 변화 대응)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=config.brightness_prob
        ),
        
        # 5. 가우시안 노이즈 (스캔/촬영 노이즈 시뮬레이션)
        A.GaussNoise(
            var_limit=(30.0, 100.0),  # 노이즈 강도 범위
            p=config.noise_prob
        ),
        
        # 6. 수평 플립 (일반적인 기하학적 변환)
        A.HorizontalFlip(p=config.flip_prob),
        
        # 7. 정규화 (ImageNet 평균/표준편차)
        A.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std
        ),
        
        # 8. PyTorch 텐서로 변환
        ToTensorV2(),
    ])


def get_test_transforms(config: Config) -> A.Compose:
    """
    테스트/검증용 데이터 변환 파이프라인
    
    Features:
        - 증강 없이 기본 전처리만
        - 훈련과 동일한 리사이징/정규화
    
    Args:
        config (Config): 설정 객체
        
    Returns:
        A.Compose: 테스트용 변환 파이프라인
    """
    
    return A.Compose([
        # 비율 보존 리사이징
        A.LongestMaxSize(max_size=config.img_size),
        A.PadIfNeeded(
            min_height=config.img_size,
            min_width=config.img_size,
            border_mode=0,
            value=0
        ),
        
        # 정규화
        A.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std
        ),
        
        # 텐서 변환
        ToTensorV2(),
    ])


def get_tta_transforms(config: Config) -> List[A.Compose]:
    """
    Test Time Augmentation용 변환 리스트
    
    Features:
        - 핵심 TTA 변형들만 선별 (성능 vs 속도 최적화)
        - 원본 + 회전 3개 + 밝기 1개 = 총 5개 변형
        - 문서 특성에 최적화된 변형들
    
    Args:
        config (Config): 설정 객체
        
    Returns:
        List[A.Compose]: TTA 변환 리스트
    """
    
    base_transforms = [
        A.LongestMaxSize(max_size=config.img_size),
        A.PadIfNeeded(
            min_height=config.img_size,
            min_width=config.img_size,
            border_mode=0,
            value=0
        )
    ]
    
    tta_transforms = []
    
    # 1. 원본 (증강 없음)
    tta_transforms.append(A.Compose(
        base_transforms + [
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]
    ))
    
    # 2-4. 90도 단위 회전들
    rotation_angles = [90, 180, -90]
    for angle in rotation_angles:
        tta_transforms.append(A.Compose(
            base_transforms + [
                A.Rotate(limit=[angle, angle], p=1.0),
                A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
                ToTensorV2()
            ]
        ))
    
    # 5. 밝기 개선 (어두운 문서 대응)
    tta_transforms.append(A.Compose(
        base_transforms + [
            A.RandomBrightnessContrast(
                brightness_limit=[0.3, 0.3],  # 고정된 밝기 증가
                contrast_limit=[0.3, 0.3],    # 고정된 대비 증가
                p=1.0
            ),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]
    ))
    
    return tta_transforms


def get_heavy_tta_transforms(config: Config) -> List[A.Compose]:
    """
    더 많은 TTA 변형 (고성능 추구시 사용)
    
    Features:
        - 15개 변형으로 확장
        - 다양한 노이즈/블러 조합
        - 계산 시간 증가하지만 성능 향상 기대
    
    Args:
        config (Config): 설정 객체
        
    Returns:
        List[A.Compose]: 확장된 TTA 변환 리스트
    """
    
    # 기본 TTA 변형들 가져오기
    transforms = get_tta_transforms(config)
    
    base_transforms = [
        A.LongestMaxSize(max_size=config.img_size),
        A.PadIfNeeded(
            min_height=config.img_size,
            min_width=config.img_size,
            border_mode=0,
            value=0
        )
    ]
    
    # 추가 변형들
    additional_transforms = [
        # 블러 변형들
        A.Compose(base_transforms + [
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]),
        
        A.Compose(base_transforms + [
            A.MotionBlur(blur_limit=5, p=1.0),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]),
        
        # 노이즈 변형들
        A.Compose(base_transforms + [
            A.GaussNoise(var_limit=(50.0, 100.0), p=1.0),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]),
        
        # 수평 플립 + 회전 조합
        A.Compose(base_transforms + [
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]),
        
        A.Compose(base_transforms + [
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=[90, 90], p=1.0),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ]),
        
        # 대비 조정 변형들
        A.Compose(base_transforms + [
            A.RandomBrightnessContrast(
                brightness_limit=[-0.2, -0.2],  # 어둡게
                contrast_limit=[0.2, 0.2],      # 대비 증가
                p=1.0
            ),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2()
        ])
    ]
    
    return transforms + additional_transforms


def create_custom_transforms(
    img_size: int = 384,
    rotation_prob: float = 0.6,
    blur_prob: float = 0.9,
    brightness_prob: float = 0.8,
    noise_prob: float = 0.7,
    flip_prob: float = 0.5,
    mean: List[float] = None,
    std: List[float] = None
) -> A.Compose:
    """
    커스텀 변환 생성 함수
    
    Args:
        img_size: 이미지 크기
        rotation_prob: 회전 확률
        blur_prob: 블러 확률
        brightness_prob: 밝기 조정 확률
        noise_prob: 노이즈 확률
        flip_prob: 플립 확률
        mean: 정규화 평균값
        std: 정규화 표준편차
        
    Returns:
        A.Compose: 커스텀 변환 파이프라인
    """
    
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            value=0
        ),
        
        A.OneOf([
            A.Rotate(limit=[90, 90], p=1.0),
            A.Rotate(limit=[180, 180], p=1.0),
            A.Rotate(limit=[270, 270], p=1.0),
        ], p=rotation_prob),
        
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=blur_prob),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=brightness_prob
        ),
        
        A.GaussNoise(var_limit=(30.0, 100.0), p=noise_prob),
        A.HorizontalFlip(p=flip_prob),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# Alias for backward compatibility
get_val_transforms = get_test_transforms

if __name__ == "__main__":
    # 테스트 코드
    import numpy as np
    from PIL import Image
    
    # 설정 생성
    config = Config()
    
    # 가상 이미지 생성 (문서 형태)
    img = np.random.randint(0, 255, (600, 400, 3), dtype=np.uint8)  # 세로가 긴 문서
    
    print("=== Transform Test ===")
    print(f"Original image shape: {img.shape}")
    
    # 각 변환 테스트
    transforms = {
        "Train": get_train_transforms(config),
        "Test": get_test_transforms(config),
    }
    
    for name, transform in transforms.items():
        transformed = transform(image=img)
        result = transformed['image']
        print(f"{name} transform result shape: {result.shape}")
    
    # TTA 변환 테스트
    tta_transforms = get_tta_transforms(config)
    print(f"\nTTA transforms count: {len(tta_transforms)}")
    
    for i, transform in enumerate(tta_transforms):
        transformed = transform(image=img)
        result = transformed['image']
        print(f"TTA {i+1} result shape: {result.shape}")
    
    # Heavy TTA 테스트
    heavy_tta = get_heavy_tta_transforms(config)
    print(f"\nHeavy TTA transforms count: {len(heavy_tta)}")
    
    print("\n=== Memory Usage Estimation ===")
    
    # 메모리 사용량 추정
    def estimate_memory_usage(img_size: int, batch_size: int, num_transforms: int):
        # 단일 이미지: H x W x C x 4bytes (float32)
        single_img_bytes = img_size * img_size * 3 * 4
        # 배치 메모리
        batch_memory = single_img_bytes * batch_size * num_transforms
        # GPU 오버헤드 고려 (약 2배)
        gpu_memory = batch_memory * 2
        
        # GB 단위로 변환
        gb = gpu_memory / (1024**3)
        return gb
    
    standard_memory = estimate_memory_usage(384, 64, 5)
    heavy_memory = estimate_memory_usage(384, 64, len(heavy_tta))
    
    print(f"Standard TTA memory usage: {standard_memory:.2f} GB")
    print(f"Heavy TTA memory usage: {heavy_memory:.2f} GB")
    
    # 권장사항
    print(f"\n=== Recommendations ===")
    if heavy_memory > 8:
        print("⚠️  Heavy TTA requires >8GB GPU memory")
        print("💡 Consider reducing batch size or using standard TTA")
    else:
        print("✅ Heavy TTA should work with current settings")
